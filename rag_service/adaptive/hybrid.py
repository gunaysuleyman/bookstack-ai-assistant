import json
import logging
import re
from typing import Dict, Iterable, List, Optional, Sequence

from adaptive.catalog import real_name, real_shelves
from adaptive.contracts import AuthorizationScope, RetrievalCandidate
from adaptive.router import fold
from adaptive.store import StateStore
from adaptive.vector_index import VectorIndex


_FTS_TOKEN = re.compile(r"[\w]+", re.UNICODE)
logger = logging.getLogger("AdaptiveHybrid")


def fts_match(text: str) -> str:
    tokens = []
    for token in _FTS_TOKEN.findall(text):
        if len(token) < 2:
            continue
        if token not in tokens:
            tokens.append(token)
    if not tokens:
        return ""
    return " OR ".join(f'"{token}"' for token in tokens[:24])


def reciprocal_rank_fusion(rankings: Sequence[Sequence[str]], k: int = 60) -> Dict[str, float]:
    scores: Dict[str, float] = {}
    for ranking in rankings:
        for index, chunk_id in enumerate(ranking, start=1):
            scores[chunk_id] = scores.get(chunk_id, 0.0) + 1.0 / (k + index)
    return scores


class HybridSearcher:
    """Vector + FTS5 search fused with reciprocal rank fusion.

    Each channel returns a pool wider than `limit`. The top `CHANNEL_QUOTA`
    hits of each channel are kept in the result even when RRF ranks them
    lower, so an exact lexical match (a code, a name) cannot be pushed out by
    passages that are mediocre in both channels.
    """

    CHANNEL_QUOTA = 2
    MAX_VECTOR_OVERFETCH = 500

    def __init__(self, store: StateStore, vectors: VectorIndex, acl_batch: int = 200):
        self.store = store
        self.vectors = vectors
        self.acl_batch = acl_batch

    def search(
        self,
        query: str,
        scope: AuthorizationScope,
        limit: int,
        active: Optional[Dict[int, str]] = None,
    ) -> List[RetrievalCandidate]:
        if not scope.can_use_ai or limit <= 0:
            return []
        if scope.allowed_page_ids is not None and len(scope.allowed_page_ids) == 0:
            return []
        allowed = None if scope.allows_all() else set(scope.allowed_page_ids or [])
        if allowed is not None and not allowed:
            return []
        pool = max(limit * 2, limit + 8)
        try:
            vector_ids = self._vector_ranks(query, allowed, pool)
        except Exception:
            logger.warning("Vector search unavailable; using lexical search")
            vector_ids = []
        lexical_ids = self._lexical_ranks(query, allowed, active, pool)
        fused = reciprocal_rank_fusion([vector_ids, lexical_ids])
        if not fused:
            return []
        ordered = sorted(fused, key=lambda chunk_id: fused[chunk_id], reverse=True)
        records = self.store.search_rows(ordered)
        vector_rank = {chunk_id: index for index, chunk_id in enumerate(vector_ids, start=1)}
        lexical_rank = {chunk_id: index for index, chunk_id in enumerate(lexical_ids, start=1)}
        valid: List[RetrievalCandidate] = []
        for chunk_id in ordered:
            row = records.get(chunk_id)
            if row is None:
                continue
            page_id = int(row["page_id"])
            revision_id = str(row["revision_id"])
            if active is not None and active.get(page_id) != revision_id:
                continue
            if allowed is not None and page_id not in allowed:
                continue
            valid.append(
                RetrievalCandidate(
                    page_id=page_id,
                    chunk_id=chunk_id,
                    parent_id=str(row["parent_id"] or ""),
                    revision_id=revision_id,
                    heading=str(row["heading"] or ""),
                    text=str(row["body"] or ""),
                    title=str(row["title"] or ""),
                    url=str(row["url"] or ""),
                    book_name=real_name(row["book_name"]),
                    chapter_name=real_name(row["chapter_name"]),
                    shelf_names=real_shelves(_json_list(row["shelf_names"])),
                    vector_rank=vector_rank.get(chunk_id),
                    lexical_rank=lexical_rank.get(chunk_id),
                    fusion_score=fused[chunk_id],
                    channel="hybrid",
                )
            )
        return _with_channel_quota(valid, limit, self.CHANNEL_QUOTA)

    def _vector_ranks(self, query: str, allowed: Optional[set], limit: int) -> List[str]:
        fetch = max(limit, 1)
        if allowed is None:
            return [row["chunk_id"] for row in self.vectors.query(query, fetch)]
        if len(allowed) <= self.acl_batch:
            rows = self.vectors.query(query, fetch, where={"page_id": {"$in": sorted(allowed)}})
            return _ranked_ids(rows, fetch)
        # A filtered HNSW query gets slower as the $in list grows. When the
        # user can see a large share of the index, over-fetch unfiltered and
        # drop the rows they cannot see instead.
        share = len(allowed) / max(1, self.store.published_count())
        overfetch = int(fetch / max(share, 1e-6) * 1.5) + 1
        if overfetch <= self.MAX_VECTOR_OVERFETCH:
            rows = self.vectors.query(query, overfetch)
            visible = [row for row in rows if _row_page_id(row) in allowed]
            if len(visible) >= fetch or len(rows) < overfetch:
                return _ranked_ids(visible, fetch)
        rows = self.vectors.query(query, fetch, where={"page_id": {"$in": sorted(allowed)}})
        return _ranked_ids(rows, fetch)

    def _lexical_ranks(self, query: str, allowed: Optional[set], active: Optional[Dict[int, str]], limit: int) -> List[str]:
        page_ids = None if allowed is None else sorted(allowed)
        rows = self.store.lexical_search(fts_match(query), limit, page_ids=page_ids)
        ids = []
        for row in rows:
            try:
                page_id = int(row["page_id"])
            except (TypeError, ValueError):
                continue
            if active is not None and active.get(page_id) != str(row["revision_id"]):
                continue
            chunk_id = str(row["chunk_id"])
            if chunk_id not in ids:
                ids.append(chunk_id)
            if len(ids) >= limit:
                break
        return ids


def _with_channel_quota(candidates: List[RetrievalCandidate], limit: int, quota: int) -> List[RetrievalCandidate]:
    """First `limit` by fusion order, but always include each channel's top `quota` hits."""
    if len(candidates) <= limit:
        return candidates
    required = set()
    for rank_of in (lambda item: item.vector_rank, lambda item: item.lexical_rank):
        ranked = sorted((item for item in candidates if rank_of(item) is not None), key=rank_of)
        required.update(item.chunk_id for item in ranked[:quota])
    chosen = [item for item in candidates if item.chunk_id in required][:limit]
    chosen_ids = {item.chunk_id for item in chosen}
    for item in candidates:
        if len(chosen) >= limit:
            break
        if item.chunk_id not in chosen_ids:
            chosen.append(item)
            chosen_ids.add(item.chunk_id)
    order = {item.chunk_id: index for index, item in enumerate(candidates)}
    return sorted(chosen, key=lambda item: order[item.chunk_id])


def _row_page_id(row: dict) -> int:
    try:
        return int((row.get("metadata") or {}).get("page_id"))
    except (TypeError, ValueError):
        return -1


def _ranked_ids(rows: List[dict], fetch: int) -> List[str]:
    ranked = sorted(rows, key=lambda row: row.get("distance") if row.get("distance") is not None else 999)
    ids: List[str] = []
    for row in ranked:
        if row["chunk_id"] not in ids:
            ids.append(row["chunk_id"])
        if len(ids) >= fetch:
            break
    return ids


def _json_list(raw) -> List[str]:
    try:
        value = json.loads(raw or "[]")
    except (TypeError, ValueError):
        return []
    return [str(item) for item in value] if isinstance(value, list) else []


def dedupe_candidates(candidates: Sequence[RetrievalCandidate]) -> List[RetrievalCandidate]:
    kept: List[RetrievalCandidate] = []
    for candidate in candidates:
        tokens = set(fold(candidate.text).split())
        duplicate = False
        for existing in kept:
            if existing.page_id == candidate.page_id and existing.heading == candidate.heading:
                other = set(fold(existing.text).split())
                if tokens and other:
                    overlap = len(tokens & other) / len(tokens | other)
                    if overlap >= 0.8:
                        duplicate = True
                        break
        if not duplicate:
            kept.append(candidate)
    return kept
