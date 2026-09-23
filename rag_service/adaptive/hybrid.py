import logging
import re
from typing import Dict, Iterable, List, Optional, Sequence

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
        if not scope.can_use_ai:
            return []
        if scope.allowed_page_ids is not None and len(scope.allowed_page_ids) == 0:
            return []
        active_map = active if active is not None else self.store.active_revision_map()
        allowed = None if scope.allows_all() else set(scope.allowed_page_ids or [])
        try:
            vector_ids = self._vector_ranks(query, allowed, limit)
        except Exception:
            logger.warning("Vector search unavailable; using lexical search")
            vector_ids = []
        lexical_ids = self._lexical_ranks(query, allowed, active_map, limit)
        fused = reciprocal_rank_fusion([vector_ids, lexical_ids])
        if not fused:
            return []
        ordered = sorted(fused, key=lambda chunk_id: fused[chunk_id], reverse=True)
        records = self.store.get_chunks(ordered)
        vector_rank = {chunk_id: index for index, chunk_id in enumerate(vector_ids, start=1)}
        lexical_rank = {chunk_id: index for index, chunk_id in enumerate(lexical_ids, start=1)}
        candidates = []
        for chunk_id in ordered:
            row = records.get(chunk_id)
            if row is None:
                continue
            page_id = int(row["page_id"])
            revision_id = str(row["revision_id"])
            if active_map.get(page_id) != revision_id:
                continue
            if allowed is not None and page_id not in allowed:
                continue
            state = self.store.get_page_state(page_id)
            candidates.append(
                RetrievalCandidate(
                    page_id=page_id,
                    chunk_id=chunk_id,
                    parent_id=str(row["parent_id"] or ""),
                    revision_id=revision_id,
                    heading=str(row["heading"] or ""),
                    text=str(row["body"] or ""),
                    title=str(state["title"] or "") if state else "",
                    url=str(state["url"] or "") if state else "",
                    vector_rank=vector_rank.get(chunk_id),
                    lexical_rank=lexical_rank.get(chunk_id),
                    fusion_score=fused[chunk_id],
                    channel="hybrid",
                )
            )
            if len(candidates) >= limit:
                break
        return candidates

    def _vector_ranks(self, query: str, allowed: Optional[set], limit: int) -> List[str]:
        fetch = max(limit, 1)
        if allowed is None:
            rows = self.vectors.query(query, fetch)
            return [row["chunk_id"] for row in rows]
        ids: List[str] = []
        ranked = []
        allowed_list = list(allowed)
        for start in range(0, len(allowed_list), self.acl_batch):
            batch = allowed_list[start : start + self.acl_batch]
            rows = self.vectors.query(query, fetch, where={"page_id": {"$in": batch}})
            ranked.extend(rows)
        ranked.sort(key=lambda row: row.get("distance") if row.get("distance") is not None else 999)
        for row in ranked:
            chunk_id = row["chunk_id"]
            if chunk_id not in ids:
                ids.append(chunk_id)
            if len(ids) >= fetch:
                break
        return ids

    def _lexical_ranks(self, query: str, allowed: Optional[set], active_map: Dict[int, str], limit: int) -> List[str]:
        page_ids = None if allowed is None else list(allowed)
        rows = self.store.lexical_search(fts_match(query), limit, page_ids=page_ids)
        ids = []
        for row in rows:
            try:
                page_id = int(row["page_id"])
            except (TypeError, ValueError):
                continue
            if active_map.get(page_id) != str(row["revision_id"]):
                continue
            chunk_id = str(row["chunk_id"])
            if chunk_id not in ids:
                ids.append(chunk_id)
            if len(ids) >= limit:
                break
        return ids


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
