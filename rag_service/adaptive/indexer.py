import hashlib
import json
import logging
import uuid
from typing import List, Optional, Sequence

from adaptive.chunking import chunk_document
from adaptive.config import Settings
from adaptive.contracts import PageDocument
from adaptive.store import StateStore
from adaptive.vector_index import VectorIndex

logger = logging.getLogger("AdaptiveIndex")


def content_hash(markdown: str) -> str:
    return hashlib.sha256(markdown.encode("utf-8")).hexdigest()


def metadata_hash(page: PageDocument) -> str:
    payload = {
        "name": page.name,
        "book_id": page.book_id,
        "book_name": page.book_name,
        "chapter_id": page.chapter_id,
        "chapter_name": page.chapter_name,
        "shelf_names": list(page.shelf_names),
        "tags_str": page.tags_str,
        "url": page.url,
    }
    raw = json.dumps(payload, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


class Indexer:
    def __init__(self, store: StateStore, vectors: VectorIndex, settings: Settings):
        self.store = store
        self.vectors = vectors
        self.settings = settings
        self.fail_fts = False
        self.stop_after: Optional[str] = None

    def upsert(self, page: PageDocument, generation: Optional[int] = None) -> str:
        generation = page.generation if generation is None else generation
        state = self.store.get_page_state(page.page_id)
        if state and int(state["generation"]) > generation:
            return "stale"
        if state and state["status"] == "tombstone" and int(state["generation"]) >= generation:
            return "tombstoned"
        body_hash = content_hash(page.markdown)
        meta_hash = metadata_hash(page)
        unchanged = (
            state
            and state["status"] == "published"
            and state["content_hash"] == body_hash
            and state["metadata_hash"] == meta_hash
            and state["chunk_schema_version"] == self.settings.chunk_schema_version
            and state["embedding_model_id"] == self.settings.embedding_model_id
        )
        if unchanged:
            return "unchanged"
        metadata_only = (
            state
            and state["status"] == "published"
            and state["content_hash"] == body_hash
            and state["chunk_schema_version"] == self.settings.chunk_schema_version
            and state["embedding_model_id"] == self.settings.embedding_model_id
        )
        if metadata_only:
            self.store.update_metadata_only(page, meta_hash, generation)
            return "metadata_only"

        revision_id = uuid.uuid4().hex
        parents, children = chunk_document(
            page,
            child_tokens=self.settings.child_tokens,
            embed_max_tokens=self.settings.embed_max_tokens,
        )
        for parent in parents:
            parent.parent_id = f"{revision_id}_{parent.parent_id}"
        for child in children:
            child.parent_id = f"{revision_id}_{child.parent_id}"
            child.chunk_id = f"{revision_id}_{child.chunk_id}"
        self.store.stage_revision(
            revision_id=revision_id,
            page=page,
            parents=parents,
            children=children,
            content_hash=body_hash,
            metadata_hash=meta_hash,
            generation=generation,
        )
        metadatas = []
        for child in children:
            metadatas.append(
                {
                    "page_id": int(page.page_id),
                    "revision_id": revision_id,
                    "parent_id": child.parent_id,
                    "name": page.name,
                    "url": page.url,
                    "book_name": page.book_name,
                    "shelf_name": page.shelf_label(),
                    "heading": child.heading,
                    "index_version": self.settings.index_version,
                }
            )
        self.vectors.add(
            [child.chunk_id for child in children],
            [child.embed_text for child in children],
            metadatas,
        )
        if self.stop_after == "vectors":
            return "stopped_after_vectors"
        try:
            if self.fail_fts:
                raise RuntimeError("lexical write failed")
            self.store.write_fts(revision_id)
        except Exception:
            ids = self.store.discard_revision(revision_id)
            self.vectors.delete(ids)
            logger.warning("Lexical write failed for page %s; active revision kept", page.page_id)
            raise
        self.store.mark_revision(revision_id, "indexed")
        if self.stop_after == "indexed":
            return "stopped_after_indexed"
        old_ids = self.store.publish(
            page,
            revision_id,
            body_hash,
            meta_hash,
            generation,
            self.settings.chunk_schema_version,
            self.settings.embedding_model_id,
        )
        if old_ids is None:
            ids = self.store.discard_revision(revision_id)
            self.vectors.delete(ids)
            return "stale"
        self.vectors.delete(old_ids)
        self.store.clear_gc(old_ids)
        return "published"

    def delete(self, page_id: int, generation: int) -> str:
        ids = self.store.tombstone(page_id, generation)
        if not ids and self.store.get_page_state(page_id) and int(self.store.get_page_state(page_id)["generation"]) > generation:
            return "stale"
        self.vectors.delete(ids)
        self.store.clear_gc(ids)
        return "tombstone"

    def recover(self) -> List[str]:
        actions = []
        for revision in self.store.incomplete_revisions():
            revision_id = revision["revision_id"]
            if revision["state"] == "prepared":
                ids = self.store.discard_revision(revision_id)
                self.vectors.delete(ids)
                actions.append(f"discarded:{revision_id}")
            elif revision["state"] == "indexed":
                page = PageDocument.model_validate_json(revision["page_json"])
                old_ids = self.store.publish(
                    page,
                    revision_id,
                    revision["content_hash"],
                    revision["metadata_hash"],
                    int(revision["generation"]),
                    self.settings.chunk_schema_version,
                    self.settings.embedding_model_id,
                )
                if old_ids is None:
                    ids = self.store.discard_revision(revision_id)
                    self.vectors.delete(ids)
                    actions.append(f"discarded:{revision_id}")
                else:
                    self.vectors.delete(old_ids)
                    self.store.clear_gc(old_ids)
                    actions.append(f"published:{revision_id}")
        gc_ids = self.store.gc_ids()
        if gc_ids:
            self.vectors.delete(gc_ids)
            self.store.clear_gc(gc_ids)
            actions.append(f"gc:{len(gc_ids)}")
        return actions

    def active_text(self, page_id: int) -> str:
        state = self.store.get_page_state(page_id)
        if state is None or not state["active_revision"]:
            return ""
        rows = self.store.page_chunks(page_id, state["active_revision"])
        return "\n".join(row["body"] for row in rows)


def plan_reconciliation(
    local_ids: Sequence[int],
    remote_pages: Sequence[dict],
    scan_complete: bool,
    read_errors: int,
) -> dict:
    remote_ids = []
    changed = []
    for page in remote_pages:
        remote_ids.append(int(page["id"]))
        if page.get("changed"):
            changed.append(int(page["id"]))
    deletes = []
    if scan_complete and read_errors == 0:
        remote_set = set(remote_ids)
        deletes = [page_id for page_id in local_ids if page_id not in remote_set]
    return {"upserts": changed, "deletes": deletes, "scan_complete": scan_complete and read_errors == 0}
