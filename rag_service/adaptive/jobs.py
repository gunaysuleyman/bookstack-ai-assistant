from typing import Callable, Optional, Sequence

from adaptive.contracts import PageDocument, SyncJob
from adaptive.indexer import Indexer, plan_reconciliation
from adaptive.store import StateStore
from sync import SyncError


PageLoader = Callable[[int], Optional[PageDocument]]


def document_from_payload(payload: dict, generation: int) -> Optional[PageDocument]:
    if not payload or "markdown" not in payload:
        return None
    return PageDocument(
        page_id=int(payload["page_id"]),
        name=str(payload.get("name") or ""),
        markdown=str(payload.get("markdown") or ""),
        book_id=int(payload.get("book_id") or 0),
        book_name=str(payload.get("book_name") or "General Library"),
        chapter_id=int(payload.get("chapter_id") or 0),
        chapter_name=str(payload.get("chapter_name") or "General Chapter"),
        shelf_names=list(payload.get("shelf_names") or []),
        tags_str=str(payload.get("tags_str") or ""),
        url=str(payload.get("url") or ""),
        updated_at=str(payload.get("updated_at") or ""),
        generation=generation,
    )


def next_generation(store: StateStore, page_id: int) -> int:
    state = store.get_page_state(page_id)
    current = int(state["generation"]) if state else 0
    return current + 1


def apply_page_job(job: SyncJob, indexer: Indexer, loader: PageLoader, legacy=None) -> str:
    """Index a webhook job. A page id without markdown is loaded; load errors propagate."""
    if job.event == "page_delete":
        if legacy is not None:
            legacy.delete_page(job.page_id)
        indexer.delete(job.page_id, job.generation)
        return "deleted"
    if job.event != "page_upsert":
        return "ignored"
    document = document_from_payload(job.payload, job.generation)
    if document is None:
        loaded = loader(job.page_id)
        if loaded is None:
            if legacy is not None:
                legacy.delete_page(job.page_id)
            indexer.delete(job.page_id, job.generation)
            return "missing"
        document = loaded.model_copy(update={"generation": job.generation})
    if legacy is not None:
        legacy.index_loaded_page(document)
    indexer.upsert(document, generation=job.generation)
    return "upserted"


def apply_reconcile(
    store: StateStore,
    indexer: Indexer,
    loader: PageLoader,
    list_remote: Callable[[], Sequence[dict]],
    legacy=None,
) -> dict:
    """Compare remote stubs with the adaptive index. A failed read does not delete local pages."""
    remote = list(list_remote())
    local_ids = store.list_page_ids()
    remote_pages = []
    for stub in remote:
        page_id = int(stub["id"])
        state = store.get_page_state(page_id)
        local_updated = str(state["source_updated_at"] or "") if state else ""
        updated = str(stub.get("updated_at") or "")
        model_mismatch = bool(state) and state["embedding_model_id"] != indexer.settings.embedding_model_id
        schema_mismatch = bool(state) and state["chunk_schema_version"] != indexer.settings.chunk_schema_version
        changed = page_id not in local_ids or local_updated != updated or model_mismatch or schema_mismatch
        remote_pages.append({"id": page_id, "updated_at": updated, "changed": changed})
    plan = plan_reconciliation(local_ids, remote_pages, scan_complete=True, read_errors=0)
    read_errors = 0
    for page_id in plan["upserts"]:
        try:
            loaded = loader(page_id)
        except Exception:
            read_errors += 1
            continue
        generation = next_generation(store, page_id)
        if loaded is None:
            if legacy is not None:
                legacy.delete_page(page_id)
            indexer.delete(page_id, generation)
            continue
        document = loaded.model_copy(update={"generation": generation})
        if legacy is not None:
            legacy.index_loaded_page(document)
        indexer.upsert(document, generation=generation)
    if read_errors:
        raise SyncError(f"{read_errors} page reads failed; missing pages were not deleted")
    for page_id in plan["deletes"]:
        generation = next_generation(store, page_id)
        if legacy is not None:
            legacy.delete_page(page_id)
        indexer.delete(page_id, generation)
    return plan
