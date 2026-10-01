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


# Container jobs share the sync_jobs table with page jobs, whose key is the
# page id. Negative keys in separate ranges keep a book, chapter, or shelf job
# from superseding (or taking the generation of) a page job with the same id.
_CONTAINER_KEY_BASE = {"book": 0, "chapter": 1_000_000_000, "shelf": 2_000_000_000}


def container_job_key(kind: str, item_id: int) -> int:
    return -(_CONTAINER_KEY_BASE[kind] + int(item_id))


PageEnqueuer = Callable[[int], object]


def apply_book_refresh(store: StateStore, book_id: int, fetch_book: Callable[[int], Optional[dict]], enqueue_page: PageEnqueuer) -> dict:
    """Apply a book change. `fetch_book` returns {"book_name", "shelf_names"} or None when the book is gone.

    A shelf-only change is relabelled in place. A renamed or deleted book
    re-queues its pages: the book name is embedded, and a page load of a
    deleted book returns 404, which tombstones the page.
    """
    pages = store.catalog_page_ids_for(book_id=book_id)
    info = fetch_book(book_id)
    if info is None:
        for page_id in pages:
            enqueue_page(page_id)
        return {"action": "requeued", "reason": "book_missing", "pages": len(pages)}
    if store.book_pages_to_reembed(book_id, info["book_name"]):
        for page_id in pages:
            enqueue_page(page_id)
        return {"action": "requeued", "reason": "book_renamed", "pages": len(pages)}
    updated = store.relabel_book(book_id, info["book_name"], info["shelf_names"])
    return {"action": "relabelled", "pages": updated}


def apply_chapter_refresh(
    store: StateStore, chapter_id: int, fetch_chapter: Callable[[int], Optional[dict]], enqueue_page: PageEnqueuer
) -> dict:
    """Re-queue every page that is or was in the chapter. `fetch_chapter` returns {"page_ids": [...]} or None."""
    local = set(store.catalog_page_ids_for(chapter_id=chapter_id))
    info = fetch_chapter(chapter_id)
    remote = set(int(page_id) for page_id in (info or {}).get("page_ids") or [])
    targets = sorted(local | remote)
    for page_id in targets:
        enqueue_page(page_id)
    return {"action": "requeued", "pages": len(targets), "chapter_missing": info is None}


def apply_shelf_refresh(
    store: StateStore,
    shelf_id: int,
    shelf_name: str,
    fetch_shelf: Callable[[int], Optional[dict]],
    enqueue_book: Callable[[int], object],
) -> dict:
    """Refresh books now on the shelf and books indexed under its name. `fetch_shelf` returns {"name", "book_ids"} or None."""
    info = fetch_shelf(shelf_id)
    books = set(int(book_id) for book_id in (info or {}).get("book_ids") or [])
    for name in {shelf_name, (info or {}).get("name") or ""}:
        if name:
            books.update(store.catalog_book_ids_for_shelf(name))
    for book_id in sorted(books):
        enqueue_book(book_id)
    return {"action": "books_queued", "books": len(books)}
