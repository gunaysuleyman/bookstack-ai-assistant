import hashlib
import hmac
import logging
import os
import time
from typing import Any, Dict, List, Optional

import httpx
from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from adaptive.auth import AuthFailure, scope_from_payload, service_token_ok, verify_signed_payload
from adaptive.config import load_settings, manifest
from adaptive.contracts import AuthorizationScope
from adaptive.embeddings import build_embedding
from adaptive.engine import AdaptiveEngine
from adaptive.indexer import Indexer
from adaptive.jobs import apply_page_job, apply_reconcile
from adaptive.store import StateStore
from adaptive.vector_index import VectorIndex
from adaptive.worker import IndexWorker
from adaptive.provider import collect_turn_usages, complete_llm, summarize_turn
from rag_engine import RAGEngine
from sync import BookStackSync

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("RAGService")

settings = load_settings()
app = FastAPI(
    title="BookStack AI / RAG Service",
    version="1.1.0",
    description="Permission-bounded RAG for BookStack",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

rag_engine = RAGEngine()
sync_engine = BookStackSync(rag_engine)
state_store = StateStore(os.path.join(settings.state_dir, "rag_state.sqlite"))
adaptive_engine: Optional[AdaptiveEngine] = None
adaptive_indexer: Optional[Indexer] = None
index_worker: Optional[IndexWorker] = None


def _build_adaptive() -> None:
    global adaptive_engine, adaptive_indexer
    if settings.mode == "off" and not settings.adaptive_indexing:
        return
    embedding = build_embedding(settings.embedding_model_id)
    vectors = VectorIndex(settings.chroma_dir, settings.adaptive_collection, embedding)
    adaptive_indexer = Indexer(state_store, vectors, settings)
    adaptive_engine = AdaptiveEngine(state_store, vectors, settings, llm=_adaptive_llm)


def _adaptive_llm(system_instruction: str, user_prompt: str, purpose: str, **kwargs):
    if settings.answer_mode == "extractive" and purpose in {"answer", "tool_select", "tool_answer"}:
        raise RuntimeError("extractive mode")
    if rag_engine.provider not in {"gemini", "openai"}:
        raise ValueError(f"Unsupported AI_PROVIDER: {rag_engine.provider}")
    result = complete_llm(
        provider=rag_engine.provider,
        model=rag_engine.gemini_model,
        fallbacks=rag_engine.gemini_fallbacks,
        api_key=rag_engine.gemini_key,
        openai_model=rag_engine.openai_model,
        openai_key=rag_engine.openai_key,
        reasoning_effort=rag_engine.openai_reasoning,
        system_instruction=system_instruction,
        user_prompt=user_prompt,
        purpose=purpose,
        timeout_s=float(os.getenv("LLM_TIMEOUT_SECONDS", "90" if rag_engine.openai_reasoning else "30")),
        max_retries=0 if purpose in {"tool_select", "tool_answer"} else 2,
        contents=kwargs.get("contents"),
        tools=kwargs.get("tools"),
        tool_config=kwargs.get("tool_config"),
        max_output_tokens=settings.output_reserve_tokens,
    )
    rag_engine.last_usage = result.usage
    return result


class ChatMessage(BaseModel):
    role: str = Field(max_length=20)
    content: str = Field(max_length=4000)


class SearchQuery(BaseModel):
    query: str = Field(min_length=1, max_length=2000)
    top_k: Optional[int] = Field(default=None, ge=1, le=50)
    current_page: Optional[Dict[str, Any]] = None
    user_token: Optional[Dict[str, Any]] = None
    history: Optional[List[ChatMessage]] = Field(default=None, max_length=12)
    total_session_turns: Optional[int] = Field(default=None, ge=0, le=100)


class ScopeRegistration(BaseModel):
    user_id: int | str
    is_admin: bool
    can_use_ai: bool
    roles: List[str] = Field(default_factory=list)
    allowed_page_ids: Optional[List[int]] = None
    ts: int


def _require_service(token: Optional[str]) -> None:
    if not service_token_ok(token, settings.service_secret):
        raise HTTPException(status_code=401, detail="Invalid or missing service token")


def resolve_scope(user_token: Optional[Dict[str, Any]]) -> AuthorizationScope:
    if not user_token:
        raise AuthFailure(401, "Missing user security token")
    scope_ref = user_token.get("scope_ref")
    if isinstance(scope_ref, str) and scope_ref:
        scope = state_store.get_scope(scope_ref)
        if scope is None:
            raise AuthFailure(403, "Security token expired")
        if not scope.can_use_ai:
            raise AuthFailure(403, "AI Assistant is not enabled for your user role.")
        return scope
    return verify_signed_payload(
        user_token,
        settings.service_secret,
        settings.token_ttl_seconds,
        skew_seconds=settings.clock_skew_seconds,
    )


def _context_usage(payload: SearchQuery, history_dicts: List[dict], answer: str, extra: Optional[dict] = None) -> dict:
    client_turns = payload.total_session_turns or 0
    history_turns = (len(history_dicts) // 2) + 1
    turn_count = max(client_turns, history_turns)
    history_chars = sum(len(item["content"]) for item in history_dicts)
    est_tokens = (len(payload.query) + history_chars + len(answer)) // 4
    max_turns = int(os.getenv("MAX_RECOMMENDED_TURNS", "5"))
    usage = {
        "turn_count": turn_count,
        "estimated_tokens": est_tokens,
        "actual_tokens": None,
        "max_recommended_turns": max_turns,
        "suggest_new_chat": bool(turn_count >= max_turns or est_tokens >= 3500),
    }
    if extra:
        if extra.get("estimated_tokens") is not None:
            usage["estimated_tokens"] = extra["estimated_tokens"]
        usage["actual_tokens"] = extra.get("actual_tokens")
        usage["extra_rounds"] = extra.get("extra_rounds", 0)
        usage["stop_reason"] = extra.get("stop_reason")
    return usage


def _public_result(result: dict, payload: SearchQuery, history_dicts: List[dict]) -> dict:
    extra = result.get("context_meta") or {
        "estimated_tokens": result.get("estimated_tokens"),
        "actual_tokens": result.get("actual_tokens"),
        "extra_rounds": result.get("extra_rounds", 0),
        "stop_reason": result.get("stop_reason"),
    }
    body = {
        "answer": result.get("answer", ""),
        "sources": result.get("sources", []),
        "context_usage": _context_usage(payload, history_dicts, result.get("answer", ""), extra),
    }
    return body


def _handle_job(job) -> None:
    if settings.adaptive_indexing and adaptive_indexer is not None and job.event in {"page_upsert", "page_delete", "full_reconcile", "book_refresh"}:
        if job.event in {"page_upsert", "page_delete"}:
            apply_page_job(job, adaptive_indexer, sync_engine.load_page, legacy=sync_engine)
            return
        if job.event == "full_reconcile":
            apply_reconcile(state_store, adaptive_indexer, sync_engine.load_page, sync_engine.list_page_stubs, legacy=sync_engine)
            return
        book_id = job.payload.get("book_id")
        if book_id:
            _refresh_book(int(book_id))
        return
    if job.event == "page_delete":
        sync_engine.delete_page(job.page_id)
    elif job.event == "page_upsert":
        sync_engine.sync_single_page(job.page_id)
    elif job.event == "full_reconcile":
        sync_engine.sync_all_pages()
    elif job.event == "book_refresh":
        book_id = job.payload.get("book_id")
        if book_id:
            _refresh_book(int(book_id))


def _refresh_book(book_id: int) -> None:
    sync_engine._require_credentials()
    url = f"{sync_engine.bookstack_url}/api/books/{book_id}"
    with httpx.Client(timeout=30.0) as client:
        response = client.get(url, headers=sync_engine._get_headers())
        response.raise_for_status()
        data = response.json()
    info = {
        "book_name": data.get("name") or f"Book #{book_id}",
        "shelf_names": [shelf.get("name") for shelf in (data.get("shelves") or []) if isinstance(shelf, dict) and shelf.get("name")],
    }
    shelf_name = " | ".join(info["shelf_names"]) if info["shelf_names"] else "General Shelf"
    rag_engine.update_book_metadata(book_id, info["book_name"], shelf_name)
    if adaptive_indexer is not None:
        state_store.relabel_book(book_id, info["book_name"], info["shelf_names"])


@app.get("/health")
def health_check():
    return {"status": "healthy", "provider": rag_engine.provider, "chroma_dir": rag_engine.chroma_dir}


@app.get("/ready")
def ready():
    info = manifest(settings)
    worker_required = settings.adaptive_indexing and os.getenv("ENABLE_INDEX_WORKER", "1") == "1"
    worker_alive = bool(index_worker and index_worker._thread and index_worker._thread.is_alive())
    webhook_ready = not settings.adaptive_indexing or bool(settings.webhook_secret)
    service_secret_ready = bool(settings.service_secret) and settings.service_secret != "my_super_secret_local_token_123"
    index_ready = settings.mode != "on" or adaptive_engine is not None
    if index_ready and settings.mode == "on":
        try:
            published = bool(state_store.active_revision_map())
            index_ready = not published or adaptive_engine.vectors.collection.count() > 0
        except Exception:
            index_ready = False
    info["worker"] = worker_alive
    info["checks"] = {
        "worker": not worker_required or worker_alive,
        "webhook_configured": webhook_ready,
        "service_secret_configured": service_secret_ready,
        "index_available": index_ready,
    }
    info["ready"] = all(info["checks"].values())
    return JSONResponse(status_code=200 if info["ready"] else 503, content=info)


@app.get("/api/jobs/status")
def job_status(x_rag_token: Optional[str] = Header(None)):
    _require_service(x_rag_token)
    return state_store.job_counts()


@app.post("/api/scope")
def register_scope(body: ScopeRegistration, x_rag_token: Optional[str] = Header(None)):
    _require_service(x_rag_token)
    payload = body.model_dump()
    scope = scope_from_payload(payload, settings.token_ttl_seconds)
    scope_ref = hashlib.sha256(os.urandom(32)).hexdigest()
    state_store.save_scope(scope_ref, scope)
    return {"scope_ref": scope_ref, "expires_at": scope.expires_at}


@app.post("/api/ai-search")
def ai_search(payload: SearchQuery, x_rag_token: Optional[str] = Header(None)):
    try:
        scope = resolve_scope(payload.user_token)
    except AuthFailure as exc:
        raise HTTPException(status_code=exc.status_code, detail=exc.detail) from exc
    allowed = None if scope.allows_all() else list(scope.allowed_page_ids or [])
    history_dicts = [{"role": item.role, "content": item.content} for item in payload.history] if payload.history else []
    diagnostics = service_token_ok(x_rag_token, settings.service_secret)
    with collect_turn_usages() as usages:
        if settings.mode == "on" and adaptive_engine is not None:
            result = adaptive_engine.answer(
                payload.query,
                scope,
                current_page=payload.current_page,
                history=history_dicts,
                diagnostics=diagnostics,
            )
        else:
            result = rag_engine.search_and_answer(
                query=payload.query,
                top_k=settings.legacy_result_limit,
                current_page=payload.current_page,
                allowed_page_ids=allowed,
                history=history_dicts,
            )
            if rag_engine.last_usage is not None:
                state_store.log_usage(rag_engine.last_usage)
            if settings.mode == "shadow" and adaptive_engine is not None:
                _record_shadow(payload.query, scope, result)
        _log_assistant_turn(scope, payload.query, result, usages)
    body = _public_result(result, payload, history_dicts)
    if diagnostics and "assessments" in result:
        body["diagnostics"] = {
            "intent": result.get("intent"),
            "stop_reason": result.get("stop_reason"),
            "extra_rounds": result.get("extra_rounds"),
            "reranker": result.get("reranker"),
            "evidence_page_ids": result.get("evidence_page_ids", []),
            "assessments": [
                {"sub_question": item.get("sub_question"), "status": item.get("status"), "page_ids": item.get("page_ids")}
                for item in result.get("assessments", [])
            ],
        }
    return body


def _log_assistant_turn(scope: AuthorizationScope, query: str, result: dict, usages: list) -> None:
    summary = summarize_turn(usages)
    effort = ""
    if not summary["provider"]:
        summary["provider"] = rag_engine.provider
        summary["model"] = rag_engine.openai_model if rag_engine.provider == "openai" else rag_engine.gemini_model
    if summary["provider"] == "openai":
        effort = rag_engine.openai_reasoning
    try:
        state_store.log_assistant_turn(
            user_id=str(scope.principal),
            query=query,
            answer=str((result or {}).get("answer") or ""),
            reasoning_effort=effort,
            provider=summary["provider"],
            model=summary["model"],
            input_tokens=summary["input_tokens"],
            output_tokens=summary["output_tokens"],
            input_usd_per_mtok=summary["input_usd_per_mtok"],
            output_usd_per_mtok=summary["output_usd_per_mtok"],
            cost_usd=summary["cost_usd"],
        )
    except Exception:
        logger.warning("Assistant turn log failed")


def _record_shadow(query: str, scope: AuthorizationScope, legacy_result: dict) -> None:
    if adaptive_engine is None:
        return
    started = time.perf_counter()
    try:
        pages = adaptive_engine.retrieve_pages(query, scope)
    except Exception as exc:
        logger.info("Shadow retrieval failed: %s", exc)
        return
    legacy_pages = [item.get("page_id") for item in legacy_result.get("sources", []) if item.get("page_id") is not None]
    elapsed = (time.perf_counter() - started) * 1000
    state_store.log_shadow(hashlib.sha256(query.encode("utf-8")).hexdigest(), legacy_pages, pages, elapsed)


@app.post("/api/sync")
def trigger_sync(x_rag_token: Optional[str] = Header(None)):
    _require_service(x_rag_token)
    job_id = state_store.enqueue(0, "full_reconcile", {})
    return {"message": "Reconciliation queued.", "job_id": job_id}


def _webhook_authorized(presented: Optional[str]) -> bool:
    secret = settings.webhook_secret
    if not secret or not isinstance(presented, str) or len(presented) != len(secret):
        return False
    return hmac.compare_digest(presented, secret)


@app.post("/api/webhook")
async def handle_webhook(request: Request, x_webhook_token: Optional[str] = Header(None), token: Optional[str] = None):
    if not _webhook_authorized(x_webhook_token or token):
        raise HTTPException(status_code=401, detail="Invalid webhook token")
    try:
        data = await request.json()
    except Exception as exc:
        raise HTTPException(status_code=400, detail="Invalid webhook payload") from exc
    event = data.get("event")
    related = data.get("related_item") or {}
    page_id = related.get("id")
    if event in {"page_create", "page_update"} and page_id:
        job_id = state_store.enqueue(int(page_id), "page_upsert", {"page_id": int(page_id)})
        return {"status": "accepted", "job_id": job_id}
    if event == "page_delete" and page_id:
        job_id = state_store.enqueue(int(page_id), "page_delete", {"page_id": int(page_id)})
        return {"status": "accepted", "job_id": job_id}
    if event == "book_update" and page_id:
        job_id = state_store.enqueue(int(page_id), "book_refresh", {"book_id": int(page_id)})
        return {"status": "accepted", "job_id": job_id}
    if event == "bookshelf_update":
        return {"status": "ignored", "reason": "Shelf membership is applied on book_update or full reconciliation"}
    return {"status": "ignored", "reason": "Unhandled event"}


@app.on_event("startup")
def startup_event():
    global index_worker
    if not settings.service_secret or settings.service_secret == "my_super_secret_local_token_123":
        raise RuntimeError("RAG_SECRET_TOKEN must be a unique non-default secret")
    if settings.adaptive_indexing and not settings.webhook_secret:
        raise RuntimeError("WEBHOOK_SECRET is required when ADAPTIVE_INDEXING is enabled")
    logger.info("RAG service starting. Startup full sync is disabled.")
    _build_adaptive()
    if settings.adaptive_indexing and adaptive_indexer is not None:
        adaptive_indexer.recover()
    if os.getenv("ENABLE_INDEX_WORKER", "1") == "1":
        index_worker = IndexWorker(state_store, settings, _handle_job)
        index_worker.start()
