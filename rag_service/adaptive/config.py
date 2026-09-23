import os
from dataclasses import dataclass

from adaptive import INDEX_VERSION, SCHEMA_VERSION


def _flag(name: str, default: str = "0") -> bool:
    return os.getenv(name, default).strip().lower() in {"1", "true", "yes", "on"}


def adaptive_mode() -> str:
    raw = os.getenv("ADAPTIVE_RAG", "0").strip().lower()
    if raw in {"1", "true", "yes", "on"}:
        return "on"
    if raw in {"shadow"}:
        return "shadow"
    return "off"


@dataclass(frozen=True)
class Settings:
    mode: str
    adaptive_indexing: bool
    token_ttl_seconds: int
    clock_skew_seconds: int
    service_secret: str
    webhook_secret: str
    state_dir: str
    chroma_dir: str
    legacy_collection: str
    adaptive_collection: str
    chunk_schema_version: str
    index_version: str
    embedding_model_id: str
    embed_max_tokens: int
    child_tokens: int
    parent_expand_tokens: int
    context_token_budget: int
    history_token_budget: int
    output_reserve_tokens: int
    max_extra_rounds: int
    candidate_batch: int
    candidate_step: int
    max_candidates: int
    acl_batch: int
    max_model_calls: int
    request_deadline_s: float
    legacy_result_limit: int
    planner_enabled: bool
    reranker_enabled: bool
    answer_mode: str
    overview_page_size: int
    job_lease_seconds: int
    job_max_attempts: int
    max_query_chars: int
    max_history_messages: int
    max_history_chars: int
    image_prompt_version: str
    tools_enabled: bool
    max_tool_result_passages: int
    catalog_book_page_size: int


def load_settings() -> Settings:
    chroma_dir = os.getenv("CHROMA_PERSIST_DIR", "/app/chroma_db")
    state_dir = os.getenv("RAG_STATE_DIR", chroma_dir)
    mode = adaptive_mode()
    return Settings(
        mode=mode,
        adaptive_indexing=_flag("ADAPTIVE_INDEXING", "1" if mode == "on" else "0"),
        token_ttl_seconds=int(os.getenv("TOKEN_TTL_SECONDS", "900")),
        clock_skew_seconds=int(os.getenv("TOKEN_CLOCK_SKEW_SECONDS", "60")),
        service_secret=os.getenv("RAG_SECRET_TOKEN", ""),
        webhook_secret=os.getenv("WEBHOOK_SECRET", ""),
        state_dir=state_dir,
        chroma_dir=chroma_dir,
        legacy_collection=os.getenv("LEGACY_COLLECTION", "bookstack_articles"),
        adaptive_collection=os.getenv("ADAPTIVE_COLLECTION", "bookstack_articles_pc_v1"),
        chunk_schema_version=SCHEMA_VERSION,
        index_version=INDEX_VERSION,
        embedding_model_id=os.getenv("EMBEDDING_MODEL_ID", "gemini-embedding-001"),
        embed_max_tokens=int(os.getenv("EMBED_MAX_TOKENS", "180")),
        child_tokens=int(os.getenv("CHILD_CHUNK_TOKENS", "160")),
        parent_expand_tokens=int(os.getenv("PARENT_EXPAND_TOKENS", "220")),
        context_token_budget=int(os.getenv("CONTEXT_TOKEN_BUDGET", "3000")),
        history_token_budget=int(os.getenv("HISTORY_TOKEN_BUDGET", "800")),
        output_reserve_tokens=int(os.getenv("OUTPUT_RESERVE_TOKENS", "800")),
        max_extra_rounds=int(os.getenv("MAX_EXTRA_ROUNDS", "2")),
        candidate_batch=int(os.getenv("CANDIDATE_BATCH", "12")),
        candidate_step=int(os.getenv("CANDIDATE_STEP", "12")),
        max_candidates=int(os.getenv("MAX_CANDIDATES", "40")),
        acl_batch=int(os.getenv("ACL_FILTER_BATCH", "200")),
        max_model_calls=int(os.getenv("MAX_MODEL_CALLS", "3")),
        request_deadline_s=float(os.getenv("REQUEST_DEADLINE_SECONDS", "25")),
        legacy_result_limit=int(os.getenv("LEGACY_RESULT_LIMIT", "8")),
        planner_enabled=_flag("PLANNER_ENABLED", "0"),
        reranker_enabled=_flag("RERANKER_ENABLED", "0"),
        answer_mode=os.getenv("ADAPTIVE_ANSWER_MODE", "llm").strip().lower(),
        overview_page_size=int(os.getenv("OVERVIEW_PAGE_SIZE", "20")),
        job_lease_seconds=int(os.getenv("JOB_LEASE_SECONDS", "120")),
        job_max_attempts=int(os.getenv("JOB_MAX_ATTEMPTS", "5")),
        max_query_chars=int(os.getenv("MAX_QUERY_CHARS", "2000")),
        max_history_messages=int(os.getenv("MAX_HISTORY_MESSAGES", "12")),
        max_history_chars=int(os.getenv("MAX_HISTORY_CHARS", "4000")),
        image_prompt_version=os.getenv("IMAGE_PROMPT_VERSION", "v1"),
        tools_enabled=_flag("ADAPTIVE_TOOLS", "1"),
        max_tool_result_passages=int(os.getenv("MAX_TOOL_RESULT_PASSAGES", "8")),
        catalog_book_page_size=int(os.getenv("CATALOG_BOOK_PAGE_SIZE", "20")),
    )


def manifest(settings: Settings) -> dict:
    active = settings.adaptive_collection if settings.mode == "on" else settings.legacy_collection
    return {
        "adaptive_rag": settings.mode,
        "adaptive_indexing": settings.adaptive_indexing,
        "active_collection": active,
        "legacy_collection": settings.legacy_collection,
        "adaptive_collection": settings.adaptive_collection,
        "chunk_schema_version": settings.chunk_schema_version,
        "index_version": settings.index_version,
        "embedding_model_id": settings.embedding_model_id,
        "embed_max_tokens": settings.embed_max_tokens,
        "embedding_truncation_tokens": 256,
        "chroma_owner": "single_process_persistent_client",
        "answer_cache": "disabled",
        "token_ttl_seconds": settings.token_ttl_seconds,
        "tools_enabled": settings.tools_enabled,
        "max_model_calls": settings.max_model_calls,
    }
