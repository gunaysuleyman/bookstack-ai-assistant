import os
import hmac
import hashlib
import json
import logging
from fastapi import FastAPI, Header, HTTPException, BackgroundTasks, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List, Dict, Any

from rag_engine import RAGEngine
from sync import BookStackSync

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("RAGService")

app = FastAPI(
    title="BookStack AI / RAG Service",
    version="1.0.0",
    description="RAG and AI Search Layer for BookStack with Page Awareness and Permission Trimming"
)

origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

rag_engine = RAGEngine()
sync_engine = BookStackSync(rag_engine)

RAG_SECRET_TOKEN = os.getenv("RAG_SECRET_TOKEN", "my_super_secret_local_token_123")

def verify_token(x_rag_token: Optional[str] = Header(None)):
    if RAG_SECRET_TOKEN and x_rag_token != RAG_SECRET_TOKEN:
        raise HTTPException(status_code=401, detail="Invalid or missing X-RAG-Token header")

class ChatMessage(BaseModel):
    role: str
    content: str

class SearchQuery(BaseModel):
    query: str
    top_k: Optional[int] = 6
    current_page: Optional[Dict[str, Any]] = None
    user_token: Optional[Dict[str, Any]] = None
    history: Optional[List[ChatMessage]] = None
    total_session_turns: Optional[int] = None

def extract_allowed_page_ids(user_token: Optional[Dict[str, Any]]) -> Optional[List[int]]:
    """
    Validates HMAC signature on user_token and extracts allowed_page_ids.
    Returns:
      None -> Admin user (access to all pages)
      List[int] -> Explicit list of permitted page IDs
      [] -> No pages permitted
    """
    if not user_token:
        # Fallback if user_token is not provided (e.g. direct API or curl mode)
        logger.info("No user_token in request. Defaulting to full access (admin/direct API mode).")
        return None

    raw_payload = user_token.get("payload")
    signature = user_token.get("sig")

    if not raw_payload or not signature:
        logger.warning("user_token provided without payload or signature. Rejecting access.")
        return []

    # Verify HMAC
    expected_sig = hmac.new(
        RAG_SECRET_TOKEN.encode("utf-8"),
        raw_payload.encode("utf-8"),
        hashlib.sha256
    ).hexdigest()

    if not hmac.compare_digest(signature, expected_sig):
        logger.error("Tampered user_token signature detected!")
        raise HTTPException(status_code=403, detail="Invalid user security token signature")

    try:
        data = json.loads(raw_payload) if isinstance(raw_payload, str) else raw_payload
    except Exception as e:
        logger.error(f"Failed to parse user_token payload: {e}")
        return []

    if data.get("can_use_ai") is False:
        logger.warning(f"AI Assistant access denied for user {data.get('user_id')} with roles {data.get('roles')}")
        raise HTTPException(status_code=403, detail="AI Assistant is not enabled for your user role.")

    if data.get("is_admin") is True:
        return None

    allowed_ids = data.get("allowed_page_ids", [])
    if isinstance(allowed_ids, list):
        parsed = []
        for x in allowed_ids:
            try:
                parsed.append(int(x))
            except (ValueError, TypeError):
                pass
        return parsed
    return []

@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "provider": rag_engine.provider,
        "chroma_dir": rag_engine.chroma_dir
    }

@app.post("/api/ai-search")
def ai_search(payload: SearchQuery, x_rag_token: Optional[str] = Header(None)):
    verify_token(x_rag_token)
    if not payload.query or not payload.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty")

    allowed_page_ids = extract_allowed_page_ids(payload.user_token)

    logger.info(
        f"AI Search Query received: '{payload.query}' "
        f"(Current Page: {payload.current_page.get('title') if payload.current_page else 'None'}, "
        f"Permitted Pages: {len(allowed_page_ids) if allowed_page_ids is not None else 'All'})"
    )

    history_dicts = [{"role": m.role, "content": m.content} for m in payload.history] if payload.history else []

    result = rag_engine.search_and_answer(
        query=payload.query,
        top_k=payload.top_k,
        current_page=payload.current_page,
        allowed_page_ids=allowed_page_ids,
        history=history_dicts
    )

    # Context Usage & Threshold Calculation
    client_turns = payload.total_session_turns or 0
    history_turns = (len(history_dicts) // 2) + 1
    turn_count = max(client_turns, history_turns)

    history_chars = sum(len(m["content"]) for m in history_dicts)
    answer_chars = len(result.get("answer", ""))
    est_tokens = (len(payload.query) + history_chars + answer_chars) // 4
    
    max_turns = int(os.getenv("MAX_RECOMMENDED_TURNS", "5"))
    # Proactively suggest new chat if turn_count reaches threshold or token estimate is high
    suggest_new_chat = bool(turn_count >= max_turns or est_tokens >= 3500)

    result["context_usage"] = {
        "turn_count": turn_count,
        "estimated_tokens": est_tokens,
        "max_recommended_turns": max_turns,
        "suggest_new_chat": suggest_new_chat
    }

    return result

@app.post("/api/sync")
def trigger_sync(background_tasks: BackgroundTasks, x_rag_token: Optional[str] = Header(None)):
    verify_token(x_rag_token)
    background_tasks.add_task(sync_engine.sync_all_pages)
    return {"message": "Background full synchronization started."}

@app.post("/api/webhook")
async def handle_webhook(request: Request):
    try:
        data = await request.json()
        event = data.get("event")
        related_item = data.get("related_item", {})
        page_id = related_item.get("id")

        logger.info(f"Webhook received! Event: '{event}', Page ID: {page_id}")

        if not page_id:
            return {"status": "ignored", "reason": "No page ID in event"}

        if event in ["page_create", "page_update"]:
            sync_engine.sync_single_page(page_id)
            return {"status": "success", "action": f"Synced page {page_id}"}

        elif event == "page_delete":
            sync_engine.delete_page(page_id)
            return {"status": "success", "action": f"Deleted page {page_id} from vector index"}

        return {"status": "ignored", "reason": f"Unhandled event '{event}'"}

    except Exception as e:
        logger.error(f"Error processing webhook: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.on_event("startup")
def startup_event():
    logger.info("RAG Service starting up. Checking BookStack connectivity...")
    try:
        sync_engine.sync_all_pages()
    except Exception as e:
        logger.warning(f"Initial sync at startup deferred: {e}")
