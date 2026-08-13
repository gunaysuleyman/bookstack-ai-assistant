import os
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
    description="RAG and AI Search Layer for BookStack with Page Awareness"
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

class SearchQuery(BaseModel):
    query: str
    top_k: Optional[int] = 6
    current_page: Optional[Dict[str, Any]] = None

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

    logger.info(f"AI Search Query received: '{payload.query}' (Current Page: {payload.current_page.get('title') if payload.current_page else 'None'})")
    result = rag_engine.search_and_answer(
        query=payload.query,
        top_k=payload.top_k,
        current_page=payload.current_page
    )
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
