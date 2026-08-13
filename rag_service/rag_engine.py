import os
import logging
import chromadb
from chromadb.utils import embedding_functions
from typing import List, Dict, Any, Optional
import httpx
import json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("RAGEngine")

class RAGEngine:
    def __init__(self):
        self.provider = os.getenv("AI_PROVIDER", "gemini").lower()
        self.gemini_key = os.getenv("GEMINI_API_KEY", "")
        self.openai_key = os.getenv("OPENAI_API_KEY", "")
        self.chroma_dir = os.getenv("CHROMA_PERSIST_DIR", "/app/chroma_db")

        # Default local embedding function (runs 100% locally, fast & reliable)
        self.embedding_fn = embedding_functions.DefaultEmbeddingFunction()

        self.chroma_client = chromadb.PersistentClient(path=self.chroma_dir)
        self.collection_name = "bookstack_articles"
        self.collection = self._get_or_create_collection()

    def _get_or_create_collection(self):
        try:
            return self.chroma_client.get_or_create_collection(
                name=self.collection_name,
                embedding_function=self.embedding_fn,
                metadata={"hnsw:space": "cosine"}
            )
        except Exception as e:
            logger.error(f"Error initializing Chroma collection: {e}")
            return self.chroma_client.create_collection(
                name=self.collection_name,
                embedding_function=self.embedding_fn
            )

    def _get_indexed_catalog(self) -> Dict[str, Any]:
        """Retrieves a full 4-tier BookStack hierarchy tree (Shelves -> Books -> Chapters -> Pages) of all indexed items in ChromaDB."""
        try:
            all_meta = self.collection.get(include=["metadatas"])
            tree_map = {}
            all_pages_map = {}

            if all_meta and all_meta.get("metadatas"):
                for meta in all_meta["metadatas"]:
                    pid = meta.get("page_id")
                    sname = meta.get("shelf_name", "General Shelf")
                    bname = meta.get("book_name", "General Library")
                    cname = meta.get("chapter_name", "General Chapter")
                    title = meta.get("name", f"Article #{pid}")
                    url = meta.get("url", "")
                    tags_str = meta.get("tags_str", "")

                    if pid and pid not in all_pages_map:
                        all_pages_map[pid] = {
                            "page_id": pid,
                            "title": title,
                            "url": url,
                            "book_name": bname,
                            "shelf_name": sname,
                            "chapter_name": cname,
                            "tags_str": tags_str
                        }

                    if sname not in tree_map:
                        tree_map[sname] = {}
                    if bname not in tree_map[sname]:
                        tree_map[sname][bname] = {}
                    if cname not in tree_map[sname][bname]:
                        tree_map[sname][bname][cname] = []

                    page_desc = f"{title}" + (f" [Tags: {tags_str}]" if tags_str else "")
                    if page_desc not in tree_map[sname][bname][cname]:
                        tree_map[sname][bname][cname].append(page_desc)

            if not tree_map:
                return {"summary": "No documents or books indexed in the system yet.", "pages": {}}

            tree_lines = [f"=== FULL BOOKSTACK LIBRARY & HIERARCHY CATALOG ===",
                          f"All Shelves, Books, Chapters, and Pages currently available in the system:\n"]

            for sname, books in tree_map.items():
                tree_lines.append(f"📂 Shelf: '{sname}'")
                for bname, chapters in books.items():
                    tree_lines.append(f"  └─ 📚 Book: '{bname}'")
                    for cname, pages in chapters.items():
                        c_prefix = f"        └─ 📑 Chapter: '{cname}'" if cname != "General Chapter" else "        └─ 📄 Pages:"
                        tree_lines.append(c_prefix)
                        for pdesc in pages:
                            tree_lines.append(f"              └─ 📄 Page: {pdesc}")

            return {"summary": "\n".join(tree_lines), "pages": all_pages_map}
        except Exception as e:
            logger.warning(f"Failed to fetch document catalog: {e}")
            return {"summary": "", "pages": {}}

    def _call_llm_api(self, system_instruction: str, user_prompt: str) -> str:
        """Internal helper to call Gemini or OpenAI API."""
        full_prompt = f"{system_instruction}\n\n{user_prompt}"

        if self.provider == "gemini":
            models_to_try = ["gemini-3.6-flash", "gemini-2.5-flash", "gemini-flash-latest"]
            last_err = None
            for model_name in models_to_try:
                url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:generateContent?key={self.gemini_key}"
                payload = {
                    "contents": [{"parts": [{"text": full_prompt}]}]
                }
                try:
                    with httpx.Client(timeout=60.0) as client:
                        res = client.post(url, json=payload)
                        res.raise_for_status()
                        data = res.json()
                        return data["candidates"][0]["content"]["parts"][0]["text"]
                except Exception as e:
                    logger.warning(f"Failed with model {model_name}: {e}")
                    last_err = e
            
            raise RuntimeError(f"All Gemini models failed. Last error: {last_err}")

        elif self.provider == "openai":
            headers = {"Authorization": f"Bearer {self.openai_key}"}
            payload = {
                "model": "gpt-4o-mini",
                "messages": [
                    {"role": "system", "content": system_instruction},
                    {"role": "user", "content": user_prompt}
                ]
            }
            with httpx.Client(timeout=60.0) as client:
                res = client.post("https://api.openai.com/v1/chat/completions", json=payload, headers=headers)
                res.raise_for_status()
                return res.json()["choices"][0]["message"]["content"]
        else:
            raise ValueError(f"Unsupported AI_PROVIDER: {self.provider}")

    def classify_and_route_intent(self, query: str) -> Dict[str, Any]:
        """
        LAYER 1: Intent Router AI
        Classifies user query intent into:
        - 'GREETING': Smalltalk, hello, hi, merhaba, thanks
        - 'OVERVIEW': Asking about catalog, books count, shelves structure
        - 'SEARCH': Specific topic or document search
        And optimizes search query for vector retrieval.
        """
        router_prompt = (
            "You are an AI Intent Router for a BookStack Documentation Assistant.\n"
            "Classify the user query into JSON format with keys:\n"
            "- 'intent': string ('GREETING' | 'OVERVIEW' | 'SEARCH')\n"
            "- 'optimized_query': string (corrected and expanded search query for vector search if SEARCH, else empty)\n\n"
            f"User Query: '{query}'\n\n"
            "Respond ONLY with valid JSON, e.g. {\"intent\": \"SEARCH\", \"optimized_query\": \"PISA 2025 questionnaire verification steps\"}"
        )

        try:
            res_text = self._call_llm_api("Return valid JSON only.", router_prompt)
            clean_json = res_text.strip().lstrip("```json").rstrip("```").strip()
            data = json.loads(clean_json)
            return data
        except Exception as e:
            logger.warning(f"Intent Router fallback due to parsing error: {e}")
            q_lower = query.strip().lower()
            if q_lower in ["hello", "hi", "hey", "merhaba", "selam", "günaydın", "iyi günler", "thanks", "teşekkürler"]:
                return {"intent": "GREETING", "optimized_query": query}
            elif any(w in q_lower for w in ["kaç", "hangi", "makale", "doküman", "sayfa", "kitap", "raf", "bölüm", "etiket", "liste", "list", "how many", "which", "books", "pages", "shelves"]):
                return {"intent": "OVERVIEW", "optimized_query": query}
            else:
                return {"intent": "SEARCH", "optimized_query": query}

    def generate_llm_response(self, prompt: str, context: str) -> str:
        """LAYER 2: Generates final response based on retrieved context and system instructions."""
        system_instruction = (
            "You are an expert AI Assistant integrated into BookStack Documentation System.\n"
            "You have complete mastery over BookStack's 4-Tier Hierarchy: Shelves -> Books -> Chapters -> Pages and Tags.\n"
            "PAGE AWARENESS RULE: Pay special attention to the active page the user is currently reading ('ACTIVE PAGE CONTEXT'). If the user asks to summarize the page, asks 'what is this', or asks about steps on the active page, prioritize the active page context.\n"
            "Answer the user's question accurately, concisely, and based strictly on the provided Context documents and Hierarchy Catalog.\n"
            "LANGUAGE DYNAMICS RULE: Match the language of the user's question. If the user asks in Turkish, reply in Turkish. If the user asks in English, reply in English.\n"
            "Use clean markdown formatting, bullet points, and bold terms for key concepts."
        )

        full_prompt = f"--- CONTEXT & FULL HIERARCHY CATALOG ---\n{context}\n\n--- USER QUESTION ---\n{prompt}"
        return self._call_llm_api(system_instruction, full_prompt)

    def add_page_chunks(self, page_id: int, chunks: List[Dict[str, Any]]):
        """Removes existing chunks for the page and adds new chunks with full 4-tier hierarchy metadata."""
        self.delete_page(page_id)
        if not chunks:
            return

        ids = []
        documents = []
        metadatas = []

        for chunk in chunks:
            ids.append(chunk["id"])
            documents.append(chunk["text"])

            b_id = chunk["metadata"].get("book_id")
            c_id = chunk["metadata"].get("chapter_id")

            meta = {
                "page_id": int(chunk["metadata"]["id"]),
                "name": str(chunk["metadata"].get("name", "")),
                "book_id": int(b_id) if b_id is not None else 0,
                "book_name": str(chunk["metadata"].get("book_name", "General Library")),
                "shelf_name": str(chunk["metadata"].get("shelf_name", "General Shelf")),
                "chapter_id": int(c_id) if c_id is not None else 0,
                "chapter_name": str(chunk["metadata"].get("chapter_name", "General Chapter")),
                "tags_str": str(chunk["metadata"].get("tags_str", "")),
                "slug": str(chunk["metadata"].get("slug", "")),
                "url": str(chunk["metadata"].get("url", "")),
                "updated_at": str(chunk["metadata"].get("updated_at", "")),
                "chunk_index": int(chunk["metadata"].get("chunk_index", 0))
            }
            metadatas.append(meta)

        if ids:
            self.collection.add(
                ids=ids,
                documents=documents,
                metadatas=metadatas
            )
            logger.info(f"Successfully indexed {len(ids)} chunks for page ID {page_id}")

    def delete_page(self, page_id: int):
        """Deletes all chunks associated with a specific page ID."""
        try:
            results = self.collection.get(where={"page_id": page_id})
            if results and results.get("ids"):
                self.collection.delete(ids=results["ids"])
                logger.info(f"Deleted {len(results['ids'])} existing chunks for page ID {page_id}")
        except Exception as e:
            logger.warning(f"Failed to delete chunks for page ID {page_id}: {e}")

    def search_and_answer(self, query: str, top_k: int = 6, current_page: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        2-LAYER AI RAG PIPELINE WITH PAGE AWARENESS:
        Layer 1: Intent Classification & Query Optimization
        Layer 2: ChromaDB Vector Retrieval (with Current Page context prioritization) + Gemini Answer Generation
        """
        # --- LAYER 1: INTENT ROUTER ---
        router_result = self.classify_and_route_intent(query)
        intent = router_result.get("intent", "SEARCH")
        search_query = router_result.get("optimized_query", query) or query

        logger.info(f"Intent Router Result -> Intent: {intent}, Search Query: '{search_query}'")

        catalog_info = self._get_indexed_catalog()
        catalog_summary = catalog_info.get("summary", "")
        all_pages = catalog_info.get("pages", {})

        # ROUTE 1: GREETING INTENT
        if intent == "GREETING":
            answer = self.generate_llm_response(query, f"DOCUMENT CATALOG:\n{catalog_summary}\nUser greeted you. Welcome them warmly.")
            return {
                "answer": answer,
                "sources": []
            }

        # ROUTE 2: OVERVIEW INTENT
        if intent == "OVERVIEW":
            context_str = f"=== FULL BOOKSTACK LIBRARY & HIERARCHY CATALOG ===\n{catalog_summary}"
            answer = self.generate_llm_response(query, context_str)
            sources = [{"page_id": pid, "title": info["title"], "url": info["url"]} for pid, info in all_pages.items()]
            return {
                "answer": answer,
                "sources": sources
            }

        # ROUTE 3: SEARCH INTENT (with Page Awareness)
        current_page_context = ""
        current_page_id = current_page.get("page_id") if current_page else None
        current_page_title = current_page.get("title") if current_page else None
        current_page_url = current_page.get("url") if current_page else None

        if current_page_id:
            try:
                active_meta = self.collection.get(where={"page_id": int(current_page_id)}, include=["documents"])
                if active_meta and active_meta.get("documents"):
                    page_docs = active_meta["documents"]
                    current_page_context = f"=== ACTIVE PAGE CONTEXT ===\nPage Title: '{current_page_title}'\nURL: {current_page_url}\n\n" + "\n\n".join(page_docs)
            except Exception as e:
                logger.warning(f"Could not fetch active page chunks for ID {current_page_id}: {e}")

        if not current_page_context and current_page_title:
            current_page_context = f"=== ACTIVE PAGE CONTEXT ===\nPage Title: '{current_page_title}'\nURL: {current_page_url}"

        results = self.collection.query(
            query_texts=[search_query],
            n_results=top_k
        )

        documents = results["documents"][0] if (results and results.get("documents")) else []
        metadatas = results["metadatas"][0] if (results and results.get("metadatas")) else []

        context_parts = []
        if current_page_context:
            context_parts.append(current_page_context)

        sources_map = {}
        for doc, meta in zip(documents, metadatas):
            context_parts.append(f"{doc}")
            page_id = meta.get("page_id")
            if page_id and page_id not in sources_map:
                sources_map[page_id] = {
                    "page_id": page_id,
                    "title": meta.get("name"),
                    "url": meta.get("url")
                }

        context_str = f"=== FULL BOOKSTACK LIBRARY & HIERARCHY CATALOG ===\n{catalog_summary}\n\n=== RELEVANT CONTEXT & ARTICLES ===\n" + "\n\n".join(context_parts)
        answer = self.generate_llm_response(query, context_str)

        return {
            "answer": answer,
            "sources": list(sources_map.values())
        }
