import os
import json
import logging
import chromadb
from chromadb.config import Settings as ChromaSettings
from typing import List, Dict, Any, Optional

from adaptive.embeddings import build_embedding
from adaptive.provider import complete_llm

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("RAGEngine")

class RAGEngine:
    def __init__(self, embedding_fn=None, collection_name: Optional[str] = None):
        self.provider = os.getenv("AI_PROVIDER", "gemini").lower()
        self.gemini_key = os.getenv("GEMINI_API_KEY", "")
        self.openai_key = os.getenv("OPENAI_API_KEY", "")
        self.gemini_model = os.getenv("GEMINI_MODEL", "gemini-3.8-flash")
        fallback_str = os.getenv("GEMINI_FALLBACK_MODELS", "gemini-flash-latest,gemini-3.6-flash")
        self.gemini_fallbacks = [m.strip() for m in fallback_str.split(",") if m.strip()]
        self.openai_model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
        self.chroma_dir = os.getenv("CHROMA_PERSIST_DIR", "/app/chroma_db")
        self.metadata_scans = 0
        self.last_usage = None

        self.embedding_fn = embedding_fn or build_embedding()

        self.chroma_client = chromadb.PersistentClient(
            path=self.chroma_dir,
            settings=ChromaSettings(anonymized_telemetry=False),
        )
        self.collection_name = collection_name or os.getenv("LEGACY_COLLECTION", "bookstack_articles")
        self.collection = self._get_or_create_collection()

    def _get_or_create_collection(self):
        model_id = "custom"
        name = getattr(self.embedding_fn, "name", None)
        if callable(name):
            try:
                model_id = str(name())
            except TypeError:
                model_id = str(name)
        metadata = {"hnsw:space": "cosine", "embedding_model": model_id}
        try:
            existing = self.chroma_client.get_collection(name=self.collection_name)
        except Exception:
            existing = None
        if existing is not None and (existing.metadata or {}).get("embedding_model") != model_id:
            self.chroma_client.delete_collection(self.collection_name)
        return self.chroma_client.get_or_create_collection(
            name=self.collection_name,
            embedding_function=self.embedding_fn,
            metadata=metadata,
        )

    def _get_indexed_catalog(self, allowed_page_ids: Optional[List[int]] = None) -> Dict[str, Any]:
        self.metadata_scans += 1
        try:
            all_meta = self.collection.get(include=["metadatas"])
            tree_map = {}
            all_pages_map = {}

            if all_meta and all_meta.get("metadatas"):
                for meta in all_meta["metadatas"]:
                    pid = meta.get("page_id")
                    if allowed_page_ids is not None and pid not in allowed_page_ids:
                        continue
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
        if self.provider not in {"gemini", "openai"}:
            raise ValueError(f"Unsupported AI_PROVIDER: {self.provider}")
        result = complete_llm(
            provider=self.provider,
            model=self.gemini_model,
            fallbacks=self.gemini_fallbacks,
            api_key=self.gemini_key,
            openai_model=self.openai_model,
            openai_key=self.openai_key,
            system_instruction=system_instruction,
            user_prompt=user_prompt,
            purpose="legacy",
            timeout_s=float(os.getenv("LLM_TIMEOUT_SECONDS", "30")),
        )
        self.last_usage = result.usage
        return result.text

    def classify_and_route_intent(self, query: str, history: Optional[List[Dict[str, str]]] = None) -> Dict[str, Any]:
        """
        LAYER 1: Intent Router AI
        Classifies user query intent into:
        - 'GREETING': Smalltalk, hello, hi, merhaba, thanks
        - 'OVERVIEW': Asking about catalog, books count, shelves structure
        - 'SEARCH': Specific topic or document search
        And optimizes search query for vector retrieval using history if available.
        """
        history_snippet = ""
        if history:
            recent = history[-4:]
            history_lines = [f"- {m.get('role')}: {m.get('content')[:180]}" for m in recent]
            history_snippet = "Recent Conversation Context (Use this to resolve references like 'it', 'this', 'o', 'bu'):\n" + "\n".join(history_lines) + "\n\n"

        router_prompt = (
            "You are an AI Intent Router for a BookStack Documentation Assistant.\n"
            "Classify the user query into JSON format with keys:\n"
            "- 'intent': string ('GREETING' | 'OVERVIEW' | 'SEARCH')\n"
            "  * 'GREETING': Pure greetings only (hello, hi, hey, merhaba, selam, thanks).\n"
            "  * 'OVERVIEW': ONLY when the user asks for a global list of all shelves/books or library statistics (e.g. 'hangi kitaplıklar var', 'bütün sayfaları listele', 'sistemde neler var'). Questions about 'this page' / 'bu sayfa' or specific topics are NEVER OVERVIEW.\n"
            "  * 'SEARCH': For ANY question about a topic, procedure, or asking about a specific page or the current page (e.g. 'bu sayfa ile alakalı ne biliyorsun', 'bu nedir', 'özetle', 'nasıl yapılır' are ALWAYS SEARCH).\n"
            "- 'optimized_query': string (search query for vector retrieval if SEARCH, else empty).\n"
            "CRITICAL NOTE FOR 'optimized_query': Expand the query with relevant technical keywords, synonyms, and translations matching the documentation language to maximize vector search retrieval accuracy.\n\n"
            f"{history_snippet}User Query: '{query}'\n\n"
            "Respond ONLY with valid JSON, e.g. {\"intent\": \"SEARCH\", \"optimized_query\": \"expanded technical keywords\"}"
        )

        try:
            res_text = self._call_llm_api("Return valid JSON only.", router_prompt)
            clean_json = res_text.strip().lstrip("```json").rstrip("```").strip()
            data = json.loads(clean_json)
            return data
        except Exception as e:
            logger.warning(f"Intent Router parsing failed: {e}. Defaulting to standard search.")
            return {"intent": "SEARCH", "optimized_query": query}

    def generate_llm_response(self, prompt: str, context: str, history: Optional[List[Dict[str, str]]] = None) -> str:
        """LAYER 2: Generates final response based on retrieved context, history and system instructions."""
        system_instruction = (
            "You are an expert AI Assistant integrated into BookStack Documentation System.\n"
            "You have complete mastery over BookStack's 4-Tier Hierarchy: Shelves -> Books -> Chapters -> Pages and Tags.\n\n"
            "CRITICAL ANSWERING RULES:\n"
            "1. PRIMARY SOURCE: Rely strictly on the provided 'SEARCH RESULTS (MOST RELEVANT ARTICLES)' and 'DOCUMENT CATALOG' to answer questions.\n"
            "2. SPECIFIC TARGETING: If the user asks about a specific issue, answer directly from the matching article.\n"
            "3. STRICT PERMISSION BOUNDARY: The user is only authorized to see the articles provided in the context. You must ONLY answer based on these provided articles. Never mention, reference, assume, or invent articles, departments, or personnel that are not explicitly present in the provided context.\n"
            "4. MISSING DOCUMENTATION RULE: If a user asks about a topic or procedure that is not present in the provided documentation:\n"
            "   - If the documentation contains a relevant contact or support directory, guide the user to that department or person.\n"
            "   - Otherwise, state clearly and politely that there is no accessible documentation for this topic in the system.\n"
            "5. ACTIVE PAGE AWARENESS & HYBRID CONTEXT ROUTING:\n"
            "   - When 'CURRENT ACTIVE PAGE' is provided, the user is currently reading that specific article in BookStack.\n"
            "   - If the user's question relates to the topics, steps, instructions, requirements, or content on this active page, or uses contextual references (such as 'buradaki', 'bu adımlar', 'bu sayfa', 'bu işlem', 'here', 'these steps', 'bu doküman'), prioritize answering directly and thoroughly from the CURRENT ACTIVE PAGE context.\n"
            "   - If the user's question asks about a DIFFERENT topic, department, contact person, policy, or procedure that is NOT covered on the active page (for example, asking about IT support, hardware issues, or other project documents while on an unrelated page), rely on the SEARCH RESULTS (MOST RELEVANT ARTICLES) from other pages across the library. Answer clearly using that documentation.\n"
            "   - If the question connects both the active page and external articles, synthesize information from both smoothly and cite all relevant articles.\n"
            "6. CONVERSATION CONTINUITY: When 'RECENT CONVERSATION HISTORY' is provided, maintain context and continuity with earlier answers while staying strictly grounded in the documentation.\n"
            "7. LANGUAGE DYNAMICS: Match the language of the user's question. If the user asks in Turkish, reply in Turkish. If the user asks in English, reply in English.\n"
            "8. FORMATTING: Use clean markdown, bullet points, and bold terms for key names/titles.\n"
            "9. VISUAL INTELLIGENCE (NO IMAGE LINKS): When a document contains visual context blocks (marked by '📷 Visual Context & OCR'), treat these descriptions as if you have directly viewed the screenshot. Explain the visual interface, UI elements, and steps thoroughly and cleanly in text using bullet points. DO NOT output image markdown embeds (NEVER write ![...](url)) and DO NOT output raw image URLs. Refer to images simply as 'Ekran görüntüsünde...', 'Görsel 1'de...', etc."
        )

        history_text = ""
        if history:
            recent_turns = history[-6:]
            history_lines = [f"- {m.get('role', 'user').capitalize()}: {m.get('content', '')}" for m in recent_turns]
            history_text = "=== RECENT CONVERSATION HISTORY (Previous turns in this chat session) ===\n" + "\n".join(history_lines) + "\n\n"

        full_prompt = f"{history_text}--- CONTEXT & FULL HIERARCHY CATALOG ---\n{context}\n\n--- USER QUESTION ---\n{prompt}"
        return self._call_llm_api(system_instruction, full_prompt)

    def add_page_chunks(self, page_id: int, chunks: List[Dict[str, Any]]):
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
        try:
            results = self.collection.get(where={"page_id": page_id})
            if results and results.get("ids"):
                self.collection.delete(ids=results["ids"])
                logger.info(f"Deleted {len(results['ids'])} existing chunks for page ID {page_id}")
        except Exception as exc:
            logger.warning("Failed to delete chunks for page ID %s: %s", page_id, exc)
            raise

    def update_book_metadata(self, book_id: int, book_name: str, shelf_name: str) -> int:
        """Update hierarchy labels without embedding the page again."""
        found = self.collection.get(where={"book_id": int(book_id)}, include=["metadatas"])
        ids = found.get("ids") or []
        metas = found.get("metadatas") or []
        if not ids:
            return 0
        updated = []
        for meta in metas:
            copied = dict(meta)
            copied["book_name"] = book_name
            copied["shelf_name"] = shelf_name
            updated.append(copied)
        self.collection.update(ids=ids, metadatas=updated)
        return len(ids)

    def _flatten_query(self, result: Optional[Dict[str, Any]]) -> List[Dict[str, Any]]:
        if not result:
            return []
        docs = (result.get("documents") or [[]])[0]
        metas = (result.get("metadatas") or [[]])[0]
        distances = (result.get("distances") or [[]])[0]
        rows = []
        for doc, meta, dist in zip(docs, metas, distances):
            rows.append({"document": doc, "metadata": meta or {}, "distance": dist})
        return rows

    def _query_permitted(self, search_query: str, allowed_page_ids: Optional[List[int]], limit: int) -> List[Dict[str, Any]]:
        include = ["documents", "metadatas", "distances"]
        if allowed_page_ids is not None and len(allowed_page_ids) == 0:
            return []
        if allowed_page_ids is None:
            return self._flatten_query(
                self.collection.query(query_texts=[search_query], n_results=limit, include=include)
            )
        batch_size = int(os.getenv("ACL_FILTER_BATCH", "200"))
        rows: List[Dict[str, Any]] = []
        allowed = set(allowed_page_ids)
        for start in range(0, len(allowed_page_ids), batch_size):
            batch = allowed_page_ids[start : start + batch_size]
            try:
                result = self.collection.query(
                    query_texts=[search_query],
                    n_results=limit,
                    where={"page_id": {"$in": batch}},
                    include=include,
                )
            except Exception as exc:
                logger.warning("Filtered vector query failed: %s", exc)
                continue
            rows.extend(self._flatten_query(result))
        rows.sort(key=lambda item: item["distance"] if item["distance"] is not None else 999)
        filtered = []
        for row in rows:
            page_id = row["metadata"].get("page_id")
            if page_id not in allowed:
                continue
            filtered.append(row)
            if len(filtered) >= limit:
                break
        return filtered

    def search_and_answer(self, query: str, top_k: int = 6, current_page: Optional[Dict[str, Any]] = None, allowed_page_ids: Optional[List[int]] = None, history: Optional[List[Dict[str, str]]] = None) -> Dict[str, Any]:
        """
        2-LAYER AI RAG PIPELINE WITH PERMISSION FILTERING, PAGE AWARENESS & ACCURATE CITATIONS
        """
        self.last_usage = None
        # --- LAYER 1: INTENT ROUTER ---
        router_result = self.classify_and_route_intent(query, history=history)
        intent = router_result.get("intent", "SEARCH")
        search_query = router_result.get("optimized_query", query) or query

        # Force SEARCH intent if query is page-specific
        q_lower = query.strip().lower()
        is_page_request = any(w in q_lower for w in ["bu sayfa", "this page", "bu makale", "this article", "burada", "buradaki", "özet", "özetle", "summarize", "ne biliyorsun", "hakkında ne"])
        if is_page_request and intent == "OVERVIEW":
            logger.info("Routing override: query refers to active page, forcing SEARCH intent instead of OVERVIEW.")
            intent = "SEARCH"

        current_page_title = current_page.get("title") if current_page else None
        if is_page_request and current_page_title:
            if not any(w in search_query.lower() for w in current_page_title.lower().split() if len(w) > 3):
                search_query = f"{current_page_title} {search_query}"

        logger.info(f"Intent Router Result -> Intent: {intent}, Search Query: '{search_query}', Allowed Pages: {len(allowed_page_ids) if allowed_page_ids is not None else 'All'}")
        del top_k  # Client top_k is accepted by the API and does not set retrieval depth.

        if intent == "GREETING":
            try:
                answer = self.generate_llm_response(
                    query,
                    "The user greeted you. Reply briefly in their language. Do not list or invent documents.",
                    history=history,
                )
            except Exception:
                answer = "Merhaba. Belgelerinizde arama yapabilirim." if any(ch in query.lower() for ch in "çğıöşü") or "merhaba" in query.lower() else "Hello. I can search the documents you can access."
            return {"answer": answer, "sources": []}

        if intent == "OVERVIEW":
            catalog_info = self._get_indexed_catalog(allowed_page_ids=allowed_page_ids)
            catalog_summary = catalog_info.get("summary", "")
            all_pages = catalog_info.get("pages", {})
            answer = self.generate_llm_response(query, catalog_summary, history=history)
            sources = [
                {"page_id": pid, "title": info["title"], "url": info["url"]}
                for pid, info in all_pages.items()
                if allowed_page_ids is None or pid in allowed_page_ids
            ]
            return {"answer": answer, "sources": sources}

        server_limit = int(os.getenv("LEGACY_RESULT_LIMIT", "8"))
        rows = self._query_permitted(search_query, allowed_page_ids, server_limit)
        search_parts = []
        primary_sources_map = {}
        for row in rows:
            meta = row["metadata"]
            page_id = meta.get("page_id")
            if allowed_page_ids is not None and page_id not in allowed_page_ids:
                continue
            search_parts.append(row["document"])
            distance = row["distance"] if row["distance"] is not None else 1
            if page_id and page_id not in primary_sources_map and distance <= 0.85:
                primary_sources_map[page_id] = {
                    "page_id": page_id,
                    "title": meta.get("name"),
                    "url": meta.get("url"),
                }

        current_page_context = ""
        current_page_id = current_page.get("page_id") if current_page else None
        current_page_title = current_page.get("title") if current_page else None
        current_page_url = current_page.get("url") if current_page else None
        active_page_loaded = False
        if current_page_id:
            page_permitted = (allowed_page_ids is None) or (int(current_page_id) in allowed_page_ids)
            if page_permitted:
                try:
                    active_meta = self.collection.get(where={"page_id": int(current_page_id)}, include=["documents", "metadatas"])
                    if active_meta and active_meta.get("documents"):
                        current_page_context = (
                            f"=== CURRENT ACTIVE PAGE (User is currently reading this article in BookStack) ===\n"
                            f"Page Title: '{current_page_title}'\n"
                            f"Page ID: {current_page_id}\n"
                            f"URL: {current_page_url}\n\n"
                            + "\n\n".join(active_meta["documents"])
                        )
                        active_page_loaded = True
                except Exception as exc:
                    logger.warning(f"Could not fetch active page chunks for ID {current_page_id}: {exc}")
            else:
                logger.warning(f"Active page ID {current_page_id} is not in user's permitted pages. Skipping context injection.")

        context_parts = []
        if current_page_context:
            context_parts.append(current_page_context)
        context_parts.append(
            "=== SEARCH RESULTS (permitted articles) ===\n"
            + ("\n\n".join(search_parts) if search_parts else "No matching permitted documents found.")
        )
        answer = self.generate_llm_response(query, "\n\n".join(context_parts), history=history)

        if active_page_loaded and current_page_id and is_page_request:
            primary_sources_map[int(current_page_id)] = {
                "page_id": int(current_page_id),
                "title": current_page_title or f"Page #{current_page_id}",
                "url": current_page_url or f"/link/{current_page_id}",
            }
        return {"answer": answer, "sources": list(primary_sources_map.values())}
