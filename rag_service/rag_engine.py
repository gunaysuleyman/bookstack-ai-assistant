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

    def _get_indexed_catalog(self, allowed_page_ids: Optional[List[int]] = None) -> Dict[str, Any]:
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
        full_prompt = f"{system_instruction}\n\n{user_prompt}"

        if self.provider == "gemini":
            models_to_try = ["gemini-2.5-flash", "gemini-flash-latest"]
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
            "CRITICAL NOTE FOR 'optimized_query': The documentation is in English. If the user asks in Turkish or other languages, ALWAYS include key English translation terms and synonyms alongside original terms (e.g., if user asks 'bilgisayarım bozuldu' or 'donanım arızası', include 'laptop computer not working broken hardware equipment IT support contact').\n\n"
            f"{history_snippet}User Query: '{query}'\n\n"
            "Respond ONLY with valid JSON, e.g. {\"intent\": \"SEARCH\", \"optimized_query\": \"bilgisayar bozuldu laptop computer not working broken hardware IT support who to contact\"}"
        )

        try:
            res_text = self._call_llm_api("Return valid JSON only.", router_prompt)
            clean_json = res_text.strip().lstrip("```json").rstrip("```").strip()
            data = json.loads(clean_json)
            return data
        except Exception as e:
            logger.warning(f"Intent Router fallback due to parsing error: {e}")
            q_lower = query.strip().lower()
            if any(w in q_lower for w in ["bu sayfa", "this page", "bu makale", "this article", "burada", "buradaki", "özet", "özetle", "ne biliyorsun"]):
                return {"intent": "SEARCH", "optimized_query": query}
            elif q_lower in ["hello", "hi", "hey", "merhaba", "selam", "günaydın", "iyi günler", "thanks", "teşekkürler"]:
                return {"intent": "GREETING", "optimized_query": query}
            elif any(w in q_lower for w in ["kaç", "hangi", "makale", "doküman", "sayfa", "kitap", "raf", "bölüm", "etiket", "liste", "list", "how many", "which", "books", "pages", "shelves"]):
                return {"intent": "OVERVIEW", "optimized_query": query}
            else:
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
            "4. INTELLIGENT DOMAIN FALLBACK RULE: If a user asks about a topic or issue that does NOT have a specific step-by-step article in the provided documentation:\n"
            "   - If the provided documentation contains a responsibility/contact guide (e.g. WHO TO CONTACT), use that guide to direct the user to the appropriate contact person.\n"
            "   - If no relevant document or contact guide is present in the provided context, state clearly and politely that there is no accessible documentation for this topic in the system.\n"
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
        except Exception as e:
            logger.warning(f"Failed to delete chunks for page ID {page_id}: {e}")

    def search_and_answer(self, query: str, top_k: int = 6, current_page: Optional[Dict[str, Any]] = None, allowed_page_ids: Optional[List[int]] = None, history: Optional[List[Dict[str, str]]] = None) -> Dict[str, Any]:
        """
        2-LAYER AI RAG PIPELINE WITH PERMISSION FILTERING, PAGE AWARENESS & ACCURATE CITATIONS
        """
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

        catalog_info = self._get_indexed_catalog(allowed_page_ids=allowed_page_ids)
        catalog_summary = catalog_info.get("summary", "")
        all_pages = catalog_info.get("pages", {})

        # ROUTE 1: GREETING INTENT
        if intent == "GREETING":
            answer = self.generate_llm_response(query, f"DOCUMENT CATALOG:\n{catalog_summary}\nUser greeted you. Welcome them warmly.", history=history)
            return {
                "answer": answer,
                "sources": []
            }

        # ROUTE 2: OVERVIEW INTENT
        if intent == "OVERVIEW":
            context_str = f"=== FULL BOOKSTACK LIBRARY & HIERARCHY CATALOG ===\n{catalog_summary}"
            answer = self.generate_llm_response(query, context_str, history=history)
            sources = [{"page_id": pid, "title": info["title"], "url": info["url"]} for pid, info in all_pages.items()]
            return {
                "answer": answer,
                "sources": sources
            }

        # ROUTE 3: SEARCH INTENT
        results = self.collection.query(
            query_texts=[search_query],
            n_results=top_k * 2 if allowed_page_ids is not None else top_k,
            include=["documents", "metadatas", "distances"]
        )

        documents = results["documents"][0] if (results and results.get("documents")) else []
        metadatas = results["metadatas"][0] if (results and results.get("metadatas")) else []
        distances = results["distances"][0] if (results and results.get("distances")) else []

        search_parts = []
        primary_sources_map = {}

        for doc, meta, dist in zip(documents, metadatas, distances):
            page_id = meta.get("page_id")
            # Permission check: skip if page is not permitted
            if allowed_page_ids is not None and (page_id is None or page_id not in allowed_page_ids):
                continue

            search_parts.append(f"{doc}")
            if page_id and page_id not in primary_sources_map and dist <= 0.85:
                primary_sources_map[page_id] = {
                    "page_id": page_id,
                    "title": meta.get("name"),
                    "url": meta.get("url")
                }

        if not primary_sources_map and metadatas:
            for meta in metadatas:
                pid = meta.get("page_id")
                if allowed_page_ids is not None and (pid is None or pid not in allowed_page_ids):
                    continue
                if pid and pid not in primary_sources_map:
                    primary_sources_map[pid] = {
                        "page_id": pid,
                        "title": meta.get("name"),
                        "url": meta.get("url")
                    }
                if len(primary_sources_map) >= 2:
                    break

        # Handle Active Page Context (always inject if user is currently reading an authorized page)
        q_lower = query.lower()
        is_page_summary_request = any(w in q_lower for w in ["bu sayfa", "this page", "bu makale", "this article", "özetle", "summarize", "buradaki", "bu doküman", "burada"])
        
        current_page_context = ""
        current_page_id = current_page.get("page_id") if current_page else None
        current_page_title = current_page.get("title") if current_page else None
        current_page_url = current_page.get("url") if current_page else None
        active_page_loaded = False

        if current_page_id:
            page_permitted = (allowed_page_ids is None) or (int(current_page_id) in allowed_page_ids)
            if page_permitted:
                try:
                    active_meta = self.collection.get(where={"page_id": int(current_page_id)}, include=["documents"])
                    if active_meta and active_meta.get("documents"):
                        page_docs = active_meta["documents"]
                        current_page_context = (
                            f"=== CURRENT ACTIVE PAGE (User is currently reading this article in BookStack) ===\n"
                            f"Page Title: '{current_page_title}'\n"
                            f"Page ID: {current_page_id}\n"
                            f"URL: {current_page_url}\n\n"
                            + "\n\n".join(page_docs)
                        )
                        active_page_loaded = True
                except Exception as e:
                    logger.warning(f"Could not fetch active page chunks for ID {current_page_id}: {e}")
            else:
                logger.warning(f"Active page ID {current_page_id} is not in user's permitted pages. Skipping context injection.")

        context_parts = []
        if current_page_context:
            context_parts.append(current_page_context)

        context_parts.append(f"=== SEARCH RESULTS ACROSS ALL PERMITTED ARTICLES (Matches for user query) ===\n" + ("\n\n".join(search_parts) if search_parts else "No matching permitted documents found."))
        context_parts.append(f"=== FULL BOOKSTACK LIBRARY & HIERARCHY CATALOG ===\n{catalog_summary}")

        context_str = "\n\n".join(context_parts)

        answer = self.generate_llm_response(query, context_str, history=history)

        # Smart Hybrid Citation Management
        answer_lower = answer.lower()
        
        # 1) If active page was loaded: check if the answer references it or if question was about it
        if active_page_loaded and current_page_id:
            act_title_words = [w for w in (current_page_title or "").lower().split() if len(w) > 3]
            is_active_page_referenced = (
                is_page_summary_request
                or (current_page_title and current_page_title.lower() in answer_lower)
                or any(w in answer_lower for w in act_title_words)
                or any(w in q_lower for w in ["buradaki", "bu sayfa", "bu makale", "bu adım", "here", "this page", "bu doküman", "özet"])
            )
            if is_active_page_referenced or not primary_sources_map:
                active_entry = all_pages.get(int(current_page_id), {
                    "page_id": int(current_page_id),
                    "title": current_page_title or f"Page #{current_page_id}",
                    "url": current_page_url or f"/link/{current_page_id}"
                })
                primary_sources_map[int(current_page_id)] = {
                    "page_id": int(current_page_id),
                    "title": active_entry.get("title", current_page_title),
                    "url": active_entry.get("url", current_page_url)
                }

        # 2) Fallback / IT / Hardware routing: redirect citations to contact page
        has_it_fallback = "süleyman" in answer_lower or "who to contact" in answer_lower or "it support" in answer_lower or "bilgisayar" in q_lower or "laptop" in q_lower
        
        if has_it_fallback:
            contact_page_id = None
            for pid, info in all_pages.items():
                if "who to contact" in info["title"].lower() or "who is responsible" in info["title"].lower():
                    contact_page_id = pid
                    primary_sources_map[pid] = {
                        "page_id": pid,
                        "title": info["title"],
                        "url": info["url"]
                    }
            
            # If an IT / responsibility redirect occurred, clean out active page or unrelated project pages from citations
            if contact_page_id:
                for pid in list(primary_sources_map.keys()):
                    if pid != contact_page_id:
                        del primary_sources_map[pid]

        return {
            "answer": answer,
            "sources": list(primary_sources_map.values())
        }
