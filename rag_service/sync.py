import os
import logging
import httpx
from typing import Dict, Any, List
from html_cleaner import HTMLCleaner
from rag_engine import RAGEngine

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("BookStackSync")

class BookStackSync:
    def __init__(self, rag_engine: RAGEngine):
        self.bookstack_url = os.getenv("BOOKSTACK_URL", "http://bookstack:80").rstrip("/")
        self.external_url = os.getenv("BOOKSTACK_EXTERNAL_URL", "http://localhost:6875").rstrip("/")
        self.token_id = os.getenv("BOOKSTACK_TOKEN_ID", "")
        self.token_secret = os.getenv("BOOKSTACK_TOKEN_SECRET", "")
        self.cleaner = HTMLCleaner()
        self.rag_engine = rag_engine
        
        self.book_cache: Dict[int, Dict[str, Any]] = {}
        self.chapter_cache: Dict[int, str] = {}

    def _get_headers(self) -> Dict[str, str]:
        return {
            "Authorization": f"Token {self.token_id}:{self.token_secret}",
            "Content-Type": "application/json"
        }

    def _get_book_details(self, book_id: int, client: httpx.Client) -> Dict[str, Any]:
        """Fetches parent book title and associated shelf names with caching."""
        if not book_id:
            return {"book_name": "Genel Kütüphane", "shelf_name": "Genel Raf"}
        if book_id in self.book_cache:
            return self.book_cache[book_id]
        try:
            url = f"{self.bookstack_url}/api/books/{book_id}"
            res = client.get(url, headers=self._get_headers())
            if res.status_code == 200:
                data = res.json()
                bname = data.get("name", f"Kitap #{book_id}")
                
                # Fetch parent shelf if exists
                shelves = data.get("shelves", [])
                sname = shelves[0]["name"] if (shelves and isinstance(shelves, list) and len(shelves) > 0 and "name" in shelves[0]) else "Genel Raf"
                
                res_dict = {"book_name": bname, "shelf_name": sname}
                self.book_cache[book_id] = res_dict
                return res_dict
        except Exception as e:
            logger.warning(f"Could not fetch book {book_id}: {e}")
        
        res_dict = {"book_name": f"Kitap #{book_id}", "shelf_name": "Genel Raf"}
        self.book_cache[book_id] = res_dict
        return res_dict

    def _get_chapter_name(self, chapter_id: int, client: httpx.Client) -> str:
        """Fetches chapter title by chapter_id with caching."""
        if not chapter_id:
            return "Genel Bölüm"
        if chapter_id in self.chapter_cache:
            return self.chapter_cache[chapter_id]
        try:
            url = f"{self.bookstack_url}/api/chapters/{chapter_id}"
            res = client.get(url, headers=self._get_headers())
            if res.status_code == 200:
                name = res.json().get("name", f"Bölüm #{chapter_id}")
                self.chapter_cache[chapter_id] = name
                return name
        except Exception as e:
            logger.warning(f"Could not fetch chapter {chapter_id}: {e}")
        return f"Bölüm #{chapter_id}"

    def sync_single_page(self, page_id: int):
        """Syncs a single page by ID with full 4-tier hierarchy metadata."""
        if not self.token_id or not self.token_secret:
            logger.error("BookStack Token ID or Secret missing! Cannot sync.")
            return

        url = f"{self.bookstack_url}/api/pages/{page_id}"
        try:
            with httpx.Client(timeout=30.0) as client:
                res = client.get(url, headers=self._get_headers())
                if res.status_code == 404:
                    logger.info(f"Page {page_id} not found in BookStack. Removing from vector index.")
                    self.rag_engine.delete_page(page_id)
                    return
                res.raise_for_status()
                page_data = res.json()

                # Fetch parent Book and Shelf details
                book_id = page_data.get("book_id")
                book_info = self._get_book_details(book_id, client) if book_id else {"book_name": "Genel Kütüphane", "shelf_name": "Genel Raf"}
                page_data["book_name"] = book_info["book_name"]
                page_data["shelf_name"] = book_info["shelf_name"]

                # Fetch parent Chapter details
                chapter_id = page_data.get("chapter_id")
                page_data["chapter_name"] = self._get_chapter_name(chapter_id, client) if chapter_id else "Genel Bölüm"

                # Process tags
                raw_tags = page_data.get("tags", [])
                tag_parts = []
                if raw_tags and isinstance(raw_tags, list):
                    for t in raw_tags:
                        t_name = t.get("name", "").strip()
                        t_val = t.get("value", "").strip()
                        if t_name and t_val:
                            tag_parts.append(f"{t_name}: {t_val}")
                        elif t_name:
                            tag_parts.append(t_name)
                page_data["tags_str"] = ", ".join(tag_parts)

            # Direct BookStack permalink: /link/{page_id}
            page_data["url"] = f"{self.external_url}/link/{page_id}"

            # Check if BookStack API returned direct Markdown first
            raw_markdown = page_data.get("markdown", "")
            if raw_markdown and isinstance(raw_markdown, str) and raw_markdown.strip():
                markdown = raw_markdown.strip()
                logger.info(f"Page ID {page_id}: Used direct Markdown from BookStack API.")
            else:
                html_content = page_data.get("html", "")
                markdown = self.cleaner.clean_to_markdown(html_content)
                logger.info(f"Page ID {page_id}: Converted HTML content to Markdown.")

            chunks = self.cleaner.chunk_markdown(markdown, page_data)

            self.rag_engine.add_page_chunks(page_id, chunks)
            logger.info(
                f"Synced page ID {page_id}: '{page_data.get('name')}' "
                f"under Shelf: '{page_data.get('shelf_name')}' > "
                f"Book: '{page_data.get('book_name')}' > "
                f"Chapter: '{page_data.get('chapter_name')}'"
            )

        except Exception as e:
            logger.error(f"Error syncing page ID {page_id}: {e}")

    def delete_page(self, page_id: int):
        """Deletes page chunks from vector store."""
        self.rag_engine.delete_page(page_id)

    def sync_all_pages(self) -> int:
        """Syncs all pages from BookStack REST API."""
        if not self.token_id or not self.token_secret:
            logger.warning("BookStack Token ID or Secret missing! Skipping API sync.")
            return 0

        url = f"{self.bookstack_url}/api/pages"
        total_synced = 0
        count = 100
        offset = 0

        try:
            with httpx.Client(timeout=30.0) as client:
                while True:
                    res = client.get(f"{url}?count={count}&offset={offset}", headers=self._get_headers())
                    res.raise_for_status()
                    data = res.json()

                    pages = data.get("data", [])
                    if not pages:
                        break

                    for page in pages:
                        page_id = page["id"]
                        self.sync_single_page(page_id)
                        total_synced += 1

                    offset += len(pages)
                    if offset >= data.get("total", 0):
                        break

            logger.info(f"Full synchronization finished. Total pages synced: {total_synced}")
            return total_synced

        except Exception as e:
            logger.error(f"Failed full sync from BookStack: {e}")
            return total_synced
