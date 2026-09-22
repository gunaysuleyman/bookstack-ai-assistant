import os
import logging
import httpx
from typing import Dict, Any, List, Optional
from adaptive.contracts import PageDocument
from html_cleaner import HTMLCleaner
from rag_engine import RAGEngine
from image_processor import ImageProcessor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("BookStackSync")

class SyncError(Exception):
    """BookStack read or credential failure. Callers must retry; this is not a successful sync."""


class BookStackSync:
    def __init__(self, rag_engine: RAGEngine):
        self.bookstack_url = os.getenv("BOOKSTACK_URL", "http://bookstack:80").rstrip("/")
        self.external_url = os.getenv("BOOKSTACK_EXTERNAL_URL", "http://localhost:6875").rstrip("/")
        self.token_id = os.getenv("BOOKSTACK_TOKEN_ID", "")
        self.token_secret = os.getenv("BOOKSTACK_TOKEN_SECRET", "")
        self.cleaner = HTMLCleaner()
        self.rag_engine = rag_engine
        self.image_processor = ImageProcessor(db_dir=self.rag_engine.chroma_dir)
        
        self.book_cache: Dict[int, Dict[str, Any]] = {}
        self.chapter_cache: Dict[int, str] = {}

    def _get_headers(self) -> Dict[str, str]:
        return {
            "Authorization": f"Token {self.token_id}:{self.token_secret}",
            "Content-Type": "application/json"
        }

    def _require_credentials(self) -> None:
        if not self.token_id or not self.token_secret:
            raise SyncError("BookStack token id or secret is missing")

    def _get_book_details(self, book_id: int, client: httpx.Client) -> Dict[str, Any]:
        """Fetches parent book title and associated shelf names with caching."""
        if not book_id:
            return {"book_name": "General Library", "shelf_name": "General Shelf", "shelf_names": []}
        if book_id in self.book_cache:
            return self.book_cache[book_id]
        try:
            url = f"{self.bookstack_url}/api/books/{book_id}"
            res = client.get(url, headers=self._get_headers())
            if res.status_code == 200:
                data = res.json()
                bname = data.get("name", f"Book #{book_id}")
                
                # Fetch parent shelf if exists
                shelves = data.get("shelves", []) or []
                names = []
                if isinstance(shelves, list):
                    for shelf in shelves:
                        if isinstance(shelf, dict) and shelf.get("name"):
                            names.append(str(shelf["name"]))
                sname = " | ".join(names) if names else "General Shelf"
                res_dict = {"book_name": bname, "shelf_name": sname, "shelf_names": names}
                self.book_cache[book_id] = res_dict
                return res_dict
        except Exception as e:
            logger.warning(f"Could not fetch book {book_id}: {e}")
        
        res_dict = {"book_name": f"Book #{book_id}", "shelf_name": "General Shelf", "shelf_names": []}
        self.book_cache[book_id] = res_dict
        return res_dict

    def _get_chapter_name(self, chapter_id: int, client: httpx.Client) -> str:
        """Fetches chapter title by chapter_id with caching."""
        if not chapter_id:
            return "General Chapter"
        if chapter_id in self.chapter_cache:
            return self.chapter_cache[chapter_id]
        try:
            url = f"{self.bookstack_url}/api/chapters/{chapter_id}"
            res = client.get(url, headers=self._get_headers())
            if res.status_code == 200:
                name = res.json().get("name", f"Chapter #{chapter_id}")
                self.chapter_cache[chapter_id] = name
                return name
        except Exception as e:
            logger.warning(f"Could not fetch chapter {chapter_id}: {e}")
        return f"Chapter #{chapter_id}"

    def load_page(self, page_id: int) -> Optional[PageDocument]:
        """Read one page. 404 returns None. Credential, HTTP, and network failures raise SyncError."""
        self._require_credentials()
        url = f"{self.bookstack_url}/api/pages/{page_id}"
        try:
            with httpx.Client(timeout=30.0) as client:
                res = client.get(url, headers=self._get_headers())
                if res.status_code == 404:
                    logger.info(f"Page {page_id} not found in BookStack.")
                    return None
                if res.status_code >= 400:
                    raise SyncError(f"BookStack page {page_id} failed with HTTP {res.status_code}")
                page_data = res.json()

                book_id = page_data.get("book_id")
                book_info = self._get_book_details(book_id, client) if book_id else {"book_name": "General Library", "shelf_name": "General Shelf", "shelf_names": []}
                page_data["book_name"] = book_info["book_name"]
                page_data["shelf_name"] = book_info["shelf_name"]
                page_data["shelf_names"] = book_info.get("shelf_names") or []

                chapter_id = page_data.get("chapter_id")
                page_data["chapter_name"] = self._get_chapter_name(chapter_id, client) if chapter_id else "General Chapter"

                raw_tags = page_data.get("tags", [])
                tag_parts = []
                if raw_tags and isinstance(raw_tags, list):
                    for tag in raw_tags:
                        tag_name = tag.get("name", "").strip()
                        tag_value = tag.get("value", "").strip()
                        if tag_name and tag_value:
                            tag_parts.append(f"{tag_name}: {tag_value}")
                        elif tag_name:
                            tag_parts.append(tag_name)
                page_data["tags_str"] = ", ".join(tag_parts)

            page_data["url"] = f"{self.external_url}/link/{page_id}"
            html_content = page_data.get("html", "")
            image_descriptions = {}
            if html_content:
                try:
                    image_descriptions = self.image_processor.process_page_images(
                        page_id=page_id,
                        html_content=html_content,
                        auth_headers=self._get_headers()
                    )
                    if image_descriptions:
                        logger.info(f"Page ID {page_id}: Processed {len(image_descriptions)} images with visual descriptions.")
                except Exception as img_err:
                    logger.warning(f"Image processing failed for page {page_id}: {img_err}")

            raw_markdown = page_data.get("markdown", "")
            if raw_markdown and isinstance(raw_markdown, str) and raw_markdown.strip() and not image_descriptions:
                markdown = raw_markdown.strip()
                logger.info(f"Page ID {page_id}: Used direct Markdown from BookStack API.")
            else:
                markdown = self.cleaner.clean_to_markdown(html_content, image_descriptions=image_descriptions)
                logger.info(f"Page ID {page_id}: Converted HTML content to Markdown with visual context.")

            return PageDocument(
                page_id=int(page_id),
                name=str(page_data.get("name") or ""),
                markdown=markdown,
                book_id=int(page_data.get("book_id") or 0),
                book_name=str(page_data.get("book_name") or "General Library"),
                chapter_id=int(page_data.get("chapter_id") or 0),
                chapter_name=str(page_data.get("chapter_name") or "General Chapter"),
                shelf_names=list(page_data.get("shelf_names") or []),
                tags_str=str(page_data.get("tags_str") or ""),
                url=str(page_data.get("url") or ""),
                updated_at=str(page_data.get("updated_at") or ""),
            )
        except SyncError:
            raise
        except Exception as exc:
            logger.error(f"Error syncing page ID {page_id}: {exc}")
            raise SyncError(f"Error syncing page ID {page_id}: {exc}") from exc

    def index_loaded_page(self, page: PageDocument) -> None:
        page_data = {
            "id": page.page_id,
            "name": page.name,
            "book_id": page.book_id,
            "book_name": page.book_name,
            "chapter_id": page.chapter_id,
            "chapter_name": page.chapter_name,
            "shelf_name": page.shelf_label(),
            "shelf_names": list(page.shelf_names),
            "tags_str": page.tags_str,
            "url": page.url,
            "updated_at": page.updated_at,
        }
        chunks = self.cleaner.chunk_markdown(page.markdown, page_data)
        self.rag_engine.add_page_chunks(page.page_id, chunks)

    def sync_single_page(self, page_id: int) -> None:
        """Syncs a single page by ID with full 4-tier hierarchy metadata."""
        page = self.load_page(page_id)
        if page is None:
            logger.info(f"Page {page_id} not found in BookStack. Removing from vector index.")
            self.rag_engine.delete_page(page_id)
            return
        self.index_loaded_page(page)
        logger.info(
            f"Synced page ID {page_id}: '{page.name}' "
            f"under Shelf: '{page.shelf_label()}' > "
            f"Book: '{page.book_name}' > "
            f"Chapter: '{page.chapter_name}'"
        )

    def list_page_stubs(self) -> List[dict]:
        """List id and updated_at for every page. HTTP and network failures raise SyncError."""
        self._require_credentials()
        url = f"{self.bookstack_url}/api/pages"
        pages: List[dict] = []
        offset = 0
        count = 100
        try:
            with httpx.Client(timeout=30.0) as client:
                while True:
                    res = client.get(f"{url}?count={count}&offset={offset}", headers=self._get_headers())
                    if res.status_code >= 400:
                        raise SyncError(f"BookStack page list failed with HTTP {res.status_code}")
                    data = res.json()
                    batch = data.get("data") or []
                    if not batch:
                        break
                    for page in batch:
                        pages.append({"id": int(page["id"]), "updated_at": str(page.get("updated_at") or "")})
                    offset += len(batch)
                    total = int(data.get("total") or 0)
                    if offset >= total:
                        break
        except SyncError:
            raise
        except Exception as exc:
            logger.error(f"Failed to list BookStack pages: {exc}")
            raise SyncError(f"Failed to list BookStack pages: {exc}") from exc
        return pages

    def delete_page(self, page_id: int):
        """Deletes page chunks from vector store."""
        self.rag_engine.delete_page(page_id)

    def sync_all_pages(self) -> int:
        """Syncs all pages from BookStack REST API. A failed page fails the run."""
        stubs = self.list_page_stubs()
        for stub in stubs:
            self.sync_single_page(int(stub["id"]))
        logger.info(f"Full synchronization finished. Total pages synced: {len(stubs)}")
        return len(stubs)
