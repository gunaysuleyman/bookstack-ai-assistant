import os
import sqlite3
import hashlib
import base64
import logging
import httpx
from bs4 import BeautifulSoup
from typing import Dict, Optional, Tuple
from urllib.parse import urljoin, urlparse

from adaptive.provider import redact_secrets

logger = logging.getLogger("ImageProcessor")
IMAGE_PROMPT_VERSION = os.getenv("IMAGE_PROMPT_VERSION", "v1")

class ImageProcessor:
    def __init__(self, db_dir: Optional[str] = None):
        self.gemini_key = os.getenv("GEMINI_API_KEY", "")
        self.vision_model = os.getenv("GEMINI_VISION_MODEL") or os.getenv("GEMINI_MODEL", "gemini-3.8-flash")
        fallback_str = os.getenv("GEMINI_FALLBACK_MODELS", "gemini-flash-latest,gemini-3.6-flash")
        self.vision_fallbacks = [m.strip() for m in fallback_str.split(",") if m.strip()]
        self.bookstack_internal_url = os.getenv("BOOKSTACK_URL", "http://bookstack:80").rstrip("/")
        self.bookstack_external_url = os.getenv("BOOKSTACK_EXTERNAL_URL", "http://localhost:6875").rstrip("/")
        
        persist_dir = db_dir or os.getenv("CHROMA_PERSIST_DIR", "/app/chroma_db")
        os.makedirs(persist_dir, exist_ok=True)
        self.db_path = os.path.join(persist_dir, "image_descriptions.db")
        self._init_db()

    def _get_connection(self) -> sqlite3.Connection:
        return sqlite3.connect(self.db_path, check_same_thread=False)

    def _init_db(self):
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS page_images (
                        image_url TEXT PRIMARY KEY,
                        page_id INTEGER,
                        image_hash TEXT,
                        alt_text TEXT,
                        visual_description TEXT,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    );
                """)
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_page_images_page_id ON page_images(page_id);")
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_page_images_hash ON page_images(image_hash);")
                existing = {row[1] for row in cursor.execute("PRAGMA table_info(page_images)")}
                for name, decl in (
                    ("vision_model", "TEXT"),
                    ("prompt_version", "TEXT"),
                    ("etag", "TEXT"),
                    ("last_modified", "TEXT"),
                ):
                    if name not in existing:
                        cursor.execute(f"ALTER TABLE page_images ADD COLUMN {name} {decl}")
                conn.commit()
            logger.info(f"Image descriptions SQLite database initialized at {self.db_path}")
        except Exception as e:
            logger.error(f"Failed to initialize image descriptions SQLite DB: {e}")

    def get_internal_url(self, url: str) -> str:
        """Translates external or relative URLs to internal Docker network URL."""
        if not url:
            return ""
        if url.startswith(self.bookstack_external_url):
            return url.replace(self.bookstack_external_url, self.bookstack_internal_url, 1)
        if url.startswith("/"):
            return f"{self.bookstack_internal_url}{url}"
        return url

    def get_external_url(self, url: str) -> str:
        """Translates internal or relative URLs to external public URL."""
        if not url:
            return ""
        if url.startswith(self.bookstack_internal_url):
            return url.replace(self.bookstack_internal_url, self.bookstack_external_url, 1)
        if url.startswith("/"):
            return f"{self.bookstack_external_url}{url}"
        return url

    def _detect_mime_type(self, image_bytes: bytes, url: str) -> str:
        url_lower = url.lower()
        if url_lower.endswith(".png"):
            return "image/png"
        elif url_lower.endswith(".jpg") or url_lower.endswith(".jpeg"):
            return "image/jpeg"
        elif url_lower.endswith(".webp"):
            return "image/webp"
        elif url_lower.endswith(".gif"):
            return "image/gif"
        
        # Simple magic bytes fallback
        if image_bytes.startswith(b"\x89PNG\r\n\x1a\n"):
            return "image/png"
        elif image_bytes.startswith(b"\xff\xd8\xff"):
            return "image/jpeg"
        elif image_bytes.startswith(b"RIFF") and b"WEBP" in image_bytes[:16]:
            return "image/webp"
        elif image_bytes.startswith(b"GIF8"):
            return "image/gif"
        return "image/png"

    def analyze_image_with_gemini(self, image_bytes: bytes, mime_type: str, alt_text: str = "") -> str:
        """Calls Gemini 2.5 Flash with multimodal inline_data to generate structured OCR & UI description."""
        if not self.gemini_key:
            logger.warning("No GEMINI_API_KEY found, skipping vision analysis.")
            return ""

        models_to_try = [self.vision_model] + [m for m in self.vision_fallbacks if m != self.vision_model]
        base64_data = base64.b64encode(image_bytes).decode("utf-8")

        prompt_text = (
            "You are an expert AI documentation visual analyzer for a knowledge base RAG system.\n"
            "Analyze this technical screenshot/image thoroughly and concisely:\n"
            "1. SCREEN / UI CONTEXT: Identify what interface, window, dialog, page, or diagram is shown.\n"
            "2. INTERACTIVE ELEMENTS & STATE: List visible buttons, tabs, input fields, checkboxes (and whether they are checked/selected), dropdowns, and any highlighted/circled elements.\n"
            "3. EXACT OCR TEXT: Transcribe all readable text, labels, headers, error messages, and table content visible in the image.\n"
            "4. ACTION/PROCEDURE: Explain what action, setting, or workflow step this image illustrates.\n\n"
            f"Image Hint/Alt Text: '{alt_text}'\n"
            "Output your analysis in structured, clean Markdown with bullet points."
        )

        payload = {
            "contents": [
                {
                    "parts": [
                        {"text": prompt_text},
                        {
                            "inline_data": {
                                "mime_type": mime_type,
                                "data": base64_data
                            }
                        }
                    ]
                }
            ],
            "generationConfig": {
                "temperature": 0.2,
                "maxOutputTokens": 1000
            }
        }

        last_err = None
        for model_name in models_to_try:
            url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:generateContent"
            try:
                with httpx.Client(timeout=60.0) as client:
                    res = client.post(url, headers={"x-goog-api-key": self.gemini_key}, json=payload)
                    res.raise_for_status()
                    data = res.json()
                    candidates = data.get("candidates", [])
                    if candidates and "content" in candidates[0]:
                        parts = candidates[0]["content"].get("parts", [])
                        if parts and "text" in parts[0]:
                            return parts[0]["text"].strip()
            except Exception as exc:
                logger.warning(
                    "Vision analysis with model %s failed: %s",
                    model_name,
                    redact_secrets(str(exc), [self.gemini_key]),
                )
                last_err = exc

        logger.error("All Gemini Vision models failed: %s", redact_secrets(str(last_err), [self.gemini_key]))
        return ""

    def _bookstack_hosts(self) -> set:
        hosts = set()
        for base in (self.bookstack_internal_url, self.bookstack_external_url):
            host = urlparse(base).hostname
            if host:
                hosts.add(host.lower())
        return hosts

    def trusted_image_hosts(self) -> set:
        raw = os.getenv("IMAGE_TRUSTED_HOSTS", "")
        return {host.strip().lower() for host in raw.split(",") if host.strip()}

    def is_trusted_image_url(self, url: str) -> bool:
        host = urlparse(url).hostname if url else None
        return bool(host and host.lower() in self.trusted_image_hosts())

    def is_bookstack_url(self, url: str) -> bool:
        if not url:
            return False
        if url.startswith("/"):
            return True
        host = urlparse(url).hostname
        return bool(host and host.lower() in self._bookstack_hosts())

    def fetch_image_bytes(
        self,
        url: str,
        auth_headers: Optional[Dict[str, str]] = None,
        transport: Optional[httpx.BaseTransport] = None,
    ) -> Optional[Tuple[bytes, str, str]]:
        """Download an image without sending BookStack credentials off-host."""
        max_bytes = int(os.getenv("IMAGE_MAX_BYTES", "5000000"))
        timeout = float(os.getenv("IMAGE_TIMEOUT_SECONDS", "15"))
        allow_external = os.getenv("IMAGE_ALLOW_EXTERNAL", "0").strip().lower() in {"1", "true", "yes", "on"}
        current = url
        with httpx.Client(timeout=timeout, transport=transport, follow_redirects=False) as client:
            for _hop in range(3):
                bookstack = self.is_bookstack_url(current)
                trusted = self.is_trusted_image_url(current)
                if not bookstack and not trusted and not allow_external:
                    logger.info("Skipping non-BookStack image host: %s", urlparse(current).hostname)
                    return None
                request_url = self.get_internal_url(current) if bookstack else current
                headers = {}
                if bookstack and auth_headers:
                    headers["Authorization"] = auth_headers.get("Authorization", "")
                response = client.get(request_url, headers=headers)
                if response.status_code in {301, 302, 303, 307, 308}:
                    location = response.headers.get("location")
                    if not location:
                        return None
                    current = urljoin(request_url, location)
                    continue
                if response.status_code != 200:
                    logger.warning("Failed to download image %s: HTTP %s", request_url, response.status_code)
                    return None
                declared = response.headers.get("content-length")
                if declared and declared.isdigit() and int(declared) > max_bytes:
                    logger.warning("Image %s exceeds size limit", request_url)
                    return None
                payload = response.content
                if len(payload) > max_bytes:
                    logger.warning("Image %s exceeds size limit", request_url)
                    return None
                return payload, response.headers.get("etag", ""), response.headers.get("last-modified", "")
        return None

    def _save_image_cache(self, src: str, page_id: int, img_hash: str, alt: str, desc: str, etag: str, last_modified: str) -> None:
        with self._get_connection() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO page_images (
                    image_url, page_id, image_hash, alt_text, visual_description,
                    vision_model, prompt_version, etag, last_modified, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
                """,
                (src, page_id, img_hash, alt, desc, self.vision_model, os.getenv("IMAGE_PROMPT_VERSION", IMAGE_PROMPT_VERSION), etag, last_modified),
            )
            conn.commit()

    def process_page_images(
        self,
        page_id: int,
        html_content: str,
        auth_headers: Optional[Dict[str, str]] = None,
        transport: Optional[httpx.BaseTransport] = None,
    ) -> Dict[str, str]:
        """
        Parses page images, revalidates bytes, and analyzes changed images.
        Returns a dictionary mapping original image src -> visual_description.
        """
        if not html_content:
            return {}

        soup = BeautifulSoup(html_content, "html.parser")
        img_tags = soup.find_all("img")
        if not img_tags:
            return {}

        image_descriptions: Dict[str, str] = {}
        for img in img_tags:
            src = img.get("src", "").strip()
            alt = img.get("alt", "").strip() or img.get("title", "").strip()
            if not src:
                continue
            try:
                fetched = self.fetch_image_bytes(src, auth_headers=auth_headers, transport=transport)
                if not fetched:
                    continue
                img_bytes, etag, last_modified = fetched
                img_hash = hashlib.sha256(img_bytes).hexdigest()
                prompt_version = os.getenv("IMAGE_PROMPT_VERSION", IMAGE_PROMPT_VERSION)
                with self._get_connection() as conn:
                    cached = conn.execute(
                        """
                        SELECT visual_description FROM page_images
                        WHERE image_hash = ? AND vision_model = ? AND prompt_version = ?
                          AND visual_description IS NOT NULL AND visual_description != ''
                        """,
                        (img_hash, self.vision_model, prompt_version),
                    ).fetchone()
                if cached and cached[0]:
                    self._save_image_cache(src, page_id, img_hash, alt, cached[0], etag, last_modified)
                    image_descriptions[src] = cached[0]
                    continue
                mime_type = self._detect_mime_type(img_bytes, src)
                desc = self.analyze_image_with_gemini(img_bytes, mime_type, alt_text=alt)
                if desc:
                    self._save_image_cache(src, page_id, img_hash, alt, desc, etag, last_modified)
                    image_descriptions[src] = desc
            except Exception as exc:
                logger.error("Error processing image %s for page %s: %s", src, page_id, exc)
        return image_descriptions
