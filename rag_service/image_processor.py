import os
import sqlite3
import hashlib
import base64
import logging
import httpx
from bs4 import BeautifulSoup
from typing import Dict, Any, Optional, List

logger = logging.getLogger("ImageProcessor")

class ImageProcessor:
    def __init__(self, db_dir: Optional[str] = None):
        self.gemini_key = os.getenv("GEMINI_API_KEY", "")
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

        models_to_try = ["gemini-3.6-flash", "gemini-flash-latest", "gemini-2.5-flash"]
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
            url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:generateContent?key={self.gemini_key}"
            try:
                with httpx.Client(timeout=60.0) as client:
                    res = client.post(url, json=payload)
                    res.raise_for_status()
                    data = res.json()
                    candidates = data.get("candidates", [])
                    if candidates and "content" in candidates[0]:
                        parts = candidates[0]["content"].get("parts", [])
                        if parts and "text" in parts[0]:
                            return parts[0]["text"].strip()
            except Exception as e:
                logger.warning(f"Vision analysis with model {model_name} failed: {e}")
                last_err = e

        logger.error(f"All Gemini Vision models failed. Last error: {last_err}")
        return ""

    def process_page_images(self, page_id: int, html_content: str, auth_headers: Optional[Dict[str, str]] = None) -> Dict[str, str]:
        """
        Parses all images in page HTML, checks cache, downloads and analyzes new images using Gemini 2.5 Flash.
        Returns a dictionary mapping original image src -> visual_description.
        """
        if not html_content:
            return {}

        soup = BeautifulSoup(html_content, "html.parser")
        img_tags = soup.find_all("img")
        if not img_tags:
            return {}

        image_descriptions: Dict[str, str] = {}

        with self._get_connection() as conn:
            cursor = conn.cursor()

            for img in img_tags:
                src = img.get("src", "").strip()
                alt = img.get("alt", "").strip() or img.get("title", "").strip()
                if not src:
                    continue

                # Check if already cached by image_url
                cursor.execute("SELECT visual_description, image_hash FROM page_images WHERE image_url = ?", (src,))
                row = cursor.fetchone()
                if row and row[0]:
                    image_descriptions[src] = row[0]
                    continue

                # Download image from BookStack
                internal_url = self.get_internal_url(src)
                logger.info(f"Downloading image for Page {page_id}: {internal_url}")
                try:
                    headers = auth_headers or {}
                    with httpx.Client(timeout=30.0, follow_redirects=True) as client:
                        img_res = client.get(internal_url, headers=headers)
                        if img_res.status_code != 200:
                            logger.warning(f"Failed to download image {internal_url}: HTTP {img_res.status_code}")
                            continue
                        img_bytes = img_res.content

                    # Check by content SHA-256 hash (deduplication)
                    img_hash = hashlib.sha256(img_bytes).hexdigest()
                    cursor.execute("SELECT visual_description FROM page_images WHERE image_hash = ?", (img_hash,))
                    hash_row = cursor.fetchone()
                    if hash_row and hash_row[0]:
                        logger.info(f"Image {src} matched existing hash {img_hash}. Reusing cached description (0 API cost).")
                        desc = hash_row[0]
                        cursor.execute("""
                            INSERT OR REPLACE INTO page_images (image_url, page_id, image_hash, alt_text, visual_description, updated_at)
                            VALUES (?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
                        """, (src, page_id, img_hash, alt, desc))
                        conn.commit()
                        image_descriptions[src] = desc
                        continue

                    # Analyze with Gemini 2.5 Flash
                    mime_type = self._detect_mime_type(img_bytes, src)
                    logger.info(f"Analyzing new image with Gemini 2.5 Flash: {src} ({mime_type}, {len(img_bytes)} bytes)")
                    desc = self.analyze_image_with_gemini(img_bytes, mime_type, alt_text=alt)

                    if desc:
                        cursor.execute("""
                            INSERT OR REPLACE INTO page_images (image_url, page_id, image_hash, alt_text, visual_description, updated_at)
                            VALUES (?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
                        """, (src, page_id, img_hash, alt, desc))
                        conn.commit()
                        image_descriptions[src] = desc
                        logger.info(f"Successfully cached vision analysis for {src}")

                except Exception as e:
                    logger.error(f"Error processing image {src} for page {page_id}: {e}")

        return image_descriptions
