import re
import html2text
from typing import List, Dict, Any

class HTMLCleaner:
    def __init__(self):
        self.h2t = html2text.HTML2Text()
        self.h2t.ignore_links = False
        self.h2t.ignore_images = True
        self.h2t.ignore_tables = False
        self.h2t.body_width = 0

    def clean_to_markdown(self, html_content: str) -> str:
        """Converts BookStack HTML string to clean Markdown."""
        if not html_content:
            return ""
        markdown = self.h2t.handle(html_content)
        markdown = re.sub(r'\n{3,}', '\n\n', markdown).strip()
        return markdown

    def chunk_markdown(self, text: str, page_meta: Dict[str, Any], target_size: int = 800, max_size: int = 1200) -> List[Dict[str, Any]]:
        """
        Full 4-Tier BookStack Hierarchy aware Markdown chunking.
        Injects [Raf: S] > [Kitap: B] > [Bölüm: C] > [Sayfa: P] [Etiketler: T]
        into every chunk text so ChromaDB vector embeddings carry 100% full hierarchy context.
        """
        if not text:
            return []

        shelf_name = page_meta.get("shelf_name", "Genel Raf")
        book_name = page_meta.get("book_name", "Genel Kütüphane")
        chapter_name = page_meta.get("chapter_name", "Genel Bölüm")
        page_title = page_meta.get("name", "İsimsiz Doküman")
        page_id = page_meta.get("id")
        tags_str = page_meta.get("tags_str", "")

        # Split text into section blocks by headings (#, ##, ###)
        lines = text.splitlines()
        sections = []
        current_heading = "Giriş"
        current_lines = []

        heading_re = re.compile(r'^(#{1,6})\s+(.*)$')

        for line in lines:
            match = heading_re.match(line)
            if match:
                if current_lines:
                    sections.append((current_heading, "\n".join(current_lines).strip()))
                    current_lines = []
                current_heading = match.group(2).strip()
            else:
                current_lines.append(line)

        if current_lines:
            sections.append((current_heading, "\n".join(current_lines).strip()))

        chunks = []
        chunk_idx = 0

        tags_header = f" [Etiketler: {tags_str}]" if tags_str else ""

        for heading, section_text in sections:
            if not section_text:
                continue

            header_context = f"[Raf: {shelf_name}] > [Kitap: {book_name}] > [Bölüm: {chapter_name}] > [Sayfa: {page_title}]{tags_header}\n[Alt Başlık: {heading}]"

            # If section is small enough, make it a chunk
            if len(section_text) <= max_size:
                full_chunk_text = f"{header_context}\n\n{section_text}"
                chunks.append({
                    "id": f"page_{page_id}_chunk_{chunk_idx}",
                    "text": full_chunk_text,
                    "metadata": {
                        **page_meta,
                        "heading": heading,
                        "chunk_index": chunk_idx
                    }
                })
                chunk_idx += 1
            else:
                # Split large section by paragraphs
                paragraphs = section_text.split("\n\n")
                buffer = ""
                for para in paragraphs:
                    if len(buffer) + len(para) <= target_size:
                        buffer += ("\n\n" + para if buffer else para)
                    else:
                        if buffer.strip():
                            full_chunk_text = f"{header_context}\n\n{buffer.strip()}"
                            chunks.append({
                                "id": f"page_{page_id}_chunk_{chunk_idx}",
                                "text": full_chunk_text,
                                "metadata": {
                                    **page_meta,
                                    "heading": heading,
                                    "chunk_index": chunk_idx
                                }
                            })
                            chunk_idx += 1
                        buffer = para
                if buffer.strip():
                    full_chunk_text = f"{header_context}\n\n{buffer.strip()}"
                    chunks.append({
                        "id": f"page_{page_id}_chunk_{chunk_idx}",
                        "text": full_chunk_text,
                        "metadata": {
                            **page_meta,
                            "heading": heading,
                            "chunk_index": chunk_idx
                        }
                    })
                    chunk_idx += 1

        return chunks
