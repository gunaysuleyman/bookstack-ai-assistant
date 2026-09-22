import re
from dataclasses import dataclass
from typing import List

from adaptive.contracts import PageDocument
from adaptive.tokenizer import estimate_tokens, truncate_to_tokens


_HEADING = re.compile(r"^(#{1,6})\s+(.*)$")


@dataclass
class ParentSection:
    parent_id: str
    page_id: int
    heading: str
    text: str
    ordinal: int

    @property
    def token_estimate(self) -> int:
        return estimate_tokens(self.text)


@dataclass
class ChildChunk:
    chunk_id: str
    parent_id: str
    page_id: int
    heading: str
    text: str
    embed_text: str
    ordinal: int

    @property
    def token_estimate(self) -> int:
        return estimate_tokens(self.embed_text)


def chunk_document(page: PageDocument, child_tokens: int = 160, embed_max_tokens: int = 180) -> tuple:
    sections = _sections(page.markdown)
    parents: List[ParentSection] = []
    children: List[ChildChunk] = []
    for index, (heading, body) in enumerate(sections):
        parent_id = f"p{page.page_id}s{index}"
        parent_text = f"# {heading}\n\n{body}".strip()
        parents.append(ParentSection(parent_id, page.page_id, heading, parent_text, index))
        pieces = _pieces(body, child_tokens)
        for offset, piece in enumerate(pieces):
            embed = _embed_text(heading, piece, embed_max_tokens)
            children.append(
                ChildChunk(
                    chunk_id=f"p{page.page_id}s{index}c{offset}",
                    parent_id=parent_id,
                    page_id=page.page_id,
                    heading=heading,
                    text=piece,
                    embed_text=embed,
                    ordinal=len(children),
                )
            )
    return parents, children


def _sections(markdown: str) -> List[tuple]:
    if not markdown.strip():
        return []
    heading = "Introduction"
    lines: List[str] = []
    sections = []
    for line in markdown.splitlines():
        match = _HEADING.match(line)
        if match:
            body = "\n".join(lines).strip()
            if body:
                sections.append((heading, body))
            heading = match.group(2).strip() or heading
            lines = []
        else:
            lines.append(line)
    body = "\n".join(lines).strip()
    if body:
        sections.append((heading, body))
    return sections or [("Introduction", markdown.strip())]


def _pieces(body: str, child_tokens: int) -> List[str]:
    blocks = _blocks(body)
    pieces: List[str] = []
    buffer = ""
    for block in blocks:
        if estimate_tokens(block) > child_tokens:
            if buffer.strip():
                pieces.append(buffer.strip())
                buffer = ""
            pieces.extend(_split_oversized(block, child_tokens))
            continue
        trial = f"{buffer}\n\n{block}".strip() if buffer else block
        if estimate_tokens(trial) <= child_tokens:
            buffer = trial
        else:
            if buffer.strip():
                pieces.append(buffer.strip())
            buffer = block
    if buffer.strip():
        pieces.append(buffer.strip())
    return pieces or [body.strip()]


def _blocks(body: str) -> List[str]:
    lines = body.splitlines()
    blocks: List[str] = []
    index = 0
    while index < len(lines):
        line = lines[index]
        if line.startswith("```"):
            fence = [line]
            index += 1
            while index < len(lines) and not lines[index].startswith("```"):
                fence.append(lines[index])
                index += 1
            if index < len(lines):
                fence.append(lines[index])
                index += 1
            blocks.append("\n".join(fence))
            continue
        if "|" in line and index + 1 < len(lines) and re.match(r"^\s*\|?\s*-+", lines[index + 1]):
            table = [line]
            index += 1
            while index < len(lines) and "|" in lines[index]:
                table.append(lines[index])
                index += 1
            blocks.append("\n".join(table))
            continue
        if re.match(r"^\s*([-*+]|\d+\.)\s+", line):
            items = [line]
            index += 1
            while index < len(lines) and (re.match(r"^\s*([-*+]|\d+\.)\s+", lines[index]) or lines[index].startswith("  ")):
                items.append(lines[index])
                index += 1
            blocks.append("\n".join(items))
            continue
        paragraph = [line]
        index += 1
        while index < len(lines) and lines[index].strip() and not lines[index].startswith("```"):
            if re.match(r"^\s*([-*+]|\d+\.)\s+", lines[index]):
                break
            paragraph.append(lines[index])
            index += 1
        text = "\n".join(paragraph).strip()
        if text:
            blocks.append(text)
        while index < len(lines) and not lines[index].strip():
            index += 1
    return [block for block in blocks if block.strip()]


def _split_oversized(block: str, child_tokens: int) -> List[str]:
    if block.startswith("|") or "\n|" in block:
        return _split_table(block, child_tokens)
    words = block.split()
    pieces = []
    current: List[str] = []
    for word in words:
        trial = " ".join(current + [word])
        if current and estimate_tokens(trial) > child_tokens:
            pieces.append(" ".join(current))
            current = [word]
        else:
            current.append(word)
    if current:
        pieces.append(" ".join(current))
    return pieces


def _split_table(block: str, child_tokens: int) -> List[str]:
    rows = [row for row in block.splitlines() if row.strip()]
    if len(rows) <= 2:
        return _split_oversized_plain(block, child_tokens)
    header = rows[:2]
    pieces = []
    current = header[:]
    for row in rows[2:]:
        trial = "\n".join(current + [row])
        if estimate_tokens(trial) > child_tokens and len(current) > 2:
            pieces.append("\n".join(current))
            current = header[:] + [row]
        else:
            current.append(row)
    if len(current) > 2:
        pieces.append("\n".join(current))
    return pieces


def _split_oversized_plain(block: str, child_tokens: int) -> List[str]:
    words = block.split()
    pieces = []
    current: List[str] = []
    for word in words:
        trial = " ".join(current + [word])
        if current and estimate_tokens(trial) > child_tokens:
            pieces.append(" ".join(current))
            current = [word]
        else:
            current.append(word)
    if current:
        pieces.append(" ".join(current))
    return pieces


def _embed_text(heading: str, piece: str, embed_max_tokens: int) -> str:
    heading_budget = min(24, max(0, embed_max_tokens // 5))
    short_heading = truncate_to_tokens(heading, heading_budget) if heading else ""
    if short_heading and estimate_tokens(short_heading) < embed_max_tokens / 2:
        body_budget = max(1, embed_max_tokens - estimate_tokens(short_heading) - 1)
        body = truncate_to_tokens(piece, body_budget)
        return f"{short_heading}\n{body}".strip()
    return truncate_to_tokens(piece, embed_max_tokens)
