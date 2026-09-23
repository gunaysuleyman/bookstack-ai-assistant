"""Token estimates for budgeting.

all-MiniLM-L6-v2 truncates at 256 WordPiece tokens (chromadb ONNX wrapper).
This estimator is conservative and is not that WordPiece tokenizer.
Provider-reported usage is stored separately when the API returns it.
"""

import re
from typing import List


def estimate_tokens(text: str) -> int:
    if not text:
        return 0
    words = re.findall(r"\S+", text)
    by_words = int(len(words) * 1.4)
    by_chars = int(len(text) / 3)
    return max(1, by_words, by_chars)


def truncate_to_tokens(text: str, limit: int) -> str:
    return truncate_markdown(text, limit)


def truncate_markdown(text: str, limit: int) -> str:
    if limit <= 0 or not text:
        return ""
    if estimate_tokens(text) <= limit:
        return text
    lines = text.splitlines() or [text]
    kept = []
    for line in lines:
        trial = "\n".join(kept + [line])
        if estimate_tokens(trial) > limit:
            break
        kept.append(line)
    if kept:
        return "\n".join(kept).strip()
    return _clip_line(lines[0], limit)


def truncate_markdown_from_end(text: str, limit: int) -> str:
    if limit <= 0 or not text:
        return ""
    if estimate_tokens(text) <= limit:
        return text
    lines = text.splitlines() or [text]
    kept = []
    for line in reversed(lines):
        trial = "\n".join([line] + kept)
        if estimate_tokens(trial) > limit:
            break
        kept.insert(0, line)
    if kept:
        return "\n".join(kept).strip()
    return _clip_line(lines[-1], limit)


def cover_markdown(sections: List[str], budget: int) -> str:
    cleaned = [section.strip() for section in sections if section and section.strip()]
    if not cleaned or budget <= 0:
        return ""
    full = "\n\n".join(cleaned)
    if estimate_tokens(full) <= budget:
        return full
    # Allocate a bounded share to every section, not just the first and last.
    # For very long pages this is a sample, not a complete representation.
    if len(cleaned) == 1:
        head = truncate_markdown(cleaned[0], budget // 2)
        tail = truncate_markdown_from_end(cleaned[0], budget - estimate_tokens(head))
        if head and tail and head != tail:
            return f"{head}\n\n{tail}"
        return head or tail
    result = []
    remaining = budget
    for index, section in enumerate(cleaned):
        slots = len(cleaned) - index
        share = max(1, remaining // slots)
        excerpt = truncate_markdown_from_end(section, share) if index == len(cleaned) - 1 else truncate_markdown(section, share)
        if excerpt:
            result.append(excerpt)
            remaining -= estimate_tokens(excerpt)
    return "\n\n".join(result).strip()


def _clip_line(line: str, limit: int) -> str:
    if estimate_tokens(line) <= limit:
        return line
    words = line.split(" ")
    kept = []
    for word in words:
        trial = " ".join(kept + [word])
        if "](" in trial and trial.count("(") > trial.count(")"):
            break
        if estimate_tokens(trial) > limit:
            break
        kept.append(word)
    return " ".join(kept)
