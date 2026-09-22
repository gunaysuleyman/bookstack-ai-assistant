"""Token estimates for budgeting.

all-MiniLM-L6-v2 truncates at 256 WordPiece tokens (chromadb ONNX wrapper).
This estimator is conservative and is not that WordPiece tokenizer.
Provider-reported usage is stored separately when the API returns it.
"""

import re


def estimate_tokens(text: str) -> int:
    if not text:
        return 0
    words = re.findall(r"\S+", text)
    by_words = int(len(words) * 1.4)
    by_chars = int(len(text) / 3)
    return max(1, by_words, by_chars)


def truncate_to_tokens(text: str, limit: int) -> str:
    if limit <= 0 or not text:
        return ""
    if estimate_tokens(text) <= limit:
        return text
    words = text.split()
    kept = []
    for word in words:
        trial = " ".join(kept + [word])
        if estimate_tokens(trial) > limit:
            break
        kept.append(word)
    if not kept:
        return text[: max(1, limit * 3)]
    return " ".join(kept)
