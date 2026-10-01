"""Shelf, book, and chapter names: display cleanup and name resolution.

Names are resolved only against containers the caller can access, so an
unmatched or suggested name never reveals a hidden shelf, book, or chapter.
"""

import re
import unicodedata
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

PLACEHOLDER_BOOK = "General Library"
PLACEHOLDER_CHAPTER = "General Chapter"
PLACEHOLDER_SHELF = "General Shelf"
_PLACEHOLDERS = {PLACEHOLDER_BOOK, PLACEHOLDER_CHAPTER, PLACEHOLDER_SHELF}
_WORD = re.compile(r"\w+", re.UNICODE)
MAX_MATCHES = 50


def real_name(name: Optional[str]) -> str:
    """The name, or an empty string for a missing or placeholder container."""
    text = str(name or "").strip()
    return "" if text in _PLACEHOLDERS else text


def real_shelves(names: Optional[Iterable[str]]) -> List[str]:
    return [clean for clean in (real_name(name) for name in names or []) if clean]


def location_text(book_name: str = "", chapter_name: str = "", shelf_names: Optional[Iterable[str]] = None) -> str:
    """Indexed location words: shelves, book, and chapter. Placeholders are left out."""
    parts = real_shelves(shelf_names) + [real_name(book_name), real_name(chapter_name)]
    return "\n".join(part for part in parts if part)


def breadcrumb(book_name: str = "", chapter_name: str = "", title: str = "") -> str:
    """Book › Chapter › Page. Shelves are left out: a book can sit on many shelves and they change often."""
    parts = [real_name(book_name), real_name(chapter_name), str(title or "").strip()]
    return " › ".join(part for part in parts if part)


def normalize(text: str) -> str:
    """Case, diacritic, and dotted/dotless-i insensitive form for name matching."""
    folded = str(text or "").replace("İ", "i").replace("I", "ı").casefold().replace("ı", "i")
    decomposed = unicodedata.normalize("NFKD", folded)
    stripped = "".join(ch for ch in decomposed if not unicodedata.combining(ch))
    return " ".join(_WORD.findall(stripped))


def match_names(query: str, names: Sequence[str]) -> Tuple[List[str], List[str]]:
    """Return (matches, suggestions).

    Exact normalized matches win. Otherwise a name that contains the query, or
    is contained in it, matches. Otherwise a name matches when every query word
    is a prefix of one of its words or the other way round (at least three
    letters), so "finans ankara rafindaki" matches "Finans Ankara Rafı".
    A partial overlap is never a match: "Finans Ankara" must not silently
    become "Finans İzmir". The closest names are returned as suggestions.
    """
    wanted = normalize(query)
    if not wanted:
        return [], []
    normalized = [(name, normalize(name)) for name in dict.fromkeys(names) if name]
    exact = [name for name, norm in normalized if norm == wanted]
    if exact:
        return exact[:MAX_MATCHES], []
    contained = [
        name for name, norm in normalized
        if norm and (_contains_words(norm, wanted) or _contains_words(wanted, norm))
    ]
    if contained:
        return contained[:MAX_MATCHES], []
    query_words = [word for word in wanted.split() if len(word) >= 2]
    scored = []
    for name, norm in normalized:
        score = _word_overlap(query_words, norm.split())
        if score > 0:
            scored.append((score, name))
    if not scored:
        return [], []
    scored.sort(key=lambda item: (-item[0], item[1]))
    if scored[0][0] >= 1.0:
        return [name for score, name in scored if score >= 1.0][:MAX_MATCHES], []
    return [], [name for _score, name in scored[:5]]


def _contains_words(haystack: str, needle: str) -> bool:
    return f" {needle} " in f" {haystack} "


def _word_overlap(query_words: List[str], name_words: List[str]) -> float:
    if not query_words:
        return 0.0
    hits = 0
    for word in query_words:
        if any(_prefix_match(word, other) for other in name_words):
            hits += 1
    return hits / len(query_words)


def _prefix_match(left: str, right: str) -> bool:
    if len(left) < 3 or len(right) < 3:
        return left == right
    return left.startswith(right) or right.startswith(left)


@dataclass
class CatalogFilter:
    """Container restriction. None means "not restricted" for that level."""

    shelves: Optional[List[str]] = None
    book_ids: Optional[List[int]] = None
    chapter_ids: Optional[List[int]] = None

    def active(self) -> bool:
        return any(value is not None for value in (self.shelves, self.book_ids, self.chapter_ids))


@dataclass
class ResolvedFilter:
    filter: CatalogFilter = field(default_factory=CatalogFilter)
    matched: Dict[str, List[str]] = field(default_factory=dict)
    unmatched: Dict[str, str] = field(default_factory=dict)
    suggestions: Dict[str, List[str]] = field(default_factory=dict)

    def requested(self) -> bool:
        return bool(self.matched or self.unmatched)

    def report(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {}
        if self.matched:
            out["matched"] = self.matched
        if self.unmatched:
            out["unmatched"] = self.unmatched
        if self.suggestions:
            out["suggestions"] = self.suggestions
        return out


def resolve_filter(store, allowed: Optional[Sequence[int]], shelf: str = "", book: str = "", chapter: str = "") -> ResolvedFilter:
    """Resolve shelf/book/chapter names against the containers `allowed` can see."""
    resolved = ResolvedFilter()
    shelf = (shelf or "").strip()
    book = (book or "").strip()
    chapter = (chapter or "").strip()
    if shelf:
        matches, suggestions = match_names(shelf, store.catalog_shelf_names(allowed))
        if matches:
            resolved.filter.shelves = matches
            resolved.matched["shelf"] = matches
        else:
            resolved.unmatched["shelf"] = shelf
            if suggestions:
                resolved.suggestions["shelf"] = suggestions
    if book:
        books = store.catalog_book_names(allowed)
        matches, suggestions = match_names(book, [name for _book_id, name in books])
        if matches:
            wanted = set(matches)
            resolved.filter.book_ids = sorted({book_id for book_id, name in books if name in wanted})
            resolved.matched["book"] = matches
        else:
            resolved.unmatched["book"] = book
            if suggestions:
                resolved.suggestions["book"] = suggestions
    if chapter:
        chapters = store.catalog_chapter_names(allowed, resolved.filter.book_ids)
        matches, suggestions = match_names(chapter, [name for _chapter_id, name, _book in chapters])
        if matches:
            wanted = set(matches)
            resolved.filter.chapter_ids = sorted({chapter_id for chapter_id, name, _book in chapters if name in wanted})
            resolved.matched["chapter"] = matches
        else:
            resolved.unmatched["chapter"] = chapter
            if suggestions:
                resolved.suggestions["chapter"] = suggestions
    return resolved
