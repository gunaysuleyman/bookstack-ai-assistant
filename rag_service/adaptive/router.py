import re
from typing import Callable, List, Optional

from adaptive.contracts import QueryPlan


def fold(text: str) -> str:
    return text.replace("İ", "i").replace("I", "ı").casefold()


def route_query(
    query: str,
    history: Optional[List[dict]] = None,
    current_page: Optional[dict] = None,
    planner: Optional[Callable[[str], str]] = None,
    planner_enabled: bool = False,
) -> QueryPlan:
    del history, planner, planner_enabled
    text = (query or "").strip()
    page_reference = re.search(
        r"\b(this|current|open|active)\s+(page|article)\b|\b(bu|açık)\s+(sayfa|makale)\w*\b|\b(questa|questo)\s+(pagina|articolo)\b",
        text,
        re.I,
    )
    unrelated_pages = re.search(r"\b(other|related|different)\s+pages\b|\b(diğer|benzer)\s+sayfalar\b", text, re.I)
    intent = "page_summary" if current_page and page_reference and not unrelated_pages else "search"
    return QueryPlan(intent=intent, primary_query=text, sub_questions=[text] if text else [])
