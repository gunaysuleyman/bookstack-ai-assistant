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
    del history, current_page, planner, planner_enabled
    text = (query or "").strip()
    return QueryPlan(intent="search", primary_query=text, sub_questions=[text] if text else [])
