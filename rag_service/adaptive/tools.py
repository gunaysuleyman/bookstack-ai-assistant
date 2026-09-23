import ast
import operator
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, ValidationError

from adaptive.contracts import AuthorizationScope, RetrievalCandidate
from adaptive.hybrid import HybridSearcher, dedupe_candidates
from adaptive.store import StateStore
from adaptive.tokenizer import cover_markdown, estimate_tokens, truncate_markdown


TOOL_NAMES = (
    "summarize_current_page",
    "document_search",
    "catalog_counts",
    "catalog_list_books",
    "calculator",
)
MAX_TOOL_CALLS = 2
MAX_EXPRESSION_CHARS = 120
MAX_EXPRESSION_NODES = 40
MAX_EXPRESSION_DEPTH = 8
MAX_ABS_VALUE = 1_000_000_000
ALLOWED_OPERATORS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
    ast.Pow: operator.pow,
}


class DocumentSearchArgs(BaseModel):
    query: str = Field(min_length=1, max_length=2000)
    limit: int = Field(default=8, ge=1, le=20)


class CatalogListBooksArgs(BaseModel):
    offset: int = Field(default=0, ge=0, le=100000)
    limit: int = Field(default=20, ge=1, le=50)


class CalculatorArgs(BaseModel):
    expression: str = Field(min_length=1, max_length=MAX_EXPRESSION_CHARS)


FUNCTION_DECLARATIONS = [
    {
        "name": "summarize_current_page",
        "description": (
            "Read the page the user currently has open. The server locks this to that page id. "
            "Use this when the user asks to summarize, explain, or describe the open page. "
            "Do not pass a page id."
        ),
        "parameters": {"type": "object", "properties": {}},
    },
    {
        "name": "document_search",
        "description": (
            "Search indexed documentation the user can access. Use for procedures, contacts, policies, "
            "and content on pages other than the one currently open. Do not use this to summarize the open page."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Search query in the user's language."},
                "limit": {"type": "integer", "description": "Maximum passages to return.", "minimum": 1, "maximum": 20},
            },
            "required": ["query"],
        },
    },
    {
        "name": "catalog_counts",
        "description": "Return exact counts of pages, books, and shelves the user can access.",
        "parameters": {"type": "object", "properties": {}},
    },
    {
        "name": "catalog_list_books",
        "description": "List books the user can access with visible page counts.",
        "parameters": {
            "type": "object",
            "properties": {
                "offset": {"type": "integer", "minimum": 0},
                "limit": {"type": "integer", "minimum": 1, "maximum": 50},
            },
        },
    },
    {
        "name": "calculator",
        "description": "Evaluate a basic arithmetic expression with numbers, parentheses, and + - * / % **.",
        "parameters": {
            "type": "object",
            "properties": {"expression": {"type": "string"}},
            "required": ["expression"],
        },
    },
]


def gemini_tools() -> List[Dict[str, Any]]:
    return [{"functionDeclarations": FUNCTION_DECLARATIONS}]


def allowed_page_ids(scope: AuthorizationScope) -> Optional[List[int]]:
    if not scope.can_use_ai:
        return []
    if scope.allows_all():
        return None
    return list(scope.allowed_page_ids or [])


def serialize_passages(selected: List[RetrievalCandidate], max_tokens: int = 220) -> List[Dict[str, Any]]:
    rows = []
    for item in selected:
        text = truncate_markdown(item.text or "", max_tokens)
        rows.append(
            {
                "page_id": item.page_id,
                "title": item.title,
                "url": item.url,
                "heading": item.heading,
                "text": text,
            }
        )
    return rows


class ToolRegistry:
    def __init__(self, store: StateStore, searcher: HybridSearcher, settings):
        self.store = store
        self.searcher = searcher
        self.settings = settings

    def execute(
        self,
        name: str,
        args: Dict[str, Any],
        scope: AuthorizationScope,
        current_page: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        if name not in TOOL_NAMES:
            return {"ok": False, "error": "unknown_tool"}
        if not scope.can_use_ai:
            return {"ok": False, "error": "forbidden"}
        allowed = allowed_page_ids(scope)
        try:
            if name == "summarize_current_page":
                return self._summarize_current_page(scope, current_page)
            if name == "document_search":
                return self._document_search(DocumentSearchArgs.model_validate(args or {}), scope)
            if name == "catalog_counts":
                return {"ok": True, **self.store.catalog_counts(allowed)}
            if name == "catalog_list_books":
                parsed = CatalogListBooksArgs.model_validate(args or {})
                books = self.store.catalog_books(allowed, parsed.offset, parsed.limit)
                return {"ok": True, "offset": parsed.offset, "limit": parsed.limit, "books": books}
            parsed = CalculatorArgs.model_validate(args or {})
            return {"ok": True, "expression": parsed.expression, "result": safe_calculate(parsed.expression)}
        except ValidationError:
            return {"ok": False, "error": "invalid_arguments"}
        except ValueError as exc:
            return {"ok": False, "error": str(exc)}

    def _summarize_current_page(self, scope: AuthorizationScope, current_page: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        page_id = _page_id(current_page)
        if page_id is None:
            return {"ok": False, "error": "no_current_page"}
        if not scope.allows(page_id):
            return {"ok": False, "error": "forbidden"}
        state = self.store.get_page_state(page_id)
        if state is None or state["status"] != "published" or not state["active_revision"]:
            return {"ok": False, "error": "not_found"}
        parents = self.store.page_parents(page_id, state["active_revision"])
        seen = set()
        sections = []
        for parent in parents:
            if parent["parent_id"] in seen:
                continue
            seen.add(parent["parent_id"])
            sections.append(str(parent["body"] or ""))
        budget = max(1, int(self.settings.context_token_budget))
        text = cover_markdown(sections, budget)
        return {
            "ok": True,
            "page_id": page_id,
            "title": str(state["title"] or ""),
            "url": str(state["url"] or ""),
            "revision_id": str(state["active_revision"]),
            "text": text,
            "partial": estimate_tokens("\n\n".join(sections)) > budget,
            "section_count": len(sections),
        }

    def _document_search(self, parsed: DocumentSearchArgs, scope: AuthorizationScope) -> Dict[str, Any]:
        cap = max(1, int(self.settings.max_tool_result_passages))
        limit = min(parsed.limit, cap, self.settings.max_candidates)
        found = self.searcher.search(parsed.query, scope, limit)
        selected = [item for item in dedupe_candidates(found) if scope.allows(item.page_id)][:limit]
        return {"ok": True, "passages": serialize_passages(selected)}


def _page_id(current_page: Optional[Dict[str, Any]]) -> Optional[int]:
    if not current_page or current_page.get("page_id") is None:
        return None
    try:
        return int(current_page["page_id"])
    except (TypeError, ValueError):
        return None


def safe_calculate(expression: str) -> float:
    text = expression.strip()
    if not text or len(text) > MAX_EXPRESSION_CHARS:
        raise ValueError("invalid_expression")
    try:
        tree = ast.parse(text, mode="eval")
    except SyntaxError as exc:
        raise ValueError("invalid_expression") from exc
    if sum(1 for _ in ast.walk(tree)) > MAX_EXPRESSION_NODES:
        raise ValueError("invalid_expression")
    value = _eval_node(tree.body, 0)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("invalid_expression")
    if abs(value) > MAX_ABS_VALUE:
        raise ValueError("overflow")
    return float(value)


def _eval_node(node: ast.AST, depth: int) -> float:
    if depth > MAX_EXPRESSION_DEPTH:
        raise ValueError("invalid_expression")
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)) and not isinstance(node.value, bool):
        return float(node.value)
    if isinstance(node, ast.UnaryOp) and type(node.op) in ALLOWED_OPERATORS:
        return float(ALLOWED_OPERATORS[type(node.op)](_eval_node(node.operand, depth + 1)))
    if isinstance(node, ast.BinOp) and type(node.op) in ALLOWED_OPERATORS:
        left = _eval_node(node.left, depth + 1)
        right = _eval_node(node.right, depth + 1)
        if isinstance(node.op, (ast.Div, ast.FloorDiv, ast.Mod)) and right == 0:
            raise ValueError("division_by_zero")
        if isinstance(node.op, ast.Pow) and (abs(left) > 1000 or abs(right) > 8):
            raise ValueError("overflow")
        return float(ALLOWED_OPERATORS[type(node.op)](left, right))
    if isinstance(node, ast.Expression):
        return _eval_node(node.body, depth + 1)
    raise ValueError("invalid_expression")
