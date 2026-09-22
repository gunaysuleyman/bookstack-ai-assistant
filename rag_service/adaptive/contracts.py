from typing import Any, Callable, Dict, List, Literal, Optional

from pydantic import BaseModel, Field, field_validator


Intent = Literal["greeting", "overview", "search", "page_summary", "clarify"]
EvidenceStatus = Literal["supported", "partial", "missing", "conflicting"]


class AuthorizationScope(BaseModel):
    principal: str
    is_admin: bool
    can_use_ai: bool
    allowed_page_ids: Optional[List[int]] = None
    fingerprint: str
    acl_version: str
    issued_at: int
    expires_at: int
    roles: List[str] = Field(default_factory=list)

    def allows(self, page_id: Optional[int]) -> bool:
        if page_id is None:
            return False
        if not self.can_use_ai:
            return False
        if self.allowed_page_ids is None:
            return bool(self.is_admin)
        return int(page_id) in self.allowed_page_ids

    def allows_all(self) -> bool:
        return self.can_use_ai and self.is_admin and self.allowed_page_ids is None


class QueryPlan(BaseModel):
    intent: Intent
    sub_questions: List[str] = Field(default_factory=list, max_length=4)
    entities: List[str] = Field(default_factory=list, max_length=8)
    scope_hints: List[str] = Field(default_factory=list, max_length=8)
    required_evidence: List[str] = Field(default_factory=list, max_length=6)
    primary_query: str = ""
    search_query: str = ""
    topic: str = ""
    requested_slot: str = ""

    @field_validator("sub_questions", "entities", "scope_hints", "required_evidence")
    @classmethod
    def _trim_items(cls, value: List[str]) -> List[str]:
        cleaned = []
        for item in value:
            text = str(item).strip()[:200]
            if text:
                cleaned.append(text)
        return cleaned


class RetrievalCandidate(BaseModel):
    page_id: int
    chunk_id: str
    parent_id: str
    revision_id: str
    heading: str = ""
    text: str
    title: str = ""
    url: str = ""
    vector_rank: Optional[int] = None
    lexical_rank: Optional[int] = None
    fusion_score: float = 0.0
    rerank_score: Optional[float] = None
    channel: str = "hybrid"


class EvidenceAssessment(BaseModel):
    sub_question: str
    status: EvidenceStatus
    chunk_ids: List[str] = Field(default_factory=list)
    page_ids: List[int] = Field(default_factory=list)
    missing_topics: List[str] = Field(default_factory=list)
    continue_reason: str = ""
    conflict_values: List[str] = Field(default_factory=list)


class ContextPackage(BaseModel):
    selected: List[RetrievalCandidate] = Field(default_factory=list)
    estimated_tokens: int = 0
    actual_tokens: Optional[int] = None
    dropped: List[Dict[str, str]] = Field(default_factory=list)
    history_tokens: int = 0


class PageDocument(BaseModel):
    page_id: int
    name: str
    markdown: str
    book_id: int = 0
    book_name: str = "General Library"
    chapter_id: int = 0
    chapter_name: str = "General Chapter"
    shelf_names: List[str] = Field(default_factory=list)
    tags_str: str = ""
    url: str = ""
    updated_at: str = ""
    generation: int = 0

    def shelf_label(self) -> str:
        names = [name for name in self.shelf_names if name]
        return " | ".join(names) if names else "General Shelf"


class SyncJob(BaseModel):
    id: int
    page_id: int
    event: str
    generation: int
    status: str
    attempts: int = 0
    payload: Dict[str, Any] = Field(default_factory=dict)
    last_error: str = ""


ScopeResolver = Callable[[Optional[Dict[str, Any]]], AuthorizationScope]
