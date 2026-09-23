from typing import Dict, List, Sequence

from adaptive.contracts import ContextPackage, EvidenceAssessment, RetrievalCandidate
from adaptive.tokenizer import estimate_tokens, truncate_to_tokens


def pack_context(
    assessments: Sequence[EvidenceAssessment],
    candidates: Sequence[RetrievalCandidate],
    budget_tokens: int,
    parent_lookup,
    parent_token_limit: int,
) -> ContextPackage:
    by_id = {candidate.chunk_id: candidate for candidate in candidates}
    groups: Dict[str, List[RetrievalCandidate]] = {}
    for assessment in assessments:
        if assessment.status == "missing":
            continue
        chosen = []
        for chunk_id in assessment.chunk_ids:
            candidate = by_id.get(chunk_id)
            if candidate is not None:
                chosen.append(candidate)
        if chosen:
            groups[assessment.sub_question] = chosen
    share_count = max(1, len(groups))
    share = max(1, budget_tokens // share_count)
    selected: List[RetrievalCandidate] = []
    dropped: List[dict] = []
    used = 0
    seen = set()
    seen_parents = set()
    for _question, group in groups.items():
        share_used = 0
        for candidate in group:
            parent_key = (candidate.parent_id, candidate.revision_id)
            if candidate.chunk_id in seen or (candidate.parent_id and parent_key in seen_parents):
                dropped.append({"chunk_id": candidate.chunk_id, "reason": "duplicate"})
                continue
            text = _with_parent(candidate, parent_lookup, parent_token_limit)
            tokens = estimate_tokens(text)
            if share_used + tokens > share or used + tokens > budget_tokens:
                remaining = min(share - share_used, budget_tokens - used)
                if remaining > 0 and candidate.chunk_id not in seen:
                    excerpt = truncate_to_tokens(text, remaining)
                    if excerpt:
                        selected.append(candidate.model_copy(update={"text": excerpt}))
                        seen.add(candidate.chunk_id)
                        if candidate.parent_id:
                            seen_parents.add(parent_key)
                        used += estimate_tokens(excerpt)
                        share_used += estimate_tokens(excerpt)
                dropped.append({"chunk_id": candidate.chunk_id, "reason": "budget"})
                continue
            packed = candidate.model_copy(update={"text": text})
            selected.append(packed)
            seen.add(candidate.chunk_id)
            if candidate.parent_id:
                seen_parents.add(parent_key)
            share_used += tokens
            used += tokens
    return ContextPackage(selected=selected, estimated_tokens=used, dropped=dropped)


def _with_parent(candidate: RetrievalCandidate, parent_lookup, parent_token_limit: int) -> str:
    parent = parent_lookup(candidate.parent_id, candidate.revision_id) if parent_lookup else None
    if parent is None:
        return candidate.text
    if int(parent["page_id"]) != candidate.page_id:
        return candidate.text
    heading = str(parent["heading"] or candidate.heading)
    parent_body = str(parent["body"] or "")
    if estimate_tokens(parent_body) <= parent_token_limit and candidate.text in parent_body:
        return parent_body
    excerpt = truncate_to_tokens(parent_body, parent_token_limit)
    if candidate.text in excerpt:
        return excerpt
    return f"# {heading}\n\n{candidate.text}".strip()
