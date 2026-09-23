# Evidence contract and release gate

This contract applies to tool-based documentation answers. It is independent of
the language or subject matter of a book. An overall summary of the open page
uses the page-locked, bounded overview path. A specific question about that
page uses `search_current_page`, restricted to its active revision; the
overview sample must never serve as evidence that a particular instruction or
wording does not exist.

## One owner for answer evidence

1. Search retrieves candidates from the user's authorized, active pages. The
   first result of each document search, its adjacent chunks, and the end of
   its section get candidate slots before other hits. This is structural coverage, not a topic-specific
   rule or a promise that every long section has been read.
2. The evidence judge receives a bounded set of candidates. Its JSON is an
   *untrusted proposal*: whether the question has separate topics, the user's
   needs, unmet needs, and chunk IDs to use. Invalid JSON or invalid field
   types cannot authorize a broader search scope.
3. The server builds the **final evidence package** from those IDs. It checks
   each ID against the candidate set, current page revision, publication state,
   authorization scope, single-page lock, and the shared passage limit. It
   removes every unselected search hit from the answer model's tool results.
4. The final chunk IDs drive the answer model's passages, citation eligibility,
   and the `final` stage of `assistant_turns.evidence_json`. Earlier `search`
   stages are diagnostics, **not** answer evidence. The trace contains IDs and
   positions, not copied document text.
5. If judgment fails, selects no valid passage, or reports an unmet need, the
   service does not ask the answer model to complete a procedure. It returns an
   insufficient-evidence response. This favors a visible false negative over
   an invented instruction.

The judge may still be wrong about a need or choose a weak passage. The server
can validate identity, access, revision, and budgets; it cannot prove that a
passage semantically entails every sentence of a generated answer. Answer
quality therefore needs a real-document evaluation before wider rollout.

## Automated regression gate

Run `python -m pytest -q` from `rag_service`. Tests must include:

- a single procedure with multiple actions, without importing another page;
- a comparison that legitimately uses two pages under one passage budget;
- a long section whose decisive information is at the end;
- a selected chunk replacing an irrelevant initial hit, which must disappear
  from the answer model input and citation set;
- forged chunk IDs, malformed judgment types, revoked access and old revisions;
- unmet evidence, with no answer-model call to invent the missing step;
- current-page summary, catalog tools, greeting, and non-LLM fallback.

The synthetic test suite uses a local hash embedder and scripted model outputs.
Passing it proves the pipeline's constraints, **not** Gemini's retrieval or
answer accuracy. `rag_service/tests` is intentionally versioned; do not ignore
it again when adding new regression cases.

## Real-document acceptance gate

Before deploying this change broadly, make a versioned evaluation
set from real, permission-cleared documents. Record each question, expected
page/section or an explicit “insufficient evidence” label, allowed answer facts,
forbidden claims, and the user's language. Include the OmegaT trend correction
and save question, an HR contact question, an open-page summary, a two-page
comparison, a long-section ending, and deliberately unanswerable questions.
Keep holdout questions outside prompt and code development.

Run the set through the same endpoint and model configuration used in
production. Review both the answer and `evidence_json.final_chunk_ids`. A
release is blocked by any unauthorized/cross-page passage in a single task,
invented contact or procedure, unsupported complete answer for an unmet need,
or exceeded passage/model-call budget. Record answer correctness, evidence
recall, p50/p95 latency, and actual usage tokens/cost. Compare them with the
current production baseline and agree an acceptable quality/cost threshold
before switching traffic; unit-test counts alone are not that threshold.

Roll out to a small internal cohort first, keep the existing feature flag for
rollback, and inspect failures by final chunk IDs rather than adding
question-specific keywords to production code. New failures become evaluation
cases; change the pipeline only when they demonstrate a general invariant or
retrieval weakness.
