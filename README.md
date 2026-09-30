# BookStack AI Assistant

A self-hosted, permission-aware documentation assistant for [BookStack](https://www.bookstackapp.com/). It runs a FastAPI RAG service beside BookStack and its MariaDB database, and adds a chat widget to BookStack pages. The assistant searches only pages the current user may access, then builds answers from selected evidence and links back to the source articles.

The repository contains both the original retrieval path and a newer Adaptive RAG path. The feature flag defaults to the original path; installing the project does **not** automatically switch to the new index or run a full synchronization.

## What it does

- In Adaptive mode, answers documentation questions using hybrid vector and SQLite FTS5 search, with active-revision and page-permission checks.
- Understands requests about the open page, including a bounded page overview, a page-locked search for specific details, and summaries of image descriptions already indexed with the page.
- Provides tools for document search, accessible catalog counts and book lists, and arithmetic. A model chooses the tool from the meaning of the request, including imperfect spelling; the server validates the tool, scope, and arguments.
- Checks retrieved evidence before a document answer. It can make one targeted follow-up search, distinguish complete from partial evidence, or ask for clarification instead of filling a gap with an invented fact.
- Carries bounded conversation history through tool selection, evidence assessment, and answer generation. History helps interpret follow-ups such as “yes”; it is **not** evidence for a factual claim.
- Indexes BookStack page changes through a durable queue. Optional Gemini vision descriptions are created **at index time**, not every time a user asks about an image.
- Records model usage and evidence metadata for diagnostics without copying passage bodies into the evidence trace.

These controls reduce unsupported answers; they cannot prove that every model judgment or generated sentence is correct. Evaluate answers on your own documents before a broad rollout.

## Components and data flow

1. BookStack renders the widget and calculates the signed-in user's visible page IDs using BookStack's own permission model. The browser receives a short-lived scope reference, or a signed fallback payload, never the HMAC secret.
2. The widget sends a question, the open-page context, and recent chat history to the FastAPI service.
3. In Adaptive RAG tool mode, the model chooses a task: search the open page or accessible library, summarize the open page, inspect the accessible catalog, calculate, greet, or clarify. A model cannot supply an arbitrary open-page ID.
4. Document search combines Chroma embeddings and FTS5 lexical results. The evidence assessment proposes relevant chunk IDs; the server checks authorization, active revision, page scope, and the shared passage budget before the answer model sees them.
5. The response cites the BookStack articles represented in the final evidence package. Search hits rejected before the final package are diagnostic data, not answer evidence.

The service uses SQLite (`rag_state.sqlite`) for the job queue, page/revision state, FTS5, catalog, usage, and answer traces; Chroma stores vectors. It runs Chroma in **one process**. Do not start multiple RAG workers against the same local Chroma directory.

## Run locally with Docker Compose

Requirements: Docker with Compose, a BookStack API token for synchronization, and credentials for the chosen model provider. The default Adaptive embedding model is Gemini, so an Adaptive setup using that default also needs `GEMINI_API_KEY` even if OpenAI writes the answers.

1. Copy the template and set real secrets in `.env`:

   ```bash
   cp .env.example .env
   ```

   Set a new BookStack-compatible `APP_KEY`, a strong `DB_PASS`, unique `RAG_SECRET_TOKEN` and `WEBHOOK_SECRET` values, and the relevant API keys. Keep `.env` out of Git. `RAG_SECRET_TOKEN` is shared by the BookStack and RAG containers but must never be put in browser JavaScript. For a production host, set `BOOKSTACK_EXTERNAL_URL` and the browser-reachable `RAG_SERVICE_PUBLIC_URL` to the real HTTPS addresses.

2. Start the three services:

   ```bash
   docker compose up -d --build
   ```

   BookStack is exposed on port `6875`. By default the RAG API is bound to `127.0.0.1:8000`; expose it to remote browsers through a configured reverse proxy, not by casually changing the bind address to `0.0.0.0`.

3. Create a BookStack API token in BookStack's user settings. Set `BOOKSTACK_TOKEN_ID` and `BOOKSTACK_TOKEN_SECRET` in `.env`, then **recreate** the RAG container so it receives the new environment:

   ```bash
   docker compose up -d --no-deps --force-recreate rag_service
   ```

   `docker compose restart` does not reload changed `.env` values.

4. Check `GET http://127.0.0.1:8000/ready`. It reports the active collection and readiness checks. HTTP 200 means the configured service components are available; it does **not** prove that all BookStack pages have been indexed.

5. Queue the first full reconciliation. It returns a job ID immediately; indexing continues in the worker:

   ```bash
   curl -X POST http://127.0.0.1:8000/api/sync \
     -H "X-RAG-Token: <RAG_SECRET_TOKEN>"

   curl http://127.0.0.1:8000/api/jobs/status \
     -H "X-RAG-Token: <RAG_SECRET_TOKEN>"
   ```

   On Windows PowerShell, use `curl.exe` or `Invoke-RestMethod` instead of the `curl` alias if needed. Do not run a second writer process against the same Chroma volume.

6. In BookStack, configure a webhook for page create/update/move/delete, and for book, chapter, and shelf create/update/move/delete so renamed or moved containers are re-labelled in the index. The container-to-container URL is `http://rag_service:8000/api/webhook?token=<WEBHOOK_SECRET>`. BookStack does not sign these webhook bodies; the secret in the URL must match `.env`. Do not reuse the service HMAC secret as the webhook secret.

The index schema `pc-v3` adds page titles and shelf/book/chapter names to search. After upgrading from an earlier schema, run one full reconciliation (`POST /api/sync`); pages with an older schema are re-chunked and re-embedded. Until then, lexical search already uses the new title and location columns.

No startup full sync runs automatically. If Adaptive indexing is enabled, `WEBHOOK_SECRET` must be nonempty or the service refuses to start. The service also refuses a missing or known example `RAG_SECRET_TOKEN`.

## Choosing the RAG mode

- `ADAPTIVE_RAG=0` or `off` (template default): user answers use the legacy `bookstack_articles` collection.
- `ADAPTIVE_RAG=shadow`: users still receive legacy answers; Adaptive retrieval runs alongside it for comparison, without a second answer-model call.
- `ADAPTIVE_RAG=on`: answers use the separate Adaptive collection, default `bookstack_articles_pc_v1`. Populate and evaluate this collection before switching traffic.

Set `ADAPTIVE_INDEXING=1` when the new index should receive queued page changes and full reconciliation. `ENABLE_INDEX_WORKER=1` processes jobs in the service process. To roll back answer traffic, set `ADAPTIVE_RAG=off` and recreate the RAG container; keep the old collection intact. See [operations](docs/rag_operations.md) for the queue, backups, recovery, and rollback procedure.

Changing `EMBEDDING_MODEL_ID` is an index migration, not a harmless configuration edit. A collection built with another embedding model is preserved and startup reports a mismatch. Choose a **new versioned `ADAPTIVE_COLLECTION`**, reconcile it fully, evaluate it, then switch traffic. The default is `gemini-embedding-001`; `all-MiniLM-L6-v2` is an optional local embedder. The test-only `hash-bow` embedder does not measure semantic retrieval quality.

## Models, images, and cost boundaries

`AI_PROVIDER=gemini` is the template default for answers. `AI_PROVIDER=openai` is also supported via `OPENAI_API_KEY` and `OPENAI_MODEL`. Embeddings are configured separately with `EMBEDDING_MODEL_ID`; the default Gemini embedding model may incur indexing and query costs regardless of which provider answers. Gemini vision, when configured, analyzes permitted images during indexing and caches descriptions by image content, model, and prompt version. Ordinary questions about those images use the stored descriptions; they do not trigger a new vision call.

Images on BookStack's own host may be fetched with BookStack credentials. External image hosts are skipped by default; list only approved hosts in `IMAGE_TRUSTED_HOSTS`. BookStack authorization headers are not forwarded to those external hosts. Avoid enabling unrestricted `IMAGE_ALLOW_EXTERNAL` in production.

`MAX_MODEL_CALLS` (default `5`), `MAX_TOOL_RESULT_PASSAGES` (default `8`), `CONTEXT_TOKEN_BUDGET` (default `3000`), and `HISTORY_TOKEN_BUDGET` (default `800`) bound normal requests. These are budgets for calls and **evidence sent to the answer model**, not a fixed `TOP_K` answer strategy. The model may issue a targeted follow-up search only when needed. The widget sends up to eight recent messages; the API accepts up to twelve, and the service includes the newest messages that fit the history token budget. A long message retains its beginning and end within its per-message cap.

## Access control and diagnostics

`AI_ALLOWED_ROLES` gates use of the assistant, but a role alone does not grant access to every page. BookStack's `Page::visible()` determines the per-user page scope, including inherited permissions. A missing token is not treated as an administrator. Scope data expires after `TOKEN_TTL_SECONDS` (default `900`); permission revocation is therefore not instantaneous during an existing session.

`GET /health` is a liveness check. `GET /ready` reports active mode, collection, embedding model, worker, and basic readiness. The service-protected `GET /api/jobs/status` reports queue state. `assistant_turns.evidence_json` keeps search, candidate, and final chunk identifiers plus routing/coverage decisions; `usage_log` stores reported token usage. These traces help diagnose why an answer was partial or refused, but they do not contain a second copy of passage text. Protect and back up the state database because it still contains questions and answers.

## Development and evaluation

Install the Python requirements and run the isolated suite from `rag_service`:

```bash
cd rag_service
python -m pip install -r requirements.txt -r requirements-dev.txt
python -m pytest -q
```

Tests use scripted models and temporary indexes; passing them does not certify live-model answer quality. A synthetic capacity run can use local MiniLM without calling Gemini:

```bash
python benchmarks/capacity_smoke.py --pages 10000 --sections 12 --words-per-section 80 --queries 100 --embedder minilm
```

That run measures a synthetic corpus and sequential searches, not real BookStack content, Gemini embedding cost, or concurrent production load. For rollout, evaluate a permission-cleared real-document set, check the final evidence IDs as well as answer text, and measure latency and actual provider usage. The [evaluation notes](docs/rag_evaluation.md) and [PISA Final Check cases](docs/pisa_final_check_eval.md) provide historical baselines, not current live-model certification.

## Repository map

- `widget/bookstack_ai_widget.html` — BookStack theme widget, open-page context, and client-side conversation history.
- `rag_service/main.py` — FastAPI endpoints, authorization, job intake, and mode selection.
- `rag_service/rag_engine.py` — legacy retrieval and answer path.
- `rag_service/adaptive/` — indexing, hybrid search, tools, evidence checks, providers, queue, and state store.
- `rag_service/image_processor.py` and `rag_service/sync.py` — image descriptions and BookStack synchronization.
- `rag_service/tests/` — isolated regression tests; keep these files under version control.
- `rag_service/benchmarks/` — synthetic capacity probes.
- `docs/` — deeper architecture, operations, and historical evaluation notes. Some notes predate the current partial-answer and follow-up flow; check the code and `.env.example` for current defaults.
