# BookStack AI Assistant (RAG-Powered)

A self-hosted, enterprise-grade AI-powered documentation assistant for [BookStack](https://www.bookstackapp.com/). Built with **Retrieval-Augmented Generation (RAG)** using FastAPI, ChromaDB vector search, and Google Gemini to deliver context-aware, permission-bounded Q&A over your BookStack knowledge base.

---

## Key Features

- 💬 **Floating & Draggable AI Chat Widget** — Embedded seamlessly into BookStack pages via custom theme overriding with CSP nonce compatibility.
- 📄 **Active Page Awareness & 1-Click Summary** — Detects the current article you are viewing. Ask questions about "this page" or click `⚡ Summarize` for instant key takeaways and steps.
- 🔄 **Multi-Turn Conversation Memory** — Remembers previous turns in the session, accurately resolving pronouns and follow-up inquiries (e.g., *"What is their email address?"*, *"How long does this take?"*).
- ⏱️ **Context Window Guard & New Chat Reset** — Monitors session turns and token usage. Proactively alerts when conversations get long (`⚠️ Conversation Length Notice`) and provides a 1-click `[🔄 New Chat]` button to reset the session cleanly.
- 🛡️ **Role-Based Access Control (RBAC)** — Feature gating via `AI_ALLOWED_ROLES` (e.g. `admin,internal`). Disable access for external or guest users with zero code changes.
- 🔐 **Cryptographic Security (HMAC-SHA256)** — Session tokens are signed server-side in BookStack and cryptographically verified by the RAG service, enforcing strict page visibility boundaries.
- 🌐 **Two-Layer Bilingual Intent Router** — Translates and expands user queries (e.g., Turkish to English documentation keywords) for high-accuracy vector retrieval and cites relevant articles with clickable chips.
- ⚡ **Real-Time Webhook Synchronization** — Automatic incremental indexing on page creation, updates, and deletion.

---

## Architecture

```
┌────────────────────────────────┐       ┌─────────────────────────────────┐       ┌────────────────────────┐
│     BookStack (Port 6875)      │──────▶│    RAG Service (Port 8000)      │──────▶│   Google Gemini API    │
│                                │       │                                 │       │                        │
│  - Custom Theme Override       │       │  - FastAPI REST API             │       └────────────────────────┘
│  - Active Page Context         │       │  - Intent Router (Layer 1)      │
│  - HMAC Token Signing          │       │  - Gemini Generation (Layer 2)  │       ┌────────────────────────┐
│  - Role Gating (Admin/Internal)│       │  - BookStack Sync Engine        │──────▶│       ChromaDB         │
│  - Webhooks (CRUD)             │──────▶│  - Strict Permission Trimming   │       │      (Vector DB)       │
└────────────────────────────────┘       └─────────────────────────────────┘       └────────────────────────┘
```

---

## Quick Start

### 1. Clone & Configure

```bash
git clone https://github.com/gunaysuleyman/bookstack-ai-assistant.git
cd bookstack-ai-assistant
cp .env.example .env
```

Edit `.env` and set your **Gemini API Key**:

```env
GEMINI_API_KEY=your_gemini_api_key_here
```

### 2. Launch Services

```bash
docker compose up -d
```

### 3. Setup BookStack API Token

1. Open `http://localhost:6875` and log in (default: `admin@admin.com` / `password`).
2. Navigate to **Settings → Users → Admin → API Tokens → Create Token**.
3. Copy the Token ID and Secret into your `.env`:

```env
BOOKSTACK_TOKEN_ID=your_token_id
BOOKSTACK_TOKEN_SECRET=your_token_secret
```

4. Restart the RAG service:

```bash
docker compose restart rag_service
```

### 4. Sync Content

Trigger initial sync:

```bash
curl -X POST http://localhost:8000/api/sync \
  -H "X-RAG-Token: my_super_secret_local_token_123"
```

Or configure a BookStack webhook for automatic real-time sync:
- Navigate to **Settings → Webhooks → Create Webhook**
- **Endpoint URL**: `http://rag_service:8000/api/webhook`
- **Events**: Page Create, Page Update, Page Delete

---

## Project Structure

```
.
├── docker-compose.yml              # Multi-container orchestration (BookStack + MariaDB + RAG)
├── .env.example                    # Environment template
├── .env                            # Active environment variables (git-ignored)
├── rag_service/
│   ├── Dockerfile                  # Python container specification
│   ├── main.py                     # FastAPI endpoints, auth & context limits
│   ├── rag_engine.py               # 2-layer RAG: Intent routing + Gemini generation
│   ├── sync.py                     # BookStack API sync with shelf/book hierarchy
│   ├── html_cleaner.py             # HTML to structured Markdown text chunker
│   └── requirements.txt
└── widget/
    └── bookstack_ai_widget.html    # Blade template (mounted into BookStack custom theme)
```

---

## Configuration Reference

| Variable | Description | Default |
|----------|-------------|---------|
| `GEMINI_API_KEY` | Google Gemini API key | _(required)_ |
| `AI_PROVIDER` | AI backend provider (`gemini` or `openai`) | `gemini` |
| `BOOKSTACK_TOKEN_ID` | BookStack API token ID | _(required for sync)_ |
| `BOOKSTACK_TOKEN_SECRET` | BookStack API token secret | _(required for sync)_ |
| `RAG_SECRET_TOKEN` | Shared secret for widget & webhook authentication | `my_super_secret_local_token_123` |
| `AI_ALLOWED_ROLES` | Comma-separated BookStack roles permitted to use AI | `admin,internal` |
| `MAX_RECOMMENDED_TURNS` | Number of chat turns before suggesting new chat | `5` |
| `BOOKSTACK_EXTERNAL_URL` | Public-facing BookStack URL | `http://localhost:6875` |

---

## How It Works

1. **Custom Theme Override**: The widget Blade view (`widget/bookstack_ai_widget.html`) is mounted to `/config/www/themes/custom/layouts/parts/base-body-end.blade.php`.
2. **Access Control & HMAC**: BookStack verifies the user's role against `AI_ALLOWED_ROLES`. If authorized, it generates an HMAC-signed payload containing the user ID, role list, and allowed page IDs.
3. **Active Page Detection**: When browsing a specific page, BookStack injects the page ID and title, allowing the AI to prioritize or summarize active content.
4. **Multi-Turn Context & Guard**: Conversation history is maintained on the client and sent with each request. If session turns exceed `MAX_RECOMMENDED_TURNS`, the user receives a notice recommending `🔄 Start New Chat`.
5. **Vector Search & Grounding**: Queries are routed through ChromaDB, filtered strictly by authorized pages, and synthesized with citation chips pointing directly to original articles.

---

## License

MIT
