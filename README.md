# BookStack AI Assistant (RAG-Powered)

A self-hosted AI-powered documentation assistant for [BookStack](https://www.bookstackapp.com/). Uses **Retrieval-Augmented Generation (RAG)** with ChromaDB vector search and Google Gemini to provide intelligent Q&A over your BookStack content.

## Features

- 🤖 **AI Chat Widget** — Floating chat panel injected into every BookStack page via theme override
- 🔍 **RAG Pipeline** — ChromaDB vector search + Gemini AI for context-aware answers
- 📚 **Full Hierarchy Awareness** — Understands Shelves → Books → Chapters → Pages → Tags
- 🔄 **Auto-Sync** — Webhook-driven indexing when pages are created/updated/deleted
- 🐳 **Docker Compose** — One-command deployment with zero manual configuration
- 🎨 **Draggable FAB** — The AI button can be dragged anywhere on screen
- 🔒 **CSP Compatible** — Works with BookStack's Content Security Policy via Blade nonce

## Architecture

```
┌─────────────────┐     ┌──────────────────┐     ┌─────────────┐
│   BookStack     │────▶│   RAG Service    │────▶│  Gemini AI  │
│  (Port 6875)    │     │  (Port 8000)     │     │   API       │
│                 │     │                  │     └─────────────┘
│  Theme Mount:   │     │  - FastAPI       │
│  AI Widget HTML │     │  - ChromaDB      │     ┌─────────────┐
│                 │     │  - Sync Engine   │────▶│  ChromaDB   │
│  Webhook ──────▶│     │  - HTML Cleaner  │     │  (Vector DB)│
└─────────────────┘     └──────────────────┘     └─────────────┘
```

## Quick Start

### 1. Clone & Configure

```bash
git clone https://github.com/YOUR_USERNAME/bookstack-ai-assistant.git
cd bookstack-ai-assistant
cp .env.example .env
```

Edit `.env` and set your **Gemini API Key**:

```env
GEMINI_API_KEY=your_gemini_api_key_here
```

### 2. Launch

```bash
docker compose up -d
```

### 3. Setup BookStack API Token

1. Open `http://localhost:6875` and log in (default: `admin@admin.com` / `password`)
2. Go to **Settings → Users → Admin → API Tokens → Create Token**
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
curl -X POST http://localhost:8000/api/sync-all \
  -H "X-RAG-Token: my_super_secret_local_token_123"
```

Or set up a BookStack webhook for auto-sync:
- Go to **Settings → Webhooks → Create Webhook**
- URL: `http://rag_service:8000/api/webhook`
- Events: Page Create, Page Update, Page Delete

## Project Structure

```
.
├── docker-compose.yml          # Full stack orchestration
├── .env.example                # Environment template (safe to commit)
├── .env                        # Your secrets (NEVER commit this)
├── .gitignore
├── rag_service/
│   ├── Dockerfile              # Python 3.11 slim container
│   ├── main.py                 # FastAPI endpoints
│   ├── rag_engine.py           # ChromaDB + Gemini RAG logic
│   ├── sync.py                 # BookStack API sync with hierarchy
│   ├── html_cleaner.py         # HTML → text chunking with metadata
│   └── requirements.txt
└── widget/
    └── bookstack_ai_widget.html  # Blade template (auto-mounted via theme)
```

## Configuration

| Variable | Description | Default |
|----------|-------------|---------|
| `GEMINI_API_KEY` | Google Gemini API key | _(required)_ |
| `AI_PROVIDER` | `gemini` or `openai` | `gemini` |
| `BOOKSTACK_TOKEN_ID` | BookStack API token ID | _(required for sync)_ |
| `BOOKSTACK_TOKEN_SECRET` | BookStack API token secret | _(required for sync)_ |
| `RAG_SECRET_TOKEN` | Shared secret between widget & RAG service | `my_super_secret_local_token_123` |
| `BOOKSTACK_EXTERNAL_URL` | Public BookStack URL | `http://localhost:6875` |

## How It Works

1. **Theme Override**: The widget HTML is mounted into BookStack's theme system at `/config/www/themes/custom/layouts/parts/custom-head.blade.php`, making it appear on every page automatically.

2. **CSP Nonce**: The widget uses Blade's `{{ $cspNonce }}` to comply with BookStack's Content Security Policy.

3. **Vector Search**: Page content is chunked, embedded locally, and stored in ChromaDB. User questions are matched against these vectors.

4. **Hierarchy Context**: Every AI response includes awareness of the full Shelf → Book → Chapter → Page structure.

## License

MIT
