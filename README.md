# BrightCart Support Assistant — Assignment 2 (RAG)

A local, CPU-oriented e-commerce support chatbot built with **FastAPI**, **WebSockets**, and **Ollama**.

Assignment 1 introduced prompt orchestration and conversation memory. Assignment 2 adds **Retrieval-Augmented Generation (RAG)** over a 50-document BrightCart knowledge base. Before generating an answer, Ava retrieves relevant excerpts and can show the documents used as citations.

| Component | Implementation |
|---|---|
| Domain | Fictional electronics and accessories store, BrightCart |
| Chat model | `qwen2.5:1.5b-instruct-q4_K_M` via Ollama |
| Embedding model | `all-minilm` via Ollama |
| Vector search | Exact inner-product search with FAISS when available; NumPy fallback |
| Backend and interface | FastAPI, WebSocket streaming, HTML/CSS/JavaScript UI |
| Knowledge base | 50 Markdown documents across six categories |
| External LLM APIs | None; models run locally through Ollama |

## Contents

1. [Use case and conversation design](#1-use-case-and-conversation-design)
2. [Architecture and project structure](#2-architecture-and-project-structure)
3. [Models and vector search](#3-models-and-vector-search)
4. [Setup and run](#4-setup-and-run)
5. [Build and update the index](#5-build-and-update-the-index)
6. [How retrieval works](#6-how-retrieval-works)
7. [WebSocket and REST API](#8-websocket-and-rest-api)
8. [Tests](#9-tests)
9. [Changes since Assignment 1](#10-changes-since-assignment-1)
10. [Configuration reference](#11-configuration-reference)

---

## 1. Use case and conversation design

BrightCart is a fictional electronics and accessories store. Its support assistant, **Ava**, helps with:

1. **Product questions:** specifications, compatibility, setup, and troubleshooting.
2. **Orders:** tracking guidance, cancellations, edits, and confirmation emails.
3. **Policies and accounts:** shipping, returns, refunds, exchanges, payments, and account help.

Requests outside the supported domain (for example, weather or general coding help) are declined and redirected. Ava has **no live order, carrier, or payment-system access**. It explains documented policies and processes but must not claim that it has found, cancelled, or refunded a real order.

### Conversation stages

The conversation manager tracks these stages per session:

`greeting → intent_identification → info_gathering → resolution → confirmation → closing`

| Stage | Purpose |
|---|---|
| `greeting` | Greet the user and offer supported topics |
| `intent_identification` | Clarify a vague request, such as “I need help” |
| `info_gathering` | Collect details needed to respond, such as the product or issue |
| `resolution` | Give a policy-grounded answer or next step |
| `confirmation` | Check whether the question has been answered |
| `closing` | End the conversation when the user is finished |

**Topic switches:** Each message is keyword-scored against the supported intents. If it matches a different intent from the active one, the manager updates `active_intent` and returns to `info_gathering` while preserving history. Retrieval is performed for each substantive message, so switching from returns to shipping can retrieve shipping documents.

**Off-topic messages:** A whole-word keyword filter blocks clear off-topic requests before retrieval or generation. Less obvious cases are handled by the system prompt. When relevant excerpts are retrieved, the prompt instructs Ava to answer from those excerpts rather than asking for unnecessary details.

### Example dialogues

These examples illustrate intended behaviour. Replace them with transcripts captured from your own runs before submission.

**Grounded product answer with citations**

```text
User: Can I use the H200 headphones with my laptop?
Ava: Yes, if your laptop supports Bluetooth audio, you can pair them through its
     Bluetooth settings (see [1]). The USB-C port is for charging only, not wired audio.
     Sources: [1] WaveSound H200 Compatibility FAQ
              [2] WaveSound H200 Wireless Headphones
```

**Policy answer followed by a topic switch**

```text
User: Can I return a gift card?
Ava: No. Gift cards are non-returnable (see [1]).
User: Okay. How long does express shipping take?
Ava: Express shipping normally takes 1–2 business days after dispatch (see [1]).
     It does not guarantee a specific delivery hour.
```

**No relevant document**

```text
User: Do you sell dishwashers?
Ava: I don't have that information in my knowledge base. Please contact BrightCart support.

User: What's the weather like today?
Ava: I'm Ava, BrightCart's order support assistant. I can help with product questions,
     order tracking, shipping, and returns. Is there anything along those lines I can help with?
```

---

## 2. Architecture and project structure

### Request flow

**Offline indexing**

1. Read Markdown files from `data/corpus/`.
2. Remove boilerplate and split content into sentence-bounded chunks.
3. Prefix each chunk with its document title.
4. Embed the chunks with Ollama's `all-minilm` model.
5. Save vectors, chunk text, and metadata under `index/`.

**Online chat**

1. Receive a WebSocket message.
2. Apply the off-topic guard and update the conversation stage.
3. Retrieve relevant chunks unless retrieval is unnecessary (for example, small talk).
4. Fit retrieved excerpts and conversation history within the model's context budget.
5. Stream the answer from Ollama.
6. Send a `done` event containing timing information and source citations when available.
7. Run conversation summarisation after `done`, so it does not delay the answer.

### Key files

| File or directory | Responsibility |
|---|---|
| `app/main.py` | FastAPI app, WebSocket endpoint, startup, and health endpoint |
| `app/conversation_manager.py` | Conversation stages, retrieval integration, prompt budget, and citations |
| `app/llm_engine.py` | Ollama chat generation and streaming |
| `app/domain.py` | Assistant instructions and domain behaviour |
| `app/config.py` | Model settings and RAG configuration |
| `app/schemas.py` | Request and response schemas |
| `app/rag/chunking.py` | Markdown parsing and chunking |
| `app/rag/embedder.py` | Async Ollama embedding client |
| `app/rag/store.py` | Exact vector search and index persistence |
| `app/rag/retriever.py` | Query embedding, search, threshold, cache, timeout, and circuit breaker |
| `scripts/build_index.py` | Offline index-building pipeline |
| `scripts/eval_retrieval.py` | Retrieval evaluation and threshold calibration |
| `benchmark/benchmark_retrieval.py` | Retrieval-only latency benchmark |
| `benchmark/benchmark_e2e.py` | End-to-end comparison with and without RAG |
| `benchmark/benchmark_latency.py` | LLM latency benchmark from Assignment 1 |
| `data/corpus/` | 50 Markdown knowledge-base documents |
| `data/eval_queries.json` | Labelled retrieval evaluation queries |
| `static/` | HTML, CSS, JavaScript, and citation UI |
| `tests/` | Unit, API, and manual stress tests |
| `index/` | Generated index files; rebuildable from the corpus |

---

## 3. Models and vector search

### Chat model: `qwen2.5:1.5b-instruct-q4_K_M`

The Assignment 1 chat model is retained and served locally through Ollama. The 1.5B-parameter quantised model is intended to keep CPU inference practical while following the assistant's instructions. Change `OLLAMA_MODEL` to use another locally available Ollama model.

### Embedding model: `all-minilm`

The embedding model is served by Ollama and produces 384-dimensional vectors. Using Ollama for embeddings avoids adding a separate PyTorch or `sentence-transformers` installation.

Embedding requests use asynchronous `httpx` I/O and a shared HTTP client. The model is warmed up in the background at startup to reduce first-query delay.

**Trade-off:** The chat and embedding models compete for CPU and memory. The embedding call also adds an HTTP request to the retrieval path.

### Vector search: FAISS or NumPy

The index uses exact inner-product search over L2-normalised vectors, equivalent to cosine similarity for normalised vectors:

- If `faiss-cpu` is installed, the store uses FAISS `IndexFlatIP`.
- Otherwise, it uses NumPy matrix multiplication.
- The persisted index consists of `vectors.npy`, `chunks.json`, and `meta.json`.

For this small corpus, exact search avoids the extra setup and maintenance of an approximate index or separate vector database. The NumPy path was used during development; the FAISS path is intended to return equivalent exact-search results.

---

## 4. Setup and run

### Prerequisites

- Python 3.11 or newer
- [Ollama](https://ollama.com/) installed and running
- Project files, including `requirements.txt` and `data/corpus/`

Run commands from the **project root**—the directory containing `app/`, `scripts/`, `data/`, and `requirements.txt`.

### Step 1 — Create and activate a virtual environment

**Windows PowerShell (VS Code terminal)**

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
```

If `py -3.11` is unavailable but `python` points to Python 3.11 or newer, use:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

If PowerShell blocks activation, you can use the environment's Python directly instead. For example:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

**macOS / Linux**

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### Step 2 — Install dependencies

With the virtual environment active, run:

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

`faiss-cpu` is optional where the project and platform support it; the implementation falls back to NumPy when FAISS is unavailable.

### Step 3 — Start Ollama and download the models

Make sure the Ollama service is running. On Windows, launch Ollama from the Start menu if needed. On macOS, open the Ollama app. On Linux, use your installed Ollama service or run `ollama serve` in a separate terminal.

Download the required models:

```bash
ollama pull qwen2.5:1.5b-instruct-q4_K_M
ollama pull all-minilm
```

Keep Ollama running while you build the index and use the app.

### Step 4 — Build the retrieval index

```bash
python scripts/build_index.py
```

This creates the index files under `index/`. See [Build and update the index](#5-build-and-update-the-index) for full rebuild instructions.

### Step 5 — Evaluate retrieval and calibrate the threshold

```bash
python scripts/eval_retrieval.py
```

Review the output and set `MIN_SCORE` in `app/config.py` (or through the supported environment-variable configuration) based on the evaluation. The default value `0.30` is a placeholder and should not be treated as calibrated until evaluation has been run.

### Step 6 — Start the web application

```powershell
python -m uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
```

Using `python -m uvicorn` helps ensure Uvicorn runs from the active Python environment. `--reload` is useful during development; omit it for benchmark runs.

Open **http://127.0.0.1:8000** in your browser.

### Quick-start checklist

- [ ] Virtual environment activated
- [ ] Dependencies installed
- [ ] Ollama running
- [ ] Both Ollama models downloaded
- [ ] Index built
- [ ] Retrieval threshold evaluated and configured
- [ ] Server started and UI opens in the browser

### Updating the knowledge base

1. Add, edit, or remove Markdown files under `data/corpus/`.
2. Run `python scripts/build_index.py`.
3. Restart the web server if necessary.

The indexer is incremental: unchanged chunks can reuse their existing vectors. Use `python scripts/build_index.py --full` to re-embed the entire corpus.

---

## 5. Build and update the index

Run the index-building script from the project root:

```bash
python scripts/build_index.py          # Incremental build
python scripts/build_index.py --full   # Re-embed all chunks
```

### Chunking strategy

Implemented in `app/rag/chunking.py`:

1. The first `# ` heading is used as the document title; the parent folder identifies the category.
2. The `## Source` section is removed because the corpus files share boilerplate there. Embedding repeated boilerplate could inflate similarity for unrelated queries and make no-match detection less reliable.
3. Text is split at sentence boundaries and packed into chunks of at most **600 characters**, with a **one-sentence overlap**.
4. Each chunk is prefixed with its document title so the embedding retains context about the product or policy.

The current corpus is short—approximately 440 characters per document—so the 50 documents currently produce 50 chunks. The chunking logic is also tested against longer documents.

### Incremental indexing

Each chunk is keyed by the SHA-256 hash of its text. On an incremental run:

- Unchanged chunks reuse their vectors.
- New or edited chunks are embedded.
- Removed documents are removed from the index.
- A change to the embedding model triggers re-embedding.
- Files are written atomically using a temporary file and rename.

The implementation notes report that an unchanged rebuild embeds zero of 50 chunks. A test edit/add/delete scenario reported two new embeddings, 48 reused vectors, and two removed chunks.

---

## 6. How retrieval works

### Main configuration

Settings are defined in `app/config.py` and can be overridden through environment variables where supported.

| Setting | Default | Purpose |
|---|---:|---|
| `TOP_K` | `4` | Number of chunks retrieved per query; at least three are retrieved |
| `MIN_SCORE` | `0.30` | Minimum similarity threshold; currently an uncalibrated placeholder |
| `CHUNK_MAX_CHARS` | `600` | Maximum chunk length |
| Chunk overlap | 1 sentence | Preserves context across chunk boundaries |
| `RAG_MAX_CONTEXT_TOKENS` | `800` | Maximum token budget for retrieved text |
| `RETRIEVAL_TIMEOUT` | `1.5` seconds | Timeout for embedding and search |
| `RETRIEVAL_CACHE_SIZE` | `256` | Maximum number of cached retrieval results |

### Per-message flow

1. Check whether the message is off-topic.
2. Update the conversation stage.
3. Retrieve relevant chunks when appropriate.
4. Build a prompt that fits the model's context budget.
5. Stream the answer.
6. Record the successful turn and send the `done` event, including source information where available.

Greetings, thanks, and goodbyes skip retrieval. Short follow-ups (fewer than six words) use the latest substantive user message together with the follow-up text to preserve context.

### Adding excerpts to the prompt

Retrieved chunks are numbered and added to the final user message, followed by an instruction to answer from the excerpts. For example:

```text
Knowledge base excerpts:

[1] WaveSound H200 Compatibility FAQ
Q: Can I use H200 with my laptop and phone? A: Yes, if ...

[2] WaveSound H200 Wireless Headphones
...

Answer directly from the excerpts above. Ask a follow-up only if you truly need more information.

Customer message: Can I use the H200 headphones with my laptop?
```

The system prompt tells Ava to treat retrieved excerpts as the source of truth for product specifications and policies, cite the relevant source (for example, `(see [1])`), and admit when the excerpts do not contain the answer.

Excerpts are not stored in conversation history, so they do not accumulate across turns. Keeping excerpts in the latest user message is also intended to preserve a stable system-prompt/history prefix for prompt caching; this is a design intention, not a measured performance claim.

### Context-window budget

The model context is configured for `4096` tokens, with `512` tokens reserved for the response. This leaves a nominal prompt budget of **3584 tokens**. Token use is estimated conservatively at approximately 3.5 characters per token.

When space is limited, the prompt builder prioritises:

1. System prompt and rolling summary
2. Current customer message
3. Retrieved excerpts, ranked by relevance and capped at 800 tokens
4. Recent conversation history, dropping the oldest turns first

Chunks are kept whole where possible. If the best chunk is too large, it can be cut at a sentence boundary and marked `[truncated]`. Retrieved context should not displace the current question. Summarisation runs after the `done` event so it does not delay the response.

### Citation UI

The WebSocket `done` event includes `sources` for the chunks actually included in the prompt. The UI displays citation chips under the answer. Clicking a chip shows the document path, relevance score, and exact excerpt shown to the model.

When no relevant documents match, or the knowledge base is unavailable, the UI reports that status rather than implying that the answer is grounded in retrieved documents.

---

## 7. WebSocket and REST API

### WebSocket endpoint

`ws://localhost:8000/ws/chat`

The request format and core event types remain compatible with Assignment 1. The `done` event adds two optional, backwards-compatible fields: `sources` and `retrieval`.

**Request example**

```json
{
  "type": "message",
  "session_id": "...",
  "text": "Can I use the H200 headphones with my laptop?"
}
```

**Illustrative `done` event**

```json
{
  "type": "done",
  "time_to_first_token": 0.82,
  "total_time": 3.4,
  "tokens_per_second": 14.2,
  "stage": "info_gathering",
  "sources": [
    {
      "n": 1,
      "title": "WaveSound H200 Compatibility FAQ",
      "doc_id": "product_faqs/faq_016_headphone_compatibility.md",
      "category": "product_faqs",
      "score": 0.62,
      "snippet": "Q: Can I use H200 with my laptop and phone? A: Yes, ..."
    }
  ],
  "retrieval": {
    "status": "ok",
    "ms": 18.4,
    "cached": false
  }
}
```

The values above are an **example payload**, not a benchmark result. `retrieval.status` can be `ok`, `no_match`, `skipped`, `unavailable`, or `disabled`.

`total_time` represents wall-clock generation time, while `tokens_per_second` reports decode speed.

### REST endpoints

- `GET /api/health` — health status, including `retrieval_ready`, `index_chunks`, and `embed_model`
- `POST /api/session` — create a session
- `GET /api/session/{id}` — inspect a session
- `DELETE /api/session/{id}` — delete a session

---

## 8. Tests

Run only the RAG test file:

```powershell
python -m pytest tests/test_rag.py -v
```

Run the full test suite:

```powershell
python -m pytest tests/ -v
```

If pytest is not installed in the active environment:

```powershell
python -m pip install pytest
```

The repository includes tests for chunking, retrieval behaviour, error handling, context budgeting, conversation management, and the API. Confirm the current test count from your own run rather than relying on a count in this README.

---

## 9. Changes since Assignment 1

- **Static file serving:** Fixed the mismatch between `/static/style.css` and `/static/app.js` URLs and the static-file mount.
- **Off-topic matching:** Replaced substring matching with whole-word matching to avoid blocking legitimate support messages because of accidental substrings.
- **Timing metrics:** `total_time` now represents wall-clock time; decode speed is reported separately as `tokens_per_second`.
- **Conversation reliability:** Conversation history is recorded only after successful generation, and summarisation runs after the `done` event.
- **Context budgeting:** Token estimation is more conservative (about 3.5 characters per token), and `num_predict` caps response length.
- **Prompt budget:** Replaced the earlier `MAX_PROMPT_TOKENS` approach with a budget derived from the configured model context length.
- **Domain consistency:** Standardised the brand name as BrightCart and removed an unused example dialogue that referred to a product absent from the corpus.
- **RAG integration:** Added document chunking, local embeddings, persistent vector search, retrieval caching, timeouts, a circuit breaker, prompt context management, and visible source citations.

---

## 10. Configuration reference

Review `app/config.py` for all settings. Important configuration names include:

- `OLLAMA_MODEL`
- `EMBED_MODEL`
- `TOP_K`
- `MIN_SCORE`
- `RETRIEVAL_TIMEOUT`
- `INDEX_DIR`
- `CORPUS_DIR`
- `RAG_MAX_CONTEXT_TOKENS`

Check `app/config.py` for exact supported environment-variable names and defaults.
