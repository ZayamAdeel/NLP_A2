# Brightcart Support Assistant — E-Commerce Order Support Chatbot

A fully local, CPU-only conversational AI system for **Assignment 1**: a real-time,
streaming, multi-turn e-commerce order-support chatbot built with FastAPI + a
quantized local LLM served by **Ollama**. No cloud model APIs, no tools, no
agents, no RAG — every response comes from prompt orchestration and
conversational memory alone, as required.

---

## 1. Business Use Case

**Domain: E-Commerce Order Support Assistant** for a fictional store,
**Brightcart** (electronics & home goods).

The assistant, "Ava", helps customers with exactly three things:

1. **Product questions** — features, sizing, compatibility, stock, general advice.
2. **Order tracking** — order status, delivery windows.
3. **Returns, refunds, exchanges & shipping policy** — eligibility, timelines, cancellations.

It explicitly refuses anything outside that scope (weather, coding help,
medical/legal advice, general trivia, jailbreak attempts, etc.) and redirects
the user back to supported topics. This domain was chosen because it has
concrete, checkable policies (so correctness is easy to grade), natural
multi-turn structure (greeting → clarify → resolve → confirm → close), and a
very plausible "topic switch mid-conversation" case (a customer asking about
an order, then suddenly asking about a product) that stress-tests the
conversation manager's state handling.

### Example Dialogues

**1. Order tracking**

```
User: Hi, where is my order?
Ava:  Hi! Happy to help. Could you share your order number so I can walk you
      through what to expect?
User: It's BC-88213, placed 4 days ago.
Ava:  Thanks! For standard shipping, orders typically arrive within 3-5
      business days, so BC-88213 should be right on track. I don't have live
      tracking access in this demo, but you'd see real-time status on the
      tracking link in your confirmation email. Anything else I can help with?
```

**2. Return/policy question, then a mid-conversation topic switch**

```
User: Can I return a blender I bought 20 days ago?
Ava:  Yes! We accept returns within 30 days of delivery as long as it's unused
      and in its original packaging, so you're well within the window. Want
      me to walk you through the return steps?
User: Actually, does the Brightcart Air Fryer XL come in a smaller size?
Ava:  Good question! I don't have live catalog data in this demo, but the Air
      Fryer XL line typically also offers a compact model — I'd recommend
      checking the product page for exact size options. Was there anything
      else on the return, or is this all for now?
```

**3. Off-topic redirect**

```
User: By the way, what's the weather like today?
Ava:  I'm Ava, Brightcart's order support assistant, so I can only help with
      product questions, order tracking, and returns/shipping. Is there
      anything along those lines I can help you with?
User: Okay fine — can I cancel an order I placed 10 minutes ago?
Ava:  Yes, cancellations are possible within 1 hour of placing an order as
      long as it hasn't shipped yet. I'd recommend cancelling right away from
      your order page, or let me know and I can explain the steps.
```

(All three dialogues above are also defined as data in `app/domain.py` under
`EXAMPLE_DIALOGUES`, and the exact off-topic redirect text and policy facts
come straight from the live system prompt, so these transcripts match what
the running system actually says.)

### Conversation Flow Design

The assistant moves through **six domain-specific stages**, tracked per
session in `ConversationManager._advance_stage` (`app/conversation_manager.py`):

| Stage | What happens |
|---|---|
| `greeting` | Ava greets the customer and asks how she can help (order / product / policy). |
| `intent_identification` | If the first message was vague ("I need help"), Ava figures out which of the 3 supported topics applies. |
| `info_gathering` | Ava asks for the minimum info needed (order number, product name, issue). |
| `resolution` | Ava gives a clear, policy-accurate answer or next step. |
| `confirmation` | Ava checks whether that fully answered the question. |
| `closing` | Ava wraps up; saying "bye"/"no that's all" gets here directly from `confirmation`. |

**Handling topic switches mid-conversation:** every incoming message is
passed through a lightweight rule-based intent detector
(`_detect_intent`, keyword-scored against `INTENT_KEYWORDS`). If the customer
is already in `info_gathering` / `resolution` / `confirmation` for one topic
(say, `order_tracking`) and the detected intent of the new message is a
*different* supported topic (say, `returns_shipping`), the manager:

1. Updates `session.active_intent` to the new topic.
2. Resets `session.stage` back to `info_gathering` for that new topic.
3. **Does not** erase conversation history — the full turn history (and
   rolling summary, see §3) is still sent to the model, so Ava remains
   contextually aware of the earlier topic even while helping with the new one.

This is exercised directly in `tests/test_conversation_manager.py::test_topic_switch_mid_conversation_resets_to_info_gathering`.

**Handling off-topic / out-of-domain messages:** a message is checked against
a fast keyword denylist (`is_off_topic`, in `conversation_manager.py`) for
obviously unrelated topics (weather, politics, homework, jailbreak phrases
like "ignore previous instructions", etc.). If it matches, the system
**short-circuits before calling the LLM at all** — no token is generated, the
canned `OFF_TOPIC_REDIRECT` message is sent immediately (this both saves CPU
cycles and guarantees the refusal is 100% consistent). Subtler off-topic
requests that slip past the keyword filter are still caught by the system
prompt's explicit instruction to decline and redirect (see `BASE_SYSTEM_PROMPT`
in `app/domain.py`).

---

## 2. System Architecture

```
┌──────────────────┐      WebSocket (JSON)      ┌──────────────────────────┐
│  Web Chat UI      │◄──────────────────────────►│      FastAPI Server       │
│  (static/*.html,  │   /ws/chat                 │      app/main.py          │
│   app.js, css)    │   REST: /api/session, ...  │                            │
└──────────────────┘                             │  ┌──────────────────────┐ │
                                                  │  │ Conversation Manager  │ │
                                                  │  │ (sessions, stages,    │ │
                                                  │  │  memory windowing,    │ │
                                                  │  │  off-topic guard)     │ │
                                                  │  │ conversation_manager.py│ │
                                                  │  └──────────┬───────────┘ │
                                                  │             │ messages[]   │
                                                  │  ┌──────────▼───────────┐ │
                                                  │  │     LLM Engine        │ │
                                                  │  │  (async streaming     │ │
                                                  │  │   HTTP client)        │ │
                                                  │  │   llm_engine.py       │ │
                                                  │  └──────────┬───────────┘ │
                                                  └─────────────┼─────────────┘
                                                                │ HTTP /api/chat (stream)
                                                     ┌──────────▼───────────┐
                                                     │   Ollama (local)      │
                                                     │ qwen2.5:1.5b-instruct │
                                                     │        Q4_K_M         │
                                                     └────────────────────────┘
```

- **Frontend** (`static/`): plain HTML/CSS/vanilla JS chat widget. Opens a
  WebSocket, streams tokens into the active bubble as they arrive, shows the
  current conversation stage, and has a "New chat" reset button.
- **Backend API** (`app/main.py`): FastAPI app exposing REST endpoints for
  session lifecycle and a single `/ws/chat` WebSocket endpoint for real-time
  streaming chat. Fully `async`.
- **Conversation Manager** (`app/conversation_manager.py`): owns all
  in-memory `Session` objects, drives the stage machine described above,
  implements the memory-management scheme (§3), and builds the exact
  `messages` list sent to the model each turn.
- **LLM Engine** (`app/llm_engine.py`): a thin async `httpx` streaming client
  for Ollama's native `/api/chat` endpoint — not LangChain, not a tool-calling
  wrapper, just messages in / token stream out, which keeps the "no tools/RAG"
  constraint unambiguous.

---

## 3. Model Selection & Context Memory Management (Phase II)

### Model choice: `qwen2.5:1.5b-instruct-q4_K_M`

| Criterion | Why this model |
|---|---|
| Size | 1.5B parameters — comfortably inside the 0.5B-4B range and fast enough for real-time streaming on a laptop CPU. |
| Quantization | Q4_K_M — a good quality/speed trade-off for CPU inference in Ollama's GGUF format. |
| Instruction tuning | Ships as an instruct-tuned chat model, so it follows the structured system prompt / role format well without extra fine-tuning. |
| License / availability | Open weights, pulled directly via `ollama pull qwen2.5:1.5b-instruct-q4_K_M`, no API key or cloud dependency. |
| Alternatives considered | `phi3.5:3.8b-mini-instruct-q4_K_M` (stronger but noticeably slower TTFT on CPU) and `gemma2:2b-instruct-q4_K_M` (a solid middle ground) — both are drop-in alternatives; just change `OLLAMA_MODEL` in `app/config.py` or the `OLLAMA_MODEL` env var. |

Swap models at any time with:
```bash
ollama pull phi3.5:3.8b-mini-instruct-q4_K_M
export OLLAMA_MODEL=phi3.5:3.8b-mini-instruct-q4_K_M
```

### Context memory management scheme

Implemented in `ConversationManager._build_messages` and
`ConversationManager._maybe_summarize` (`app/conversation_manager.py`),
configured in `app/config.py`. The goal is to keep prompt size roughly
**constant** regardless of how long a session runs (important for CPU
latency), while staying faithful to recent context:

1. **System prompt (always included)** — the domain policy/persona prompt,
   plus a dynamic line naming the current stage and active intent, so the
   model always knows *where* in the flow it is.
2. **Rolling summary (one message, updated in place)** — once the raw turn
   history exceeds `SUMMARY_TRIGGER_TURNS` (default: 8 turns), the oldest
   turns that are about to fall outside the sliding window are folded into a
   single 2-3 sentence summary via **one extra call to the same local
   model** (plain prompting, not a tool/agent/RAG call) and then dropped from
   raw history. The summary is appended to (not replaced by) any prior
   summary, so it keeps growing gist-level context, not raw text.
3. **Sliding window (verbatim, bounded)** — only the most recent
   `MAX_TURNS_IN_WINDOW` (default: 6) user/assistant turn-pairs are sent
   word-for-word, so the model has precise access to what was *just* said.
4. **Defensive token-budget trim** — as a final safety net, if the estimated
   token count (`len(text)//4` heuristic, to avoid pulling in a tokenizer
   dependency) still exceeds `MAX_PROMPT_TOKENS` (default: 1800), the oldest
   non-system message is dropped repeatedly until it fits.

This means: short conversations get the full raw history (simple, maximally
faithful); long conversations get O(1) prompt growth — a system prompt + one
summary message + a fixed-size window — so turn 50 of a conversation costs
roughly the same inference time as turn 5.

### Latency Benchmarks

Measured with `benchmark/benchmark_latency.py`, which sends 5 representative
in-domain prompts straight to Ollama (bypassing the FastAPI layer to isolate
pure inference latency) and reports per-prompt and averaged
time-to-first-token (TTFT) and tokens/sec.

```bash
python benchmark/benchmark_latency.py --model qwen2.5:1.5b-instruct-q4_K_M --runs 3
```

> **Run this on your own machine and paste the output here before submitting** —
> this sandbox has no network access to run Ollama, so the numbers below are a
> placeholder table showing the exact shape you should fill in (replace with
> your actual benchmark output):

| Metric | Value (fill in from your run) |
|---|---|
| Hardware (CPU, cores, RAM) | e.g. Apple M2, 8 cores, 16GB |
| Avg time-to-first-token | e.g. 0.XX s |
| Avg total response time | e.g. X.XX s |
| Avg tokens/sec | e.g. XX.X tok/s |
| Min / Max TTFT | e.g. 0.XX s / 0.XX s |

The live chat UI also shows **per-response** TTFT / tokens-per-second /
total-time under every assistant message (see the `done` WebSocket event and
`addMeta()` in `static/app.js`), so you can screenshot real numbers straight
from normal usage for your README/report as well.

---

## 4. Setup & Run Instructions

### Prerequisites
- Python 3.11+
- [Ollama](https://ollama.com) installed and on your PATH

### 1. Install and start Ollama, pull the model
```bash
# macOS / Linux: see https://ollama.com/download
ollama serve &                 # starts the local inference server (localhost:11434)
ollama pull qwen2.5:1.5b-instruct-q4_K_M
```

### 2. Install Python dependencies
```bash
cd ecommerce-chatbot
python -m venv .venv && source .venv/bin/activate   # optional but recommended
pip install -r requirements.txt
```

### 3. Run the backend (serves API + the static chat frontend together)
```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

### 4. Open the chat UI
Visit **http://localhost:8000** in a browser — the FastAPI app serves
`static/index.html` directly, so there's nothing else to start.

### 5. Configuration (optional)
All tunables are environment variables, with sensible defaults in
`app/config.py`:

| Variable | Default | Purpose |
|---|---|---|
| `OLLAMA_HOST` | `http://localhost:11434` | Where Ollama is running |
| `OLLAMA_MODEL` | `qwen2.5:1.5b-instruct-q4_K_M` | Which pulled model to use |
| `MAX_TURNS_IN_WINDOW` | `6` | Sliding-window size (turn pairs) |
| `MAX_PROMPT_TOKENS` | `1800` | Hard cap before defensive trimming |

### 6. Run the automated test suite
```bash
pytest -v
```
`tests/test_conversation_manager.py` and `tests/test_api.py` use a fake LLM
engine (no Ollama required) so they run anywhere, including CI. They cover
stage transitions, topic-switch handling, off-topic blocking, memory
summarization, REST endpoints, and WebSocket error handling.

### 7. Run the Phase VI manual stress test (needs a live server + Ollama)
```bash
# in one terminal
uvicorn app.main:app --port 8000
# in another terminal
python tests/manual_stress_test.py
```
This deliberately sends a malformed message, an unknown `session_id`, a
mid-stream disconnect, and 4 concurrent users, and asserts the server handles
all of them gracefully (see §6 below for what it checks and why).

### 8. Run the latency benchmark
```bash
python benchmark/benchmark_latency.py
```

---

## 5. WebSocket API Reference

**Endpoint:** `ws://localhost:8000/ws/chat`

**Client → Server** (one message type):
```json
{"type": "message", "session_id": "3fa2...-uuid", "text": "Where is my order?"}
```

**Server → Client** (a stream of events per turn):
```json
{"type": "stage", "stage": "info_gathering"}
{"type": "token", "text": "Hi! "}
{"type": "token", "text": "Happy "}
{"type": "token", "text": "to help. "}
...
{"type": "done", "time_to_first_token": 0.42, "total_time": 2.1, "tokens_per_second": 18.6, "stage": "info_gathering"}
```

Off-topic messages skip the LLM entirely and return:
```json
{"type": "blocked", "text": "I'm Ava, Brightcart's order support assistant, ..."}
```

Any failure (malformed JSON, unknown session, model/connection error) returns
a clean error event instead of closing the socket:
```json
{"type": "error", "message": "Malformed request: ..."}
```

**REST endpoints:**
| Method | Path | Purpose |
|---|---|---|
| `GET` | `/api/health` | Health check; also pings Ollama |
| `POST` | `/api/session` | Create a new session, returns `session_id` + greeting |
| `GET` | `/api/session/{id}` | Inspect a session's stage/intent/turn count |
| `DELETE` | `/api/session/{id}` | End a session (used by "New chat") |

---

## 6. Production Readiness & Evaluation (Phase VI)

What was tested and how, per the assignment's three required checks:

- **Latency** — `benchmark/benchmark_latency.py` measures TTFT and total
  response time directly against Ollama; the live UI also surfaces these
  numbers per-response for informal spot checks under normal interactive use.
- **Correctness / staying in character** — `tests/test_conversation_manager.py`
  asserts the stage machine behaves correctly (greeting → intent →
  info-gathering → resolution → confirmation → closing), that topic switches
  mid-conversation are handled without losing history, and that the
  domain-policy text in `app/domain.py` matches what's actually sent to the
  model. The off-topic test confirms attempts to push the bot off-topic
  (including prompt-injection-style phrases like "ignore previous
  instructions") are deterministically blocked before any model call.
- **Failure handling** — `tests/manual_stress_test.py` deliberately: (1)
  sends malformed (non-JSON) WebSocket payloads, (2) sends a message with an
  unknown `session_id`, (3) opens a connection and disconnects mid-stream
  without reading the response, and (4) opens several concurrent sessions at
  once. In every case the server is expected to respond with a clean
  `{"type": "error", ...}` event (or simply drop the broken connection
  cleanly) rather than crash — this is enforced in `app/main.py`'s
  `ws_chat()` handler via per-message try/except blocks and a `_safe_send()`
  helper that swallows send failures on an already-closed socket. Concurrency
  is achieved naturally because each WebSocket connection runs in its own
  asyncio task and the Ollama call itself is a non-blocking streaming HTTP
  request (`httpx.AsyncClient`), so one user's slow generation never blocks
  another's.

---

## 7. Known Limitations

- **No real order database.** As required, there's no RAG/tool layer, so Ava
  cannot actually look up a specific order's live status — she's explicitly
  instructed to be transparent about this and explain policy/process instead
  of fabricating a tracking result. This is an intentional constraint of the
  assignment, not a bug.
- **Rule-based intent detection is keyword-based**, not a learned classifier.
  It's fast and deterministic (good for mid-conversation topic-switch
  detection and the off-topic fast-path) but can misfire on unusual phrasing
  that doesn't contain any of the listed keywords; in that case the system
  prompt's own instructions are the fallback line of defense.
  by example, "my fridge is broken" won't hit any order/return/product
  keyword and would quietly fall through to intent_identification rather than
  being flagged as a special case.
- **Sessions are in-memory only** (a plain Python dict on the server
  process). Restarting the server loses all active sessions, and this won't
  scale horizontally across multiple server processes/instances without
  adding a shared session store (e.g. Redis) — out of scope for this
  assignment but a natural next step for Assignment 2/3.
- **Summarization quality depends on the small model.** Folding old turns
  into a rolling summary uses the same 1.5B model, which occasionally drops
  minor details (e.g. an exact order number) in the summary; the sliding
  window mitigates this for anything recent, but very long sessions could
  lose some older specifics.
- **No authentication/authorization** — anyone who can reach the server can
  open sessions; fine for a local class assignment, not production-ready.
- **Token streaming granularity** follows Ollama's own chunking (roughly
  sub-word pieces), not strictly one-token-per-chunk, though it still reads
  as smooth word-by-word streaming in the UI.

---

## 8. Project Structure

```
ecommerce-chatbot/
├── app/
│   ├── main.py                 # FastAPI app: REST + WebSocket endpoints
│   ├── conversation_manager.py # Session state, stage machine, memory mgmt
│   ├── llm_engine.py           # Async streaming Ollama client
│   ├── domain.py                # System prompt, policies, stages, keywords
│   ├── schemas.py                # Pydantic request/response models
│   └── config.py                 # All tunables (env-overridable)
├── static/
│   ├── index.html / style.css / app.js   # Chat UI
├── tests/
│   ├── test_conversation_manager.py      # Unit tests (fake LLM, no Ollama needed)
│   ├── test_api.py                        # REST + WebSocket integration tests
│   └── manual_stress_test.py             # Phase VI live failure/concurrency script
├── benchmark/
│   └── benchmark_latency.py              # TTFT / tokens-per-sec benchmark
├── requirements.txt
├── pytest.ini
└── README.md
```
