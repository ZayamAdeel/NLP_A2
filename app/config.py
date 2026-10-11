"""
Central configuration for the E-Commerce Order Support Assistant.

All values can be overridden with environment variables so the same
code runs unchanged on any teammate's machine / any grading machine.
"""
import os

# ---- Ollama connection -----------------------------------------------------
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5:1.5b-instruct-q4_K_M")

# Request timeout for a single generation call (seconds). Local CPU inference
# on small models is slow-ish on first token, so keep this generous.
OLLAMA_TIMEOUT = float(os.getenv("OLLAMA_TIMEOUT", "120"))

# ---- Conversation memory management ----------------------------------------
# We never send the *entire* raw history once a session grows long. Instead we
# use a hybrid "sliding window + rolling summary" scheme (see README, Phase II
# section, for the full rationale):
#
#   1. The domain system prompt is ALWAYS sent (it defines who the bot is).
#   2. A single "rolling summary" message captures anything older than the
#      window (updated in place, never grows unbounded).
#   3. The most recent MAX_TURNS_IN_WINDOW user/assistant turn-pairs are sent
#      verbatim so the model has precise, faithful access to recent context.
#
# This keeps prompt size roughly constant (O(1) growth) no matter how long
# the conversation runs, which matters a lot for CPU inference latency.
MAX_TURNS_IN_WINDOW = int(os.getenv("MAX_TURNS_IN_WINDOW", "6"))

# Rough token budget for the whole prompt (system + summary + window). This is
# a heuristic (1 token ~= 4 chars for English) since we deliberately avoid
# pulling in a tokenizer dependency just to estimate length.
MAX_PROMPT_TOKENS = int(os.getenv("MAX_PROMPT_TOKENS", "1800"))
CHARS_PER_TOKEN_ESTIMATE = 4

# When the window needs to be compacted, we ask the *same* local model to
# summarize the turns falling out of the window. This is plain prompting of
# the model we already loaded -- not RAG, not an external tool/agent.
SUMMARY_TRIGGER_TURNS = MAX_TURNS_IN_WINDOW + 2

# ---- Session management -----------------------------------------------------
SESSION_IDLE_TIMEOUT_SECONDS = int(os.getenv("SESSION_IDLE_TIMEOUT_SECONDS", "1800"))

# ---- Generation parameters ---------------------------------------------------
GENERATION_OPTIONS = {
    "temperature": 0.4,
    "top_p": 0.9,
    "repeat_penalty": 1.1,
    "num_ctx": 4096,
}
