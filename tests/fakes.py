"""
Test doubles for the RAG layer. HashingEmbedder is a deterministic bag-of-words
embedder (hashed into 512 dims, L2-normalised): it lets the indexing/retrieval
PLUMBING be tested without Ollama. It is lexical, so it says nothing about the
semantic quality of the real embedding model -- use scripts/eval_retrieval.py with
the real model for that.
"""
import asyncio
import hashlib
import re

import numpy as np

_STOP = {"a", "an", "the", "is", "are", "do", "does", "i", "my", "me", "can", "to", "of",
         "and", "or", "it", "in", "on", "for", "with", "how", "what", "if", "you", "your", "be"}


class HashingEmbedder:
    model = "fake-hashing-512"
    DIM = 512

    def __init__(self, delay: float = 0.0, fail: Exception | None = None):
        self.delay, self.fail, self.calls = delay, fail, 0

    async def embed(self, texts):
        self.calls += 1
        if self.delay:
            await asyncio.sleep(self.delay)
        if self.fail:
            raise self.fail
        out = np.zeros((len(texts), self.DIM), dtype=np.float32)
        for row, text in enumerate(texts):
            for tok in re.findall(r"[a-z0-9]+", text.lower()):
                if tok in _STOP:
                    continue
                h = int(hashlib.md5(tok.encode()).hexdigest(), 16)
                out[row, h % self.DIM] += 1.0 if (h >> 20) & 1 else -1.0
                out[row, (h >> 8) % self.DIM] += 1.0
        norms = np.maximum(np.linalg.norm(out, axis=1, keepdims=True), 1e-12)
        return (out / norms).astype(np.float32)

    async def aclose(self):
        pass


class FakeMetrics:
    time_to_first_token = 0.05
    total_time = 0.4
    output_tokens_estimate = 5
    tokens_per_second = 12.5


class RecordingEngine:
    """Fake LLM: records every prompt it receives and streams a canned reply."""

    def __init__(self, reply="Sure, happy to help with that.", fail: Exception | None = None):
        self.reply, self.fail = reply, fail
        self.prompts: list[list[dict]] = []
        self.timeline: list[str] = []

    async def stream_chat(self, messages, options=None):
        self.prompts.append(messages)
        self.timeline.append("stream")
        if self.fail:
            raise self.fail
        for word in self.reply.split(" "):
            yield word + " ", FakeMetrics()

    async def chat_once(self, messages, options=None):
        self.timeline.append("summarize")
        return "Customer asked about their order."


class StaticRetriever:
    """Retriever stand-in returning a fixed result (for API-level tests)."""

    def __init__(self, result, n_chunks=50):
        self.result, self.store, self.ready = result, [None] * n_chunks, True

    async def retrieve(self, query):
        return self.result
