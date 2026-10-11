"""
Thin async wrapper around Ollama's local REST API.

We talk to Ollama's native /api/chat endpoint (not the LangChain/tool-calling
layer) so it's crystal clear no tools/agents/RAG are involved -- just a
messages list in, a token stream out.
"""
import json
import time
from dataclasses import dataclass
from typing import AsyncIterator, Optional

import httpx

from app.config import GENERATION_OPTIONS, OLLAMA_HOST, OLLAMA_MODEL, OLLAMA_TIMEOUT


class LLMEngineError(Exception):
    """Raised when the local model backend fails or is unreachable."""


@dataclass
class GenerationMetrics:
    time_to_first_token: Optional[float] = None
    total_time: Optional[float] = None
    output_tokens_estimate: int = 0

    @property
    def tokens_per_second(self) -> Optional[float]:
        if self.total_time and self.total_time > 0:
            return round(self.output_tokens_estimate / self.total_time, 2)
        return None


class OllamaEngine:
    """
    Streaming chat-completion client for a local Ollama server.

    Usage:
        engine = OllamaEngine()
        async for token, metrics in engine.stream_chat(messages):
            ...
    """

    def __init__(self, model: str = OLLAMA_MODEL, host: str = OLLAMA_HOST):
        self.model = model
        self.host = host.rstrip("/")

    async def health_check(self) -> bool:
        try:
            async with httpx.AsyncClient(timeout=5) as client:
                resp = await client.get(f"{self.host}/api/tags")
                return resp.status_code == 200
        except httpx.HTTPError:
            return False

    async def stream_chat(
        self, messages: list[dict], options: dict | None = None
    ) -> AsyncIterator[tuple[str, GenerationMetrics]]:
        """
        Streams response tokens one at a time.

        Yields (token_text, metrics) tuples. `metrics` is progressively filled
        in and the final yielded metrics object has time_to_first_token,
        total_time, and a rough tokens/sec figure -- used both for the
        live UI and for the Phase VI latency benchmarking.
        """
        payload = {
            "model": self.model,
            "messages": messages,
            "stream": True,
            "options": {**GENERATION_OPTIONS, **(options or {})},
        }

        metrics = GenerationMetrics()
        start = time.perf_counter()
        first_token_seen = False
        token_count = 0

        try:
            async with httpx.AsyncClient(timeout=OLLAMA_TIMEOUT) as client:
                async with client.stream(
                    "POST", f"{self.host}/api/chat", json=payload
                ) as response:
                    if response.status_code != 200:
                        body = await response.aread()
                        raise LLMEngineError(
                            f"Ollama returned status {response.status_code}: {body[:300]!r}"
                        )
                    async for line in response.aiter_lines():
                        if not line.strip():
                            continue
                        try:
                            chunk = json.loads(line)
                        except json.JSONDecodeError:
                            continue

                        if chunk.get("error"):
                            raise LLMEngineError(chunk["error"])

                        content = chunk.get("message", {}).get("content", "")
                        if content:
                            if not first_token_seen:
                                metrics.time_to_first_token = round(
                                    time.perf_counter() - start, 4
                                )
                                first_token_seen = True
                            token_count += 1
                            metrics.output_tokens_estimate = token_count
                            metrics.total_time = round(time.perf_counter() - start, 4)
                            yield content, metrics

                        if chunk.get("done"):
                            metrics.total_time = round(time.perf_counter() - start, 4)
                            # Prefer Ollama's own eval_count when available -- more
                            # accurate than our "one chunk ~= one token" estimate.
                            if chunk.get("eval_count"):
                                metrics.output_tokens_estimate = chunk["eval_count"]
                            if chunk.get("eval_duration"):
                                metrics.total_time = round(
                                    chunk["eval_duration"] / 1e9, 4
                                )
                            break
        except httpx.ConnectError as e:
            raise LLMEngineError(
                f"Could not connect to Ollama at {self.host}. Is `ollama serve` running "
                f"and is the model pulled (`ollama pull {self.model}`)?"
            ) from e
        except httpx.HTTPError as e:
            raise LLMEngineError(f"HTTP error talking to Ollama: {e}") from e

    async def chat_once(self, messages: list[dict], options: dict | None = None) -> str:
        """Non-streaming convenience wrapper (used for internal summarization calls)."""
        text = ""
        async for token, _ in self.stream_chat(messages, options=options):
            text += token
        return text
