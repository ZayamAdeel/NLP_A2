"""
Conversation Manager: session handling, history management, prompt
orchestration, and turn-taking logic. This is the "brain" that sits between
the WebSocket layer and the LLM engine.

No tools, agents, or RAG are used anywhere in this file -- only in-memory
Python state and prompt construction.
"""
import time
import uuid
from dataclasses import dataclass, field

from app.config import (
    CHARS_PER_TOKEN_ESTIMATE,
    MAX_PROMPT_TOKENS,
    MAX_TURNS_IN_WINDOW,
    SUMMARY_TRIGGER_TURNS,
)
from app.domain import (
    BASE_SYSTEM_PROMPT,
    GREETING_MESSAGE,
    INTENT_KEYWORDS,
    OFF_TOPIC_KEYWORDS,
    OFF_TOPIC_REDIRECT,
    STAGE_CLOSING,
    STAGE_CONFIRMATION,
    STAGE_GREETING,
    STAGE_GUIDANCE,
    STAGE_INFO_GATHERING,
    STAGE_INTENT_ID,
    STAGE_RESOLUTION,
)
from app.llm_engine import OllamaEngine

CLOSING_TRIGGERS = ["bye", "goodbye", "thank you, that's all", "thats all", "no that's all",
                     "nothing else", "that's everything", "thanks that's all"]
CONFIRM_POSITIVE = ["yes", "yep", "yeah", "sure", "please", "go ahead"]
CONFIRM_NEGATIVE = ["no", "nope", "not really", "that's all", "nothing else", "no thanks"]


def _estimate_tokens(text: str) -> int:
    return max(1, len(text) // CHARS_PER_TOKEN_ESTIMATE)


def _detect_intent(message: str) -> str | None:
    lower = message.lower()
    scores = {intent: 0 for intent in INTENT_KEYWORDS}
    for intent, keywords in INTENT_KEYWORDS.items():
        for kw in keywords:
            if kw in lower:
                scores[intent] += 1
    best = max(scores, key=scores.get)
    return best if scores[best] > 0 else None


def is_off_topic(message: str) -> bool:
    """Fast, deterministic first-pass filter. The system prompt handles
    subtler cases the model itself must catch."""
    lower = message.lower()
    return any(kw in lower for kw in OFF_TOPIC_KEYWORDS)


@dataclass
class Turn:
    role: str  # "user" | "assistant"
    content: str
    timestamp: float = field(default_factory=time.time)


@dataclass
class Session:
    session_id: str
    stage: str = STAGE_GREETING
    active_intent: str | None = None
    history: list[Turn] = field(default_factory=list)
    rolling_summary: str = ""
    created_at: float = field(default_factory=time.time)
    last_active: float = field(default_factory=time.time)
    turn_count: int = 0

    def touch(self):
        self.last_active = time.time()


class ConversationManager:
    """
    Owns all sessions and drives the domain-specific conversation flow:
    greeting -> intent identification -> info gathering -> resolution ->
    confirmation -> closing, with support for the user changing topic at
    any point (which resets the active_intent and drops back into
    info_gathering for the new topic without losing the message history).
    """

    def __init__(self, engine: OllamaEngine | None = None):
        self.sessions: dict[str, Session] = {}
        self.engine = engine or OllamaEngine()

    # ---- Session lifecycle --------------------------------------------------
    def create_session(self) -> Session:
        session_id = str(uuid.uuid4())
        session = Session(session_id=session_id)
        self.sessions[session_id] = session
        return session

    def get_session(self, session_id: str) -> Session | None:
        return self.sessions.get(session_id)

    def reset_session(self, session_id: str) -> Session:
        self.sessions.pop(session_id, None)
        new_session = Session(session_id=session_id)
        self.sessions[session_id] = new_session
        return new_session

    def end_session(self, session_id: str) -> None:
        self.sessions.pop(session_id, None)

    # ---- Turn-taking / stage logic -------------------------------------------
    def _advance_stage(self, session: Session, user_message: str):
        lower = user_message.lower().strip()

        detected_intent = _detect_intent(user_message)

        # Topic switch mid-conversation: user asks about a different supported
        # intent than the one currently being handled. We keep the history
        # (for context/faithfulness) but reset the slot-filling stage so the
        # assistant gathers whatever info the NEW topic needs, rather than
        # continuing to press for the old topic's info.
        if (
            detected_intent
            and session.active_intent
            and detected_intent != session.active_intent
            and session.stage in (STAGE_INFO_GATHERING, STAGE_RESOLUTION, STAGE_CONFIRMATION)
        ):
            session.active_intent = detected_intent
            session.stage = STAGE_INFO_GATHERING
            return

        if session.stage == STAGE_GREETING:
            session.active_intent = detected_intent
            session.stage = STAGE_INTENT_ID if not detected_intent else STAGE_INFO_GATHERING
            return

        if session.stage == STAGE_INTENT_ID:
            session.active_intent = detected_intent or session.active_intent
            session.stage = STAGE_INFO_GATHERING
            return

        if session.stage == STAGE_INFO_GATHERING:
            session.stage = STAGE_RESOLUTION
            return

        if session.stage == STAGE_RESOLUTION:
            session.stage = STAGE_CONFIRMATION
            return

        if session.stage == STAGE_CONFIRMATION:
            if any(t in lower for t in CLOSING_TRIGGERS) or any(
                lower == n for n in CONFIRM_NEGATIVE
            ):
                session.stage = STAGE_CLOSING
            elif detected_intent and detected_intent != session.active_intent:
                session.active_intent = detected_intent
                session.stage = STAGE_INFO_GATHERING
            else:
                # Customer has a follow-up on the same topic.
                session.stage = STAGE_RESOLUTION
            return

        if session.stage == STAGE_CLOSING:
            # A closed conversation can always reopen if the user keeps talking.
            if detected_intent:
                session.active_intent = detected_intent
                session.stage = STAGE_INFO_GATHERING
            else:
                session.stage = STAGE_GREETING
            return

    # ---- Memory management (Phase II write-up lives in README, this is the
    # implementation of that scheme) ------------------------------------------
    async def _build_messages(self, session: Session) -> list[dict]:
        stage_hint = STAGE_GUIDANCE.get(session.stage, "")
        system_content = (
            BASE_SYSTEM_PROMPT
            + f"\n\nCurrent conversation stage: {session.stage}. Guidance: {stage_hint}"
        )
        if session.active_intent:
            system_content += f"\nCurrent topic the customer is asking about: {session.active_intent}."

        messages = [{"role": "system", "content": system_content}]

        if session.rolling_summary:
            messages.append(
                {
                    "role": "system",
                    "content": f"Summary of earlier conversation so far: {session.rolling_summary}",
                }
            )

        # Sliding window: only the most recent MAX_TURNS_IN_WINDOW turn-pairs
        # go in verbatim, so prompt length stays roughly constant regardless
        # of how long the session has been running.
        window = session.history[-(MAX_TURNS_IN_WINDOW * 2):]
        for turn in window:
            messages.append({"role": turn.role, "content": turn.content})

        # Defensive final trim purely on estimated size, in case a single
        # message in the window is unusually long.
        total_tokens = sum(_estimate_tokens(m["content"]) for m in messages)
        while total_tokens > MAX_PROMPT_TOKENS and len(messages) > 2:
            # Drop the oldest non-system message.
            for i, m in enumerate(messages):
                if m["role"] != "system":
                    removed = messages.pop(i)
                    total_tokens -= _estimate_tokens(removed["content"])
                    break
            else:
                break

        return messages

    async def _maybe_summarize(self, session: Session):
        """
        When the raw history grows past the window, fold the turns that are
        about to fall out of the window into a running summary using a
        single extra call to the SAME local model (plain prompting -- not an
        external tool/agent/RAG). This keeps memory usage/prompt size flat
        for arbitrarily long sessions while still preserving gist-level
        context from earlier in the conversation.
        """
        if len(session.history) < SUMMARY_TRIGGER_TURNS * 2:
            return

        cutoff = len(session.history) - (MAX_TURNS_IN_WINDOW * 2)
        to_fold = session.history[:cutoff]
        if not to_fold:
            return

        transcript = "\n".join(f"{t.role}: {t.content}" for t in to_fold)
        summarizer_messages = [
            {
                "role": "system",
                "content": (
                    "Summarize the following customer support exchange in 2-3 sentences, "
                    "keeping any concrete facts (order numbers, product names, decisions "
                    "made). Be terse."
                ),
            },
            {"role": "user", "content": transcript},
        ]
        try:
            summary = await self.engine.chat_once(summarizer_messages)
            if session.rolling_summary:
                session.rolling_summary = (session.rolling_summary + " " + summary).strip()
            else:
                session.rolling_summary = summary.strip()
        except Exception:
            # Summarization is a best-effort optimization; if it fails we
            # simply keep the window as-is rather than crashing the turn.
            pass
        finally:
            session.history = session.history[cutoff:]

    # ---- Public API used by the WebSocket route ------------------------------
    async def handle_user_message(self, session: Session, user_message: str):
        """
        Async generator. Yields (event_type, payload) tuples:
          ("stage", stage_name)
          ("token", text_chunk)
          ("done", metrics_dict)
          ("blocked", redirect_message)   -- off-topic, no LLM call made
        """
        session.touch()
        session.turn_count += 1

        if is_off_topic(user_message):
            session.history.append(Turn("user", user_message))
            session.history.append(Turn("assistant", OFF_TOPIC_REDIRECT))
            yield ("stage", session.stage)
            yield ("blocked", OFF_TOPIC_REDIRECT)
            return

        self._advance_stage(session, user_message)
        session.history.append(Turn("user", user_message))

        await self._maybe_summarize(session)

        messages = await self._build_messages(session)

        yield ("stage", session.stage)

        full_response = ""
        final_metrics = None
        async for token, metrics in self.engine.stream_chat(messages):
            full_response += token
            final_metrics = metrics
            yield ("token", token)

        session.history.append(Turn("assistant", full_response))
        yield (
            "done",
            {
                "time_to_first_token": final_metrics.time_to_first_token if final_metrics else None,
                "total_time": final_metrics.total_time if final_metrics else None,
                "tokens_per_second": final_metrics.tokens_per_second if final_metrics else None,
                "stage": session.stage,
            },
        )
