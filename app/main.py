"""
FastAPI backend for the E-Commerce Order Support Assistant.

Endpoints:
  GET  /api/health          -- health check (also pings Ollama)
  POST /api/session         -- create a new chat session
  GET  /api/session/{id}    -- inspect session state
  DELETE /api/session/{id}  -- end a session
  WS   /ws/chat             -- real-time streaming chat

The WebSocket protocol is a small JSON envelope in both directions; see
app/schemas.py for the exact shapes and README.md for examples.
"""
import json
import logging

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import ValidationError

from app.config import OLLAMA_MODEL
from app.conversation_manager import ConversationManager
from app.domain import GREETING_MESSAGE
from app.llm_engine import LLMEngineError, OllamaEngine
from app.schemas import (
    ClientChatMessage,
    HealthResponse,
    NewSessionResponse,
    SessionInfoResponse,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("ecommerce-assistant")

app = FastAPI(title="E-Commerce Order Support Assistant", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # local/demo project; tighten for real deployment
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

engine = OllamaEngine()
manager = ConversationManager(engine=engine)


@app.get("/api/health", response_model=HealthResponse)
async def health():
    reachable = await engine.health_check()
    return HealthResponse(
        status="ok" if reachable else "degraded",
        ollama_reachable=reachable,
        model=OLLAMA_MODEL,
    )


@app.post("/api/session", response_model=NewSessionResponse)
async def create_session():
    session = manager.create_session()
    return NewSessionResponse(session_id=session.session_id, greeting=GREETING_MESSAGE)


@app.get("/api/session/{session_id}", response_model=SessionInfoResponse)
async def get_session(session_id: str):
    session = manager.get_session(session_id)
    if session is None:
        return SessionInfoResponse(
            session_id=session_id, stage="unknown", active_intent=None,
            turn_count=0, history_length=0,
        )
    return SessionInfoResponse(
        session_id=session.session_id,
        stage=session.stage,
        active_intent=session.active_intent,
        turn_count=session.turn_count,
        history_length=len(session.history),
    )


@app.delete("/api/session/{session_id}")
async def delete_session(session_id: str):
    manager.end_session(session_id)
    return {"status": "deleted", "session_id": session_id}


@app.websocket("/ws/chat")
async def ws_chat(websocket: WebSocket):
    """
    Real-time streaming chat endpoint.

    Client sends: {"type": "message", "session_id": "...", "text": "..."}
    Server streams back a sequence of JSON events per turn:
        {"type": "stage", "stage": "..."}
        {"type": "token", "text": "..."}          (repeated, streamed word/piece by piece)
        {"type": "done", "time_to_first_token": ..., "total_time": ..., "tokens_per_second": ..., "stage": "..."}
      or, if the message is off-topic:
        {"type": "blocked", "text": "..."}
      or, on any error:
        {"type": "error", "message": "..."}

    Each connection is handled by its own asyncio task, and the underlying
    LLM call is a fully async streaming HTTP request, so one user's slow
    generation never blocks another connected user's turn.
    """
    await websocket.accept()
    logger.info("WebSocket connection accepted")

    try:
        while True:
            try:
                raw = await websocket.receive_text()
            except WebSocketDisconnect:
                logger.info("Client disconnected")
                break

            # --- Robust error handling: malformed JSON / schema -------------
            try:
                payload = json.loads(raw)
                client_msg = ClientChatMessage(**payload)
            except (json.JSONDecodeError, ValidationError) as e:
                await _safe_send(websocket, {
                    "type": "error",
                    "message": f"Malformed request: {e}",
                })
                continue

            session = manager.get_session(client_msg.session_id)
            if session is None:
                await _safe_send(websocket, {
                    "type": "error",
                    "message": "Unknown or expired session_id. Create a new session via "
                               "POST /api/session or the 'New chat' button.",
                })
                continue

            # --- Run one full turn, streaming events back -------------------
            try:
                async for event_type, payload in manager.handle_user_message(
                    session, client_msg.text
                ):
                    if event_type == "stage":
                        await _safe_send(websocket, {"type": "stage", "stage": payload})
                    elif event_type == "token":
                        await _safe_send(websocket, {"type": "token", "text": payload})
                    elif event_type == "blocked":
                        await _safe_send(websocket, {"type": "blocked", "text": payload})
                    elif event_type == "done":
                        await _safe_send(websocket, {"type": "done", **payload})
            except LLMEngineError as e:
                logger.warning("LLM engine error: %s", e)
                await _safe_send(websocket, {
                    "type": "error",
                    "message": f"The local model backend hit an error: {e}",
                })
            except Exception as e:  # noqa: BLE001 - last-resort guard, must never crash the socket
                logger.exception("Unexpected error handling turn")
                await _safe_send(websocket, {
                    "type": "error",
                    "message": "An unexpected server error occurred while generating a response. "
                               "Please try again.",
                })

    except WebSocketDisconnect:
        logger.info("Client disconnected (outer)")
    except Exception:
        logger.exception("Fatal error in websocket handler")


async def _safe_send(websocket: WebSocket, data: dict):
    """Send JSON, swallowing errors from a socket that already closed mid-stream
    (e.g. client disconnected mid-response) so we don't raise inside the loop."""
    try:
        await websocket.send_text(json.dumps(data))
    except Exception:
        logger.info("Could not send to websocket (likely disconnected mid-stream)")


# Serve the simple chat frontend at /
app.mount("/", StaticFiles(directory="static", html=True), name="static")
