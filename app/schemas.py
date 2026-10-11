from typing import Literal, Optional

from pydantic import BaseModel, Field


class NewSessionResponse(BaseModel):
    session_id: str
    greeting: str


class SessionInfoResponse(BaseModel):
    session_id: str
    stage: str
    active_intent: Optional[str]
    turn_count: int
    history_length: int


class HealthResponse(BaseModel):
    status: str
    ollama_reachable: bool
    model: str


# ---- WebSocket message contracts -------------------------------------------
# Client -> Server
class ClientChatMessage(BaseModel):
    type: Literal["message"] = "message"
    session_id: str
    text: str = Field(min_length=1, max_length=4000)


# Server -> Client (discriminated by "type")
class ServerStageEvent(BaseModel):
    type: Literal["stage"] = "stage"
    stage: str


class ServerTokenEvent(BaseModel):
    type: Literal["token"] = "token"
    text: str


class ServerDoneEvent(BaseModel):
    type: Literal["done"] = "done"
    time_to_first_token: Optional[float]
    total_time: Optional[float]
    tokens_per_second: Optional[float]
    stage: str


class ServerBlockedEvent(BaseModel):
    type: Literal["blocked"] = "blocked"
    text: str


class ServerErrorEvent(BaseModel):
    type: Literal["error"] = "error"
    message: str
