"""
API-level tests using FastAPI's TestClient. The real OllamaEngine is
monkeypatched with a fake so these tests run without a local model or
network access -- useful for CI and for grading without requiring Ollama
to be installed on the grader's machine, though the full system obviously
needs a real Ollama instance running for actual use.
"""
import json
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app import main as main_module
from app.conversation_manager import ConversationManager


class FakeMetrics:
    time_to_first_token = 0.05
    total_time = 0.3
    output_tokens_estimate = 4

    @property
    def tokens_per_second(self):
        return 13.3


class FakeEngine:
    async def health_check(self):
        return True

    async def stream_chat(self, messages, options=None):
        for word in ["Sure, ", "happy ", "to ", "help."]:
            yield word, FakeMetrics()

    async def chat_once(self, messages, options=None):
        return "summary"


@pytest.fixture(autouse=True)
def patch_engine(monkeypatch):
    fake = FakeEngine()
    monkeypatch.setattr(main_module, "engine", fake)
    monkeypatch.setattr(main_module, "manager", ConversationManager(engine=fake))
    yield


@pytest.fixture
def client():
    return TestClient(main_module.app)


def test_health_endpoint(client):
    resp = client.get("/api/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert body["ollama_reachable"] is True


def test_create_and_fetch_session(client):
    resp = client.post("/api/session")
    assert resp.status_code == 200
    data = resp.json()
    assert "session_id" in data
    assert "greeting" in data

    info = client.get(f"/api/session/{data['session_id']}")
    assert info.status_code == 200
    assert info.json()["stage"] == "greeting"


def test_delete_session(client):
    data = client.post("/api/session").json()
    resp = client.delete(f"/api/session/{data['session_id']}")
    assert resp.status_code == 200


def test_websocket_happy_path(client):
    session_id = client.post("/api/session").json()["session_id"]
    with client.websocket_connect("/ws/chat") as ws:
        ws.send_text(json.dumps({
            "type": "message", "session_id": session_id, "text": "Where is my order?",
        }))
        events = []
        for _ in range(8):
            msg = json.loads(ws.receive_text())
            events.append(msg)
            if msg["type"] == "done":
                break
        types = [e["type"] for e in events]
        assert "stage" in types
        assert "token" in types
        assert "done" in types


def test_websocket_malformed_json_returns_error_not_crash(client):
    with client.websocket_connect("/ws/chat") as ws:
        ws.send_text("{not valid json")
        msg = json.loads(ws.receive_text())
        assert msg["type"] == "error"
        assert "Malformed" in msg["message"]

        # Connection should still be alive for a subsequent valid message.
        session_id = client.post("/api/session").json()["session_id"]
        ws.send_text(json.dumps({
            "type": "message", "session_id": session_id, "text": "hi",
        }))
        msg2 = json.loads(ws.receive_text())
        assert msg2["type"] in ("stage", "error")


def test_websocket_unknown_session_returns_error(client):
    with client.websocket_connect("/ws/chat") as ws:
        ws.send_text(json.dumps({
            "type": "message", "session_id": "does-not-exist", "text": "hi",
        }))
        msg = json.loads(ws.receive_text())
        assert msg["type"] == "error"
        assert "Unknown" in msg["message"]


def test_websocket_off_topic_message_blocked(client):
    session_id = client.post("/api/session").json()["session_id"]
    with client.websocket_connect("/ws/chat") as ws:
        ws.send_text(json.dumps({
            "type": "message", "session_id": session_id, "text": "What's the weather today?",
        }))
        events = []
        for _ in range(4):
            msg = json.loads(ws.receive_text())
            events.append(msg)
            if msg["type"] == "blocked":
                break
        assert any(e["type"] == "blocked" for e in events)
