import json

import pytest
from starlette.testclient import TestClient

from jepa_chat.core.app import CoreRuntime, create_app


@pytest.fixture
def client(runtime):
    cfg, memory, engine, bus, state, svc = runtime
    rt = CoreRuntime(cfg, memory, engine, bus, state, svc)
    with TestClient(create_app(rt), base_url="http://127.0.0.1") as c:
        yield c


def test_observe_state_events(client):
    r = client.post("/api/observe", json={"role": "user", "text": "ping"}, headers={"x-jepa-mouth": "cursor"})
    assert r.status_code == 200 and r.json()["mouth"] == "cursor"
    r = client.post("/api/observe", json={"role": "assistant", "text": "pong", "mouth": "cursor"})
    assert isinstance(r.json()["surprise"], float)
    st = client.get("/api/state").json()
    assert st["n_turns"] == 2 and st["training"]["paused"] is True
    assert client.get("/api/stats").json()["memory_turns"] == 2
    assert client.post("/api/recall", json={"query": "ping"}).json()["hits"]
    assert client.post("/api/session", json={"action": "list"}).json()["active"]
    assert client.post("/api/control", json={"action": "bogus"}).status_code == 400


def test_mcp_endpoint_mounted(client):
    r = client.post(
        "/mcp",
        json={
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": {
                "protocolVersion": "2025-06-18",
                "capabilities": {},
                "clientInfo": {"name": "t", "version": "0"},
            },
        },
        headers={"accept": "application/json, text/event-stream", "content-type": "application/json"},
    )
    assert r.status_code == 200
    body = r.text
    payload = (
        json.loads(body.split("data:", 1)[1].strip().splitlines()[0])
        if body.startswith("event:") or "data:" in body
        else r.json()
    )
    assert payload["result"]["serverInfo"]["name"] == "jepa"
