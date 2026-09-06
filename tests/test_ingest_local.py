import json
import sqlite3

from jepa_chat.ingest.codex import parse_codex, parse_rollout
from jepa_chat.ingest.cursor import parse_cursor_cli, parse_cursor_db


def _rollout(tmp_path, name="rollout-2026-07-01T00-00-00-abc.jsonl", subagent=False):
    meta = {"id": "abc", "timestamp": "2026-07-01T00:00:00Z", "cwd": "/x/proj"}
    if subagent:
        meta["source"] = {"subagent": {"other": "guardian"}}
    recs = [
        {"type": "session_meta", "timestamp": "2026-07-01T00:00:00Z", "payload": meta},
        {
            "type": "response_item",
            "timestamp": "2026-07-01T00:00:01Z",
            "payload": {"type": "message", "role": "developer", "content": [{"type": "input_text", "text": "sys"}]},
        },
        {
            "type": "response_item",
            "timestamp": "2026-07-01T00:00:02Z",
            "payload": {
                "type": "message",
                "role": "user",
                "content": [
                    {"type": "input_text", "text": "<environment_context>cwd</environment_context>"},
                    {"type": "input_text", "text": "design the onboarding flow"},
                ],
            },
        },
        {"type": "response_item", "payload": {"type": "reasoning", "summary": []}},
        {"type": "event_msg", "payload": {"type": "agent_message", "message": "dup"}},
        {
            "type": "response_item",
            "timestamp": "2026-07-01T00:00:03Z",
            "payload": {
                "type": "message",
                "role": "assistant",
                "content": [{"type": "output_text", "text": "three screens"}],
            },
        },
        {"type": "response_item", "payload": {"type": "function_call", "name": "shell"}},
        {
            "type": "response_item",
            "timestamp": "2026-07-01T00:00:04Z",
            "payload": {"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": "done"}]},
        },
    ]
    d = tmp_path / "2026" / "07" / "01"
    d.mkdir(parents=True, exist_ok=True)
    f = d / name
    f.write_text("\n".join(json.dumps(r) for r in recs))
    return f


def test_codex_rollout(tmp_path):
    f = _rollout(tmp_path)
    s = parse_rollout(f)
    assert s.id == "codex:abc" and s.title == "proj" and s.source == "codex"
    assert [(t.role, t.text) for t in s.turns] == [
        ("user", "design the onboarding flow"),
        ("assistant", "three screens\ndone"),
    ]
    assert s.turns[0].mouth == "codex" and s.created_ts is not None
    _rollout(tmp_path, name="rollout-2026-07-01T00-00-00-sub.jsonl", subagent=True)
    assert len(list(parse_codex(tmp_path))) == 1
    assert len(list(parse_codex(tmp_path, include_subagents=True))) == 2


def test_cursor_db(tmp_path):
    db = tmp_path / "state.vscdb"
    con = sqlite3.connect(db)
    con.execute("CREATE TABLE cursorDiskKV(key TEXT PRIMARY KEY, value BLOB)")
    comp = {
        "composerId": "c1",
        "name": "Product Q&A",
        "createdAt": 1700000000000,
        "fullConversationHeadersOnly": [
            {"bubbleId": "b1", "type": 1},
            {"bubbleId": "b2", "type": 2},
            {"bubbleId": "b3", "type": 2},
        ],
    }
    con.execute("INSERT INTO cursorDiskKV VALUES(?,?)", ("composerData:c1", json.dumps(comp)))
    con.execute(
        "INSERT INTO cursorDiskKV VALUES(?,?)",
        ("bubbleId:c1:b1", json.dumps({"type": 1, "text": "why this layout?", "createdAt": 1700000001000})),
    )
    con.execute(
        "INSERT INTO cursorDiskKV VALUES(?,?)",
        ("bubbleId:c1:b2", json.dumps({"type": 2, "text": "because", "createdAt": 1700000002000})),
    )
    con.execute("INSERT INTO cursorDiskKV VALUES(?,?)", ("bubbleId:c1:b3", json.dumps({"type": 2, "text": ""})))
    con.commit()
    con.close()
    (s,) = list(parse_cursor_db(db))
    assert s.id == "cursor:c1" and s.title == "Product Q&A" and s.created_ts == 1700000000.0
    assert [(t.role, t.text) for t in s.turns] == [("user", "why this layout?"), ("assistant", "because")]


def test_cursor_cli(tmp_path):
    d = tmp_path / "proj" / "agent-transcripts" / "x"
    d.mkdir(parents=True)
    recs = [
        {"type": "user", "message": {"role": "user", "content": "hi"}},
        {"type": "assistant", "message": {"role": "assistant", "content": [{"type": "text", "text": "hello"}]}},
    ]
    (d / "t.jsonl").write_text("\n".join(json.dumps(r) for r in recs))
    (s,) = list(parse_cursor_cli(tmp_path))
    assert s.source == "cursor-cli" and s.turns[0].mouth == "cursor" and len(s.turns) == 2
