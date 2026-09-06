import json
import zipfile

from jepa_chat.data.turns import Session, Turn, read_jsonl, truncate_turn, write_jsonl
from jepa_chat.ingest.chatgpt import parse_chatgpt, parse_conversation
from jepa_chat.ingest.claude import parse_claude
from jepa_chat.ingest.claude_code import parse_transcript
from jepa_chat.ingest.markdown import parse_markdown_text


def _chatgpt_conv():
    def node(i, role, text, parent, hidden=False, ctype="text"):
        return {
            "id": i,
            "parent": parent,
            "children": [],
            "message": {
                "author": {"role": role},
                "content": {"content_type": ctype, "parts": [text]},
                "create_time": 1.0,
                "metadata": {"is_visually_hidden_from_conversation": hidden},
            },
        }

    return {
        "conversation_id": "c1",
        "title": "branchy",
        "create_time": 0.0,
        "current_node": "a2",
        "mapping": {
            "root": {"id": "root", "parent": None, "children": ["s"], "message": None},
            "s": node("s", "system", "sys", "root", hidden=True),
            "u1": node("u1", "user", "hello", "s"),
            "a1": node("a1", "assistant", "old branch", "u1"),
            "a2": node("a2", "assistant", "new branch", "u1"),
        },
    }


def test_chatgpt_follows_current_node_and_drops_hidden():
    s = parse_conversation(_chatgpt_conv())
    assert [t.role for t in s.turns] == ["user", "assistant"]
    assert s.turns[1].text == "new branch"
    assert s.source == "chatgpt"


def test_chatgpt_zip(tmp_path):
    z = tmp_path / "export.zip"
    with zipfile.ZipFile(z, "w") as zf:
        zf.writestr("conversations.json", json.dumps([_chatgpt_conv()]))
    sessions = list(parse_chatgpt(z))
    assert len(sessions) == 1


def test_claude_export(tmp_path):
    data = [
        {
            "uuid": "x",
            "name": "t",
            "created_at": "2026-01-01T00:00:00Z",
            "chat_messages": [
                {"sender": "human", "text": "hi", "created_at": "2026-01-01T00:00:00Z"},
                {"sender": "assistant", "text": "", "content": [{"type": "text", "text": "yo"}]},
                {"sender": "assistant", "text": ""},
            ],
        }
    ]
    f = tmp_path / "conversations.json"
    f.write_text(json.dumps(data))
    (s,) = list(parse_claude(f))
    assert [t.text for t in s.turns] == ["hi", "yo"]
    assert s.turns[0].ts is not None


def test_claude_code_transcript(tmp_path):
    recs = [
        {"type": "custom-title", "customTitle": "My chat"},
        {
            "type": "user",
            "timestamp": "2026-01-01T00:00:00Z",
            "message": {"role": "user", "content": "do x"},
        },
        {
            "type": "assistant",
            "message": {"role": "assistant", "content": [{"type": "thinking", "thinking": "hm"}]},
        },
        {
            "type": "assistant",
            "message": {"role": "assistant", "content": [{"type": "text", "text": "ok"}]},
        },
        {
            "type": "assistant",
            "message": {"role": "assistant", "content": [{"type": "tool_use", "name": "Bash"}]},
        },
        {
            "type": "user",
            "message": {"role": "user", "content": [{"type": "tool_result", "content": "out"}]},
        },
        {
            "type": "assistant",
            "message": {"role": "assistant", "content": [{"type": "text", "text": "done"}]},
        },
        {"type": "user", "isSidechain": True, "message": {"role": "user", "content": "subagent"}},
    ]
    d = tmp_path / "-Users-me-proj"
    d.mkdir()
    f = d / "abc.jsonl"
    f.write_text("\n".join(json.dumps(r) for r in recs))
    s = parse_transcript(f)
    assert s.title == "My chat"
    assert [(t.role, t.text) for t in s.turns] == [("user", "do x"), ("assistant", "ok\ndone")]


def test_markdown_transcript():
    text = "# T\n\n**User:** hi there\n\n**Assistant:**\n\nhello\nmore\n\n**User:** bye\n"
    s = parse_markdown_text(text, "md:t")
    assert [t.role for t in s.turns] == ["user", "assistant", "user"]
    assert s.turns[1].text == "hello\nmore"


def test_jsonl_roundtrip_and_truncate(tmp_path):
    s = Session(id="a", title="t", source="x", created_ts=1.0, turns=[Turn("user", "hi", 1.0, "m")])
    out = tmp_path / "t.jsonl"
    assert write_jsonl([s], out) == 1
    (back,) = list(read_jsonl(out))
    assert back == s
    long = "a" * 100 + "b" * 100
    t = truncate_turn(long, 10, 5)
    assert t.startswith("a" * 10) and t.endswith("b" * 5)
