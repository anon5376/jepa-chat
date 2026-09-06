"""Parser for the Claude.ai data export (Settings -> Privacy -> Export data).

The export's `conversations.json` is a list of conversations, each with
`chat_messages`, where `sender` is "human" or "assistant".
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

from jepa_chat.data.turns import Session, Turn
from jepa_chat.ingest._util import iter_json_files, parse_iso_ts

_ROLE_MAP = {"human": "user", "assistant": "assistant"}


def _message_text(msg: dict) -> str | None:
    text = (msg.get("text") or "").strip()
    if not text:
        blocks = msg.get("content") or []
        text = "\n".join(
            (b.get("text") or "").strip()
            for b in blocks
            if isinstance(b, dict) and b.get("type") == "text"
        ).strip()
    return text or None


def parse_conversation(conv: dict) -> Session | None:
    turns: list[Turn] = []
    for msg in conv.get("chat_messages") or []:
        role = _ROLE_MAP.get(msg.get("sender"))
        if role is None:
            continue
        text = _message_text(msg)
        if not text:
            continue
        turns.append(
            Turn(role=role, text=text, ts=parse_iso_ts(msg.get("created_at")), mouth="claude")
        )
    if not turns:
        return None
    return Session(
        id=f"claude:{conv.get('uuid')}",
        title=conv.get("name") or "",
        source="claude",
        created_ts=parse_iso_ts(conv.get("created_at")),
        turns=turns,
    )


def parse_claude(path: str | Path) -> Iterator[Session]:
    for _name, data in iter_json_files(path, "conversations.json"):
        if not isinstance(data, list):
            continue
        for conv in data:
            session = parse_conversation(conv)
            if session is not None:
                yield session
