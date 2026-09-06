"""Parser for the official ChatGPT data export (Settings -> Data Controls -> Export).

The export contains `conversations.json` (or numbered `conversations-N.json` files),
each a list of conversations with a `mapping` tree of message nodes. We follow the
`current_node` chain back to the root so branches that were regenerated are dropped.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

from jepa_chat.data.turns import Session, Turn
from jepa_chat.ingest._util import iter_json_files

_ROLE_MAP = {"user": "user", "assistant": "assistant"}


def _message_text(msg: dict) -> str | None:
    content = msg.get("content") or {}
    if content.get("content_type") not in ("text", "multimodal_text"):
        return None
    parts = [p for p in content.get("parts") or [] if isinstance(p, str)]
    text = "\n".join(p for p in parts if p.strip()).strip()
    return text or None


def _linear_path(conv: dict) -> list[dict]:
    mapping = conv.get("mapping") or {}
    node_id = conv.get("current_node")
    chain: list[dict] = []
    seen: set[str] = set()
    while node_id and node_id in mapping and node_id not in seen:
        seen.add(node_id)
        node = mapping[node_id]
        chain.append(node)
        node_id = node.get("parent")
    chain.reverse()
    return chain


def parse_conversation(conv: dict) -> Session | None:
    turns: list[Turn] = []
    for node in _linear_path(conv):
        msg = node.get("message")
        if not msg:
            continue
        role = _ROLE_MAP.get((msg.get("author") or {}).get("role"))
        if role is None:
            continue
        if (msg.get("metadata") or {}).get("is_visually_hidden_from_conversation"):
            continue
        text = _message_text(msg)
        if not text:
            continue
        turns.append(Turn(role=role, text=text, ts=msg.get("create_time"), mouth="chatgpt"))
    if not turns:
        return None
    return Session(
        id=f"chatgpt:{conv.get('conversation_id') or conv.get('id')}",
        title=conv.get("title") or "",
        source="chatgpt",
        created_ts=conv.get("create_time"),
        turns=turns,
    )


def parse_chatgpt(path: str | Path) -> Iterator[Session]:
    for _name, data in iter_json_files(path, "conversations.json"):
        if not isinstance(data, list):
            continue
        for conv in data:
            session = parse_conversation(conv)
            if session is not None:
                yield session
    # numbered files in large exports
    for _name, data in iter_json_files(path, ".json"):
        if _name.endswith("conversations.json"):
            continue
        if isinstance(data, list) and data and isinstance(data[0], dict) and "mapping" in data[0]:
            for conv in data:
                session = parse_conversation(conv)
                if session is not None:
                    yield session
