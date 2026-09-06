"""Parsers for Cursor.

1. The IDE chat store: ~/Library/Application Support/Cursor/User/globalStorage/state.vscdb,
   table cursorDiskKV with composerData:<id> headers and bubbleId:<composer>:<bubble> messages
   (bubble type 1 = user, 2 = assistant).
2. Cursor CLI agent transcripts: ~/.cursor/projects/*/agent-transcripts/**/*.jsonl, which share
   the Claude Code record shape (type user/assistant, message.content).
"""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Iterator
from pathlib import Path

from jepa_chat.data.turns import Session, Turn
from jepa_chat.ingest.claude_code import parse_transcript

DEFAULT_DB = Path("~/Library/Application Support/Cursor/User/globalStorage/state.vscdb").expanduser()
DEFAULT_CLI_ROOT = Path("~/.cursor/projects").expanduser()
_ROLE = {1: "user", 2: "assistant"}


def parse_cursor_db(db_path: str | Path | None = None) -> Iterator[Session]:
    db = Path(db_path).expanduser() if db_path else DEFAULT_DB
    if not db.exists():
        return
    con = sqlite3.connect(f"file:{db}?mode=ro&immutable=1", uri=True)
    try:
        rows = con.execute("SELECT key, value FROM cursorDiskKV WHERE key LIKE 'composerData:%'").fetchall()
        for key, value in rows:
            try:
                comp = json.loads(value)
            except (json.JSONDecodeError, TypeError):
                continue
            cid = comp.get("composerId") or key.split(":", 1)[1]
            headers = comp.get("fullConversationHeadersOnly") or comp.get("conversation") or []
            turns: list[Turn] = []
            for h in headers:
                bid, btype = h.get("bubbleId"), h.get("type")
                role = _ROLE.get(btype)
                if not bid or role is None:
                    continue
                row = con.execute("SELECT value FROM cursorDiskKV WHERE key=?", (f"bubbleId:{cid}:{bid}",)).fetchone()
                if not row:
                    continue
                try:
                    bubble = json.loads(row[0])
                except (json.JSONDecodeError, TypeError):
                    continue
                text = (bubble.get("text") or "").strip()
                if not text:
                    continue
                ts = bubble.get("createdAt")
                ts = ts / 1000.0 if isinstance(ts, int | float) and ts > 1e11 else None
                if turns and turns[-1].role == role:
                    turns[-1].text += "\n" + text
                    continue
                turns.append(Turn(role=role, text=text, ts=ts, mouth="cursor"))
            if not turns:
                continue
            created = comp.get("createdAt")
            created = created / 1000.0 if isinstance(created, int | float) and created > 1e11 else None
            yield Session(
                id=f"cursor:{cid}", title=comp.get("name") or "", source="cursor", created_ts=created, turns=turns
            )
    finally:
        con.close()


def parse_cursor_cli(root: str | Path | None = None) -> Iterator[Session]:
    root = Path(root).expanduser() if root else DEFAULT_CLI_ROOT
    if not root.exists():
        return
    for f in sorted(root.glob("*/agent-transcripts/**/*.jsonl")):
        s = parse_transcript(f)
        if s is not None:
            s.id = f"cursor-cli:{f.stem}"
            s.source = "cursor-cli"
            for t in s.turns:
                t.mouth = "cursor"
            yield s


def parse_cursor(db_path: str | Path | None = None, cli_root: str | Path | None = None) -> Iterator[Session]:
    yield from parse_cursor_db(db_path)
    yield from parse_cursor_cli(cli_root)
