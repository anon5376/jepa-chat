"""Parser for local Claude Code session transcripts (~/.claude/projects/*/*.jsonl).

Each line is a record. We keep `user` records whose content is a string or text
blocks (tool results are skipped) and `assistant` records' text blocks (thinking and
tool calls are skipped). Sidechain (subagent) records are ignored.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path

from jepa_chat.data.turns import Session, Turn
from jepa_chat.ingest._util import parse_iso_ts

DEFAULT_ROOT = Path("~/.claude/projects").expanduser()


def _text_from_content(content: object) -> str | None:
    if isinstance(content, str):
        text = content.strip()
    elif isinstance(content, list):
        text = "\n".join(
            (b.get("text") or "").strip()
            for b in content
            if isinstance(b, dict) and b.get("type") == "text"
        ).strip()
    else:
        return None
    if not text or text.startswith("<") and text.rstrip().endswith(">") and "\n" not in text:
        return None  # pure tag payloads such as <command-name>...</command-name>
    return text


def parse_transcript(path: str | Path) -> Session | None:
    path = Path(path)
    turns: list[Turn] = []
    title = ""
    created: float | None = None
    with path.open(encoding="utf-8", errors="replace") as fh:
        for line in fh:
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            rtype = rec.get("type")
            if rtype == "custom-title":
                title = rec.get("customTitle") or title
                continue
            if rtype not in ("user", "assistant") or rec.get("isSidechain") or rec.get("isMeta"):
                continue
            text = _text_from_content((rec.get("message") or {}).get("content"))
            if not text:
                continue
            ts = parse_iso_ts(rec.get("timestamp"))
            created = created if created is not None else ts
            # merge consecutive assistant text records from one reply
            if turns and turns[-1].role == rtype == "assistant":
                turns[-1].text += "\n" + text
                continue
            turns.append(Turn(role=rtype, text=text, ts=ts, mouth="claude-code"))
    if not turns:
        return None
    project = path.parent.name.lstrip("-").replace("-", "/")
    return Session(
        id=f"claude-code:{path.stem}",
        title=title or project,
        source="claude-code",
        created_ts=created,
        turns=turns,
    )


def parse_claude_code(root: str | Path | None = None) -> Iterator[Session]:
    root = Path(root).expanduser() if root else DEFAULT_ROOT
    for f in sorted(root.glob("*/*.jsonl")):
        session = parse_transcript(f)
        if session is not None:
            yield session
