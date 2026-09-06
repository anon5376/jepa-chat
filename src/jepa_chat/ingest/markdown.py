"""Parser for markdown transcripts of the form `**User:** ...` / `**Assistant:** ...`.

Used for hand-made archives and copy-pasted chats. Accepts a file, a directory
(all *.md), or a zip (all *.md inside).
"""

from __future__ import annotations

import re
import zipfile
from collections.abc import Iterator
from pathlib import Path

from jepa_chat.data.turns import Session, Turn

_SPEAKER = re.compile(r"^\*\*(User|Assistant|Human|AI|ChatGPT|Claude)\s*:\*\*\s*", re.IGNORECASE)
_ROLE = {"user": "user", "human": "user"}


def parse_markdown_text(text: str, session_id: str, title: str = "") -> Session | None:
    turns: list[Turn] = []
    role: str | None = None
    buf: list[str] = []

    def flush() -> None:
        if role is not None:
            body = "\n".join(buf).strip()
            if body:
                turns.append(Turn(role=role, text=body, mouth="markdown"))

    for line in text.splitlines():
        m = _SPEAKER.match(line.strip())
        if m:
            flush()
            role = _ROLE.get(m.group(1).lower(), "assistant")
            buf = [line.strip()[m.end() :]]
        elif role is not None:
            buf.append(line)
    flush()
    if len(turns) < 2:
        return None
    return Session(id=session_id, title=title, source="markdown", turns=turns)


def parse_markdown(path: str | Path) -> Iterator[Session]:
    path = Path(path)
    if path.suffix == ".zip":
        with zipfile.ZipFile(path) as z:
            for n in sorted(x for x in z.namelist() if x.endswith(".md")):
                s = parse_markdown_text(
                    z.open(n).read().decode("utf-8", "replace"), f"md:{n}", Path(n).stem
                )
                if s:
                    yield s
    elif path.is_dir():
        for f in sorted(path.rglob("*.md")):
            s = parse_markdown_text(
                f.read_text(encoding="utf-8", errors="replace"), f"md:{f.name}", f.stem
            )
            if s:
                yield s
    else:
        s = parse_markdown_text(
            path.read_text(encoding="utf-8", errors="replace"), f"md:{path.name}", path.stem
        )
        if s:
            yield s
