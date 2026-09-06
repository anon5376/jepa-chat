"""Turn / session dataclasses and the unified JSONL interchange format.

One session per line:
{"id": ..., "title": ..., "source": ..., "created_ts": ..., "turns": [{"role", "text", "ts", "mouth"}, ...]}
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Iterator
from dataclasses import asdict, dataclass, field
from pathlib import Path

ROLE_IDS = {"user": 0, "assistant": 1, "other": 2}


def role_id(role: str) -> int:
    return ROLE_IDS.get(role, 2)


@dataclass
class Turn:
    role: str  # "user" | "assistant" | "other"
    text: str
    ts: float | None = None
    mouth: str = ""  # which LLM app produced/received it (claude-code, cursor, chatgpt, ...)

    @property
    def role_id(self) -> int:
        return role_id(self.role)


@dataclass
class Session:
    id: str
    title: str
    source: str  # chatgpt | claude | claude-code | markdown | live
    created_ts: float | None = None
    turns: list[Turn] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> Session:
        turns = [Turn(**t) for t in d.get("turns", [])]
        return cls(
            id=d["id"],
            title=d.get("title", ""),
            source=d.get("source", ""),
            created_ts=d.get("created_ts"),
            turns=turns,
        )


def truncate_turn(text: str, head: int, tail: int) -> str:
    """Keep the first `head` and last `tail` bytes of a long turn (UTF-8 safe)."""
    raw = text.encode("utf-8")
    if len(raw) <= head + tail:
        return text
    if tail <= 0:
        return raw[:head].decode("utf-8", errors="ignore")
    return (
        raw[:head].decode("utf-8", errors="ignore")
        + " … "
        + raw[-tail:].decode("utf-8", errors="ignore")
    )


def write_jsonl(sessions: Iterable[Session], out: str | Path) -> int:
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with out.open("w", encoding="utf-8") as fh:
        for s in sessions:
            fh.write(json.dumps(s.to_dict(), ensure_ascii=False) + "\n")
            n += 1
    return n


def read_jsonl(path: str | Path) -> Iterator[Session]:
    with Path(path).open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                yield Session.from_dict(json.loads(line))
