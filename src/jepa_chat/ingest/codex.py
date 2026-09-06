"""Parser for OpenAI Codex CLI rollouts (~/.codex/sessions and ~/.codex/archived_sessions).

Each rollout is JSONL with {type, timestamp, payload}. Dialogue lives in
type=response_item, payload={type:"message", role, content:[{type:input_text|output_text, text}]}.
Tool calls, reasoning, command output and developer/system messages are skipped.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path

from jepa_chat.data.turns import Session, Turn
from jepa_chat.ingest._util import parse_iso_ts

DEFAULT_ROOTS = [Path("~/.codex/sessions").expanduser(), Path("~/.codex/archived_sessions").expanduser()]
_INJECTED_PREFIXES = ("<environment_context>", "<user_instructions>", "# AGENTS.md", "<turn_aborted>", "<permissions")


def _clean_user_text(blocks: list[dict]) -> str:
    parts = []
    for b in blocks:
        if b.get("type") not in ("input_text", "output_text"):
            continue
        t = (b.get("text") or "").strip()
        if not t or t.startswith(_INJECTED_PREFIXES):
            continue
        parts.append(t)
    return "\n".join(parts).strip()


def parse_rollout(path: str | Path, include_subagents: bool = False) -> Session | None:
    path = Path(path)
    turns: list[Turn] = []
    sid, created, cwd, subagent = path.stem, None, "", False
    with path.open(encoding="utf-8", errors="replace") as fh:
        for line in fh:
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            t, p = rec.get("type"), rec.get("payload") or {}
            if t == "session_meta":
                sid = p.get("id") or sid
                created = parse_iso_ts(p.get("timestamp"))
                cwd = p.get("cwd") or ""
                src = p.get("source")
                subagent = isinstance(src, dict) and "subagent" in src
                if subagent and not include_subagents:
                    return None
                continue
            if t != "response_item" or p.get("type") != "message":
                continue
            role = p.get("role")
            if role not in ("user", "assistant"):
                continue
            text = _clean_user_text(p.get("content") or [])
            if not text:
                continue
            ts = parse_iso_ts(rec.get("timestamp"))
            if turns and turns[-1].role == role:
                turns[-1].text += "\n" + text
                continue
            turns.append(Turn(role=role, text=text, ts=ts, mouth="codex"))
    if not turns:
        return None
    return Session(
        id=f"codex:{sid}", title=Path(cwd).name if cwd else "", source="codex", created_ts=created, turns=turns
    )


def parse_codex(root: str | Path | None = None, include_subagents: bool = False) -> Iterator[Session]:
    roots = [Path(root).expanduser()] if root else DEFAULT_ROOTS
    for r in roots:
        if not r.exists():
            continue
        for f in sorted(r.rglob("rollout-*.jsonl")):
            s = parse_rollout(f, include_subagents)
            if s is not None:
                yield s
