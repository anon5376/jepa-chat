"""jepa-ingest: turn chat exports into the unified turns.jsonl."""

from __future__ import annotations

import argparse
import statistics
from collections import Counter
from pathlib import Path

from jepa_chat.data.turns import Session, write_jsonl
from jepa_chat.ingest import parse_chatgpt, parse_claude, parse_claude_code, parse_markdown

DEFAULT_OUT = Path("~/.jepa-chat/data/turns.jsonl").expanduser()


def collect(args: argparse.Namespace) -> list[Session]:
    sessions: list[Session] = []
    for p in args.chatgpt or []:
        sessions.extend(parse_chatgpt(p))
    for p in args.claude or []:
        sessions.extend(parse_claude(p))
    for p in args.markdown or []:
        sessions.extend(parse_markdown(p))
    if args.claude_code:
        sessions.extend(parse_claude_code(args.claude_code_root))
    sessions = [s for s in sessions if len(s.turns) >= args.min_turns]
    sessions.sort(key=lambda s: s.created_ts or 0.0)
    return sessions


def summarize(sessions: list[Session]) -> str:
    by_source = Counter(s.source for s in sessions)
    turns = [t for s in sessions for t in s.turns]
    if not turns:
        return "no turns found"
    sizes = sorted(len(t.text.encode("utf-8")) for t in turns)
    q = statistics.quantiles(sizes, n=100) if len(sizes) > 1 else [sizes[0]] * 99
    roles = Counter(t.role for t in turns)
    lines = [
        f"sessions: {len(sessions)}  turns: {len(turns)}  roles: {dict(roles)}",
        "by source: " + ", ".join(f"{k}={v}" for k, v in by_source.most_common()),
        f"bytes/turn p50={int(q[49])} p90={int(q[89])} p99={int(q[98])} max={sizes[-1]}",
    ]
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description="Ingest chat exports into turns.jsonl")
    ap.add_argument("--chatgpt", action="append", help="official ChatGPT export (.zip/.json/dir)")
    ap.add_argument("--claude", action="append", help="Claude.ai export (.zip/.json/dir)")
    ap.add_argument("--markdown", action="append", help="markdown transcript(s) (.md/.zip/dir)")
    ap.add_argument("--claude-code", action="store_true", help="include local Claude Code transcripts")
    ap.add_argument("--claude-code-root", default=None, help="override ~/.claude/projects")
    ap.add_argument("--min-turns", type=int, default=2)
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    args = ap.parse_args()
    sessions = collect(args)
    n = write_jsonl(sessions, args.out)
    print(summarize(sessions))
    print(f"wrote {n} sessions -> {args.out}")


if __name__ == "__main__":
    main()
