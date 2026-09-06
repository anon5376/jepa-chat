"""Client registration snippets and hook files for Claude Code, Claude Desktop and Cursor."""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

from jepa_chat.core.paths import ensure_dirs, jepa_home
from jepa_chat.mcp.server import BRIEFING

HOOK_SCRIPT = '''#!/usr/bin/env python3
"""Claude Code hook: forward the user's prompt / the assistant's reply to jepa-core.

stdlib only, silent when the daemon is down. Usage: observe.py user|assistant
"""
import json, os, sys, urllib.request

ROLE = sys.argv[1] if len(sys.argv) > 1 else "user"
URL = os.environ.get("JEPA_CORE_URL", "http://127.0.0.1:8765") + "/api/observe"

def main():
    try:
        payload = json.load(sys.stdin)
    except Exception:
        return
    text = payload.get("prompt") if ROLE == "user" else payload.get("last_assistant_message")
    if not text or not str(text).strip():
        return
    body = json.dumps({"role": ROLE, "text": str(text), "mouth": "claude-code"}).encode()
    req = urllib.request.Request(URL, data=body, headers={"content-type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=2.5) as r:
            res = json.load(r)
    except Exception:
        return
    if ROLE != "user":
        return
    s = res.get("surprise")
    label = res.get("surprise_label", "")
    rel = res.get("related") or []
    line = f"[JEPA] surprise {s:.2f} ({label})" if s is not None else "[JEPA] first turn"
    line += f" | session {res.get('session_id')}"
    if rel:
        r0 = rel[0]
        line += f" | related: \\"{r0.get('snippet','')[:90]}\\" ({r0.get('mouth')}, score {r0.get('score')})"
    print(line)

if __name__ == "__main__":
    main()
'''

CURSOR_RULE = (
    """---
description: ChatJEPA core protocol
alwaysApply: true
---
"""
    + BRIEFING
    + """
Always pass mouth="cursor".
"""
)

CLAUDE_MD_SNIPPET = (
    """
## ChatJEPA core (MCP server `jepa`)
"""
    + BRIEFING
    + """
Always pass mouth="claude-code". The UserPromptSubmit/Stop hooks already record turns, so
you only need jepa_recall, jepa_state and jepa_predict_next unless the hook line is missing.
"""
)


def _python() -> str:
    return sys.executable


def _jepa_mcp_bin() -> str:
    found = shutil.which("jepa-mcp")
    return found or str(Path(sys.executable).parent / "jepa-mcp")


def snippets(host: str, port: int) -> dict[str, dict]:
    base = f"http://{host}:{port}"
    home = jepa_home()
    hook = home / "hooks" / "observe.py"
    py = _python()
    return {
        "claude-code": {
            "command": f"claude mcp add --transport http --scope user jepa {base}/mcp",
            "hooks": {
                "hooks": {
                    "UserPromptSubmit": [
                        {"hooks": [{"type": "command", "timeout": 5, "command": f"{py} {hook} user"}]}
                    ],
                    "Stop": [
                        {
                            "hooks": [
                                {
                                    "type": "command",
                                    "timeout": 5,
                                    "command": f"{py} {hook} assistant",
                                }
                            ]
                        }
                    ],
                }
            },
            "hook_path": str(hook),
            "settings_path": str(Path("~/.claude/settings.json").expanduser()),
            "claude_md": CLAUDE_MD_SNIPPET,
        },
        "claude-desktop": {
            "config_path": str(Path("~/Library/Application Support/Claude/claude_desktop_config.json").expanduser()),
            "mcpServers": {
                "jepa": {
                    "command": _jepa_mcp_bin(),
                    "args": ["--url", f"{base}/mcp"],
                    "env": {"JEPA_MOUTH": "claude-desktop"},
                }
            },
        },
        "cursor": {
            "config_path": str(Path("~/.cursor/mcp.json").expanduser()),
            "mcpServers": {"jepa": {"url": f"{base}/mcp", "headers": {"X-JEPA-Mouth": "cursor"}}},
            "rule_path": ".cursor/rules/jepa.mdc",
            "rule": CURSOR_RULE,
        },
    }


def install_client(client: str, host: str = "127.0.0.1", port: int = 8765, write: bool = False) -> None:
    snips = snippets(host, port)
    targets = list(snips) if client == "all" else [client]
    home = ensure_dirs(jepa_home())
    for name in targets:
        s = snips[name]
        print(f"\n=== {name} ===")
        if name == "claude-code":
            hook = Path(s["hook_path"])
            if write:
                hook.write_text(HOOK_SCRIPT)
                hook.chmod(0o755)
                (home / "CLAUDE.md.snippet").write_text(s["claude_md"])
                print(f"wrote hook script {hook}")
                print(f"wrote CLAUDE.md snippet {home / 'CLAUDE.md.snippet'} (paste into ~/.claude/CLAUDE.md)")
            else:
                print(f"(--write would create the hook script at {hook})")
            print("1) register the MCP server:\n   " + s["command"])
            print(f"2) merge into {s['settings_path']}:")
            print(json.dumps(s["hooks"], indent=2))
        elif name == "claude-desktop":
            print(f"merge into {s['config_path']}:")
            print(json.dumps({"mcpServers": s["mcpServers"]}, indent=2))
            print("then restart Claude Desktop. Use the 'jepa_briefing' prompt in a chat/project.")
        elif name == "cursor":
            print(f"merge into {s['config_path']}:")
            print(json.dumps({"mcpServers": s["mcpServers"]}, indent=2))
            if write:
                rule = Path(s["rule_path"])
                rule.parent.mkdir(parents=True, exist_ok=True)
                rule.write_text(s["rule"])
                print(f"wrote project rule {rule}")
            else:
                print(f"(--write would create {s['rule_path']} in the current project)")
            print("allowlist jepa_observe in Cursor's MCP settings to avoid approval clicks.")
    print("\nApp config files are never modified automatically; paste the JSON above.")
