"""jepa-core: serve | status | import-memory | install-client | checkpoint."""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import sys
import urllib.error
import urllib.request

from jepa_chat.core.paths import default_checkpoint, pretrained_checkpoint, turns_jsonl


def _url(args: argparse.Namespace, path: str) -> str:
    return f"http://{args.host}:{args.port}{path}"


def _get(url: str) -> dict:
    with urllib.request.urlopen(url, timeout=5) as r:  # noqa: S310 - localhost only
        return json.load(r)


def _post(url: str, body: dict) -> dict:
    req = urllib.request.Request(url, data=json.dumps(body).encode(), headers={"content-type": "application/json"})
    with urllib.request.urlopen(req, timeout=600) as r:  # noqa: S310
        return json.load(r)


def cmd_serve(args: argparse.Namespace) -> None:
    from jepa_chat.core.app import build_runtime, serve

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    rt = build_runtime(args.checkpoint, train=not args.no_train, device=args.device)
    print(f"jepa-core on http://{args.host}:{args.port}  mcp: /mcp  api: /api  device: {rt.engine.device}")
    print(f"memory: {rt.memory.count_sessions()} sessions, {rt.memory.count_turns()} turns")
    serve(rt, args.host, args.port)


def cmd_status(args: argparse.Namespace) -> None:
    try:
        print(json.dumps(_get(_url(args, "/api/stats")), indent=2))
    except (urllib.error.URLError, ConnectionError) as e:
        print(f"jepa-core not reachable at {_url(args, '')}: {e}")
        sys.exit(1)


def cmd_import(args: argparse.Namespace) -> None:
    """Import turns.jsonl into the *running* daemon's memory (embeds on its accelerator)."""
    from jepa_chat.data.turns import read_jsonl

    sessions = list(read_jsonl(args.data))
    print(f"importing {len(sessions)} sessions via {_url(args, '/api/import')} ...")
    res = _post(_url(args, "/api/import"), {"path": args.data})
    print(res)


def cmd_control(args: argparse.Namespace) -> None:
    print(_post(_url(args, "/api/control"), {"action": args.action}))


def cmd_reset(args: argparse.Namespace) -> None:
    """Discard online learning: replace the live checkpoint with the pretrained one (stop the daemon first)."""
    src, dst = pretrained_checkpoint(), default_checkpoint()
    if not src.exists():
        print(f"no pretrained checkpoint at {src}; run jepa-pretrain first")
        sys.exit(1)
    if dst.exists():
        backup = dst.with_suffix(".bak.pt")
        shutil.copy2(dst, backup)
        print(f"backed up live checkpoint to {backup}")
    shutil.copy2(src, dst)
    print(f"reset {dst} from {src}; restart jepa-core (it will re-embed memory on start)")


def cmd_install(args: argparse.Namespace) -> None:
    from jepa_chat.mcp.clients import install_client

    install_client(args.client, host=args.host, port=args.port, write=args.write)


def main() -> None:
    ap = argparse.ArgumentParser(prog="jepa-core")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8765)
    sub = ap.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("serve", help="run the daemon")
    s.add_argument(
        "--checkpoint",
        default=str(default_checkpoint()),
        help="live checkpoint path; falls back to checkpoints/pretrained.pt when missing",
    )
    s.add_argument("--no-train", action="store_true", help="disable online learning")
    s.add_argument("--device", default="auto")
    s.set_defaults(fn=cmd_serve)

    st = sub.add_parser("status", help="print daemon stats")
    st.set_defaults(fn=cmd_status)

    im = sub.add_parser("import-memory", help="load turns.jsonl into the running daemon")
    im.add_argument("data", nargs="?", default=str(turns_jsonl()))
    im.set_defaults(fn=cmd_import)

    c = sub.add_parser("control", help="pause | resume | checkpoint | reembed")
    c.add_argument("action", choices=["pause", "resume", "checkpoint", "reembed"])
    c.set_defaults(fn=cmd_control)

    rs = sub.add_parser("reset", help="replace live checkpoint with the pretrained one")
    rs.set_defaults(fn=cmd_reset)

    ic = sub.add_parser("install-client", help="print (or write) client registration")
    ic.add_argument("client", choices=["claude-code", "claude-desktop", "cursor", "all"])
    ic.add_argument("--write", action="store_true", help="write hook/rule files and print app config")
    ic.set_defaults(fn=cmd_install)

    args = ap.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
