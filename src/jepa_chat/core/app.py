"""Starlette application: MCP at /mcp, JSON API at /api/*, SSE at /api/events."""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
from dataclasses import dataclass
from pathlib import Path

import torch
from mcp.server.transport_security import TransportSecuritySettings
from sse_starlette.sse import EventSourceResponse
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.routing import Mount, Route

from jepa_chat.config import Config
from jepa_chat.core.checkpoint import build_models, load_checkpoint
from jepa_chat.core.engine import ModelThread
from jepa_chat.core.memory import MemoryStore
from jepa_chat.core.paths import ensure_dirs, jepa_home, memory_db
from jepa_chat.core.service import CoreService
from jepa_chat.core.state import CoreState, EventBus
from jepa_chat.mcp.server import build_mcp
from jepa_chat.training.train import pick_device

log = logging.getLogger("jepa.core")


@dataclass
class CoreRuntime:
    cfg: Config
    memory: MemoryStore
    engine: ModelThread
    bus: EventBus
    state: CoreState
    service: CoreService


def build_runtime(checkpoint: str | Path | None, train: bool = True, device: str = "auto") -> CoreRuntime:
    home = ensure_dirs(jepa_home())
    if checkpoint and Path(checkpoint).expanduser().exists():
        loaded = load_checkpoint(checkpoint)
        cfg, l1, l2 = loaded.cfg, loaded.level1, loaded.level2
        step, ver, o1, o2, pstd = (
            loaded.step,
            loaded.enc_version,
            loaded.opt1_state,
            loaded.opt2_state,
            loaded.pretrain_latent_std,
        )
        log.info("loaded checkpoint %s (v%d, step %d)", checkpoint, loaded.format_version, step)
    else:
        cfg = Config()
        cfg.model.max_len = 512
        l1, l2 = build_models(cfg)
        step, ver, o1, o2, pstd = 0, 0, None, None, None
        log.warning("no checkpoint at %s: starting from random weights", checkpoint)
    dev = pick_device(device if device != "auto" else cfg.core.device)
    bus = EventBus()
    state = CoreState()
    memory = MemoryStore(memory_db(home), cfg.model.d_model)
    engine = ModelThread(
        cfg,
        l1,
        l2,
        memory,
        bus,
        state,
        dev,
        checkpoint_path=str(checkpoint or (home / "checkpoints" / "core.pt")),
        step=step,
        enc_version=ver,
        opt1_state=o1,
        opt2_state=o2,
        pretrain_latent_std=pstd,
        train=train,
    )
    service = CoreService(cfg, memory, engine, bus, state)
    return CoreRuntime(cfg, memory, engine, bus, state, service)


def _json_error(msg: str, status: int = 400) -> JSONResponse:
    return JSONResponse({"error": msg}, status_code=status)


def create_app(rt: CoreRuntime) -> Starlette:
    svc = rt.service
    mcp = build_mcp(svc)
    security = TransportSecuritySettings(
        enable_dns_rebinding_protection=True,
        allowed_hosts=["127.0.0.1", "127.0.0.1:*", "localhost", "localhost:*"],
        allowed_origins=["http://127.0.0.1", "http://127.0.0.1:*", "http://localhost", "http://localhost:*"],
    )
    mcp_app = mcp.streamable_http_app(streamable_http_path="/mcp", transport_security=security)

    async def observe(request: Request):
        body = await request.json()
        try:
            res = await asyncio.to_thread(
                svc.observe,
                body.get("role", "user"),
                body.get("text", ""),
                body.get("mouth") or request.headers.get("x-jepa-mouth", "unknown"),
                body.get("session_id"),
            )
        except ValueError as e:
            return _json_error(str(e))
        return JSONResponse(res)

    async def state(request: Request):
        sid = request.query_params.get("session_id")
        return JSONResponse(await asyncio.to_thread(svc.state_snapshot, sid))

    async def recall(request: Request):
        body = await request.json()
        res = await asyncio.to_thread(
            svc.recall,
            body.get("query", ""),
            int(body.get("k", 5)),
            bool(body.get("cross_session", True)),
            body.get("role"),
            body.get("session_id"),
        )
        return JSONResponse(res)

    async def predict(request: Request):
        body = await request.json() if request.method == "POST" else {}
        return JSONResponse(await asyncio.to_thread(svc.predict_next, body.get("session_id"), int(body.get("k", 5))))

    async def plan(request: Request):
        body = await request.json()
        return JSONResponse(
            await asyncio.to_thread(
                svc.plan, list(body.get("candidates") or []), body.get("session_id"), int(body.get("horizon", 1))
            )
        )

    async def session(request: Request):
        body = await request.json()
        try:
            res = await asyncio.to_thread(
                svc.session,
                body.get("action", "list"),
                body.get("session_id"),
                body.get("title"),
                body.get("mouth") or request.headers.get("x-jepa-mouth", ""),
            )
        except ValueError as e:
            return _json_error(str(e))
        return JSONResponse(res)

    async def control(request: Request):
        body = await request.json()
        try:
            return JSONResponse(svc.control(body.get("action", "")))
        except ValueError as e:
            return _json_error(str(e))

    async def stats(request: Request):
        return JSONResponse(svc.stats())

    async def import_memory(request: Request):
        from jepa_chat.data.turns import read_jsonl

        body = await request.json()
        path = Path(body.get("path", "")).expanduser()
        if not path.exists():
            return _json_error(f"no such file {path}", 404)
        sessions = list(read_jsonl(path))
        n = await asyncio.to_thread(svc.import_sessions, sessions, log.info)
        return JSONResponse({"imported": n, "memory_turns": rt.memory.count_turns()})

    async def events(request: Request):
        q = rt.bus.subscribe()

        async def gen():
            try:
                for ev in list(rt.bus.history)[-20:]:
                    yield {"event": ev["type"], "data": json.dumps(ev, default=str)}
                while True:
                    if await request.is_disconnected():
                        break
                    try:
                        ev = await asyncio.wait_for(q.get(), timeout=15.0)
                    except TimeoutError:
                        yield {"event": "ping", "data": "{}"}
                        continue
                    yield {"event": ev["type"], "data": json.dumps(ev, default=str)}
            finally:
                rt.bus.unsubscribe(q)

        return EventSourceResponse(gen())

    @contextlib.asynccontextmanager
    async def lifespan(app: Starlette):
        rt.bus.bind_loop(asyncio.get_running_loop())
        if not rt.engine.is_alive():
            rt.engine.start()
        async with mcp.session_manager.run():
            try:
                yield
            finally:
                rt.engine.stop()
                await asyncio.to_thread(rt.engine.join, 30.0)
                rt.memory.close()

    routes = [
        Route("/api/observe", observe, methods=["POST"]),
        Route("/api/state", state, methods=["GET"]),
        Route("/api/recall", recall, methods=["POST"]),
        Route("/api/predict", predict, methods=["GET", "POST"]),
        Route("/api/session", session, methods=["POST"]),
        Route("/api/plan", plan, methods=["POST"]),
        Route("/api/control", control, methods=["POST"]),
        Route("/api/stats", stats, methods=["GET"]),
        Route("/api/import", import_memory, methods=["POST"]),
        Route("/api/events", events, methods=["GET"]),
        Mount("/", app=mcp_app),
    ]
    app = Starlette(routes=routes, lifespan=lifespan)
    app.state.runtime = rt
    return app


def serve(rt: CoreRuntime, host: str, port: int) -> None:
    import uvicorn

    torch.set_num_threads(max(1, torch.get_num_threads() // 2))
    uvicorn.run(create_app(rt), host=host, port=port, log_level="warning")
