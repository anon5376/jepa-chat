"""MCP server exposing the JEPA core to any MCP client (Claude Code, Claude Desktop, Cursor)."""

from __future__ import annotations

import asyncio
from typing import Literal

from mcp.server.mcpserver import Context, MCPServer

from jepa_chat.core.service import CoreService

INSTRUCTIONS = """\
You are one of several "mouths" of a local JEPA (joint-embedding predictive architecture) core.
The core keeps the conversation state and long-term latent memory; you produce the words.

Protocol:
1. Before answering a user message, call jepa_observe(role="user", text=<the message>).
   Read the returned surprise level and related memories; mention relevant ones naturally.
2. After you answer, call jepa_observe(role="assistant", text=<your answer>).
3. When the user refers to something from the past, call jepa_recall(query=...).
4. If the user says "continue where I left off", call jepa_state() and jepa_recall().
5. Pass mouth=<your app name> (claude-code, claude-desktop, cursor, chatgpt) on every call.
Never invent memories: only cite what jepa_recall / jepa_observe returned.
"""

BRIEFING = (
    INSTRUCTIONS
    + """
Surprise scale: 0 = the core predicted this turn perfectly, ~1 = no better than chance.
"calm" (<0.5) means the conversation is on its predicted track; "surprising" (>0.9) means a
topic shift the core did not anticipate. You may comment on this when it is interesting.
"""
)


def _mouth(ctx: Context | None, mouth: str | None) -> str:
    if mouth and mouth != "unknown":
        return mouth
    if ctx is not None:
        try:
            h = ctx.headers
            if h and h.get("x-jepa-mouth"):
                return str(h.get("x-jepa-mouth"))
        except Exception:  # noqa: BLE001
            pass
    return mouth or "unknown"


def build_mcp(svc: CoreService) -> MCPServer:
    mcp = MCPServer(
        "jepa",
        title="ChatJEPA core",
        instructions=INSTRUCTIONS,
        version="0.1.0",
    )

    @mcp.tool(description="Record a conversation turn in the JEPA core. Returns surprise and related memories.")
    async def jepa_observe(
        role: Literal["user", "assistant"],
        text: str,
        mouth: str = "unknown",
        session_id: str | None = None,
        ctx: Context | None = None,
    ) -> dict:
        return await asyncio.to_thread(svc.observe, role, text, _mouth(ctx, mouth), session_id)

    @mcp.tool(description="Current conversation state: turns, mouths seen, surprise trend, top memories.")
    async def jepa_state(session_id: str | None = None) -> dict:
        s = await asyncio.to_thread(svc.state_snapshot, session_id)
        s.pop("map", None)
        s.pop("metrics", None)
        return s

    @mcp.tool(description="Recall past turns (across all sessions and apps) related to a query.")
    async def jepa_recall(
        query: str,
        k: int = 5,
        cross_session: bool = True,
        role: Literal["user", "assistant"] | None = None,
    ) -> dict:
        res = await asyncio.to_thread(svc.recall, query, k, cross_session, role)
        for h in res["hits"]:
            h.pop("text", None)
        return res

    @mcp.tool(description="What the core expects the user to say next: nearest past turns to the predicted latent.")
    async def jepa_predict_next(session_id: str | None = None, k: int = 5) -> dict:
        res = await asyncio.to_thread(svc.predict_next, session_id, k)
        for h in res["hints"]:
            h.pop("text", None)
        return res

    @mcp.tool(description="Manage sessions: new | resume | list | current.")
    async def jepa_session(
        action: Literal["new", "resume", "list", "current"],
        session_id: str | None = None,
        title: str | None = None,
        mouth: str = "unknown",
        ctx: Context | None = None,
    ) -> dict:
        return await asyncio.to_thread(svc.session, action, session_id, title, _mouth(ctx, mouth))

    @mcp.tool(description="Training and memory statistics of the core.")
    async def jepa_stats() -> dict:
        return svc.stats()

    @mcp.tool(description="Control online learning: pause | resume | checkpoint | reembed.")
    async def jepa_control(action: Literal["pause", "resume", "checkpoint", "reembed"]) -> dict:
        return svc.control(action)

    @mcp.prompt(name="jepa_briefing", description="How to behave as a mouth of the JEPA core.")
    def jepa_briefing() -> str:
        return BRIEFING

    @mcp.resource("jepa://state", description="Live conversation state as JSON.")
    def state_resource() -> str:
        import json

        s = svc.state_snapshot()
        s.pop("map", None)
        s.pop("metrics", None)
        return json.dumps(s, default=str)

    return mcp
