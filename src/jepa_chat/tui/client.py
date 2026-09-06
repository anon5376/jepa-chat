"""Async HTTP/SSE client for the jepa-core daemon."""

from __future__ import annotations

import json
import os
from collections.abc import AsyncIterator

import httpx

DEFAULT_BASE = os.environ.get("JEPA_CORE_URL", "http://127.0.0.1:8765")


class CoreClient:
    def __init__(self, base_url: str = DEFAULT_BASE):
        self.base = base_url.rstrip("/")
        self._http = httpx.AsyncClient(base_url=self.base, timeout=30.0)

    async def state(self, session_id: str | None = None) -> dict:
        r = await self._http.get("/api/state", params={"session_id": session_id} if session_id else None)
        r.raise_for_status()
        return r.json()

    async def stats(self) -> dict:
        r = await self._http.get("/api/stats")
        r.raise_for_status()
        return r.json()

    async def control(self, action: str) -> dict:
        r = await self._http.post("/api/control", json={"action": action})
        return r.json()

    async def session(self, action: str, session_id: str | None = None, title: str | None = None) -> dict:
        r = await self._http.post(
            "/api/session", json={"action": action, "session_id": session_id, "title": title, "mouth": "tui"}
        )
        return r.json()

    async def observe(self, role: str, text: str) -> dict:
        r = await self._http.post("/api/observe", json={"role": role, "text": text, "mouth": "tui"})
        return r.json()

    async def events(self) -> AsyncIterator[dict]:
        """Yield parsed SSE events until the connection drops."""
        async with httpx.AsyncClient(base_url=self.base, timeout=httpx.Timeout(None, connect=5.0)) as http:
            async with http.stream("GET", "/api/events") as r:
                r.raise_for_status()
                data_lines: list[str] = []
                async for line in r.aiter_lines():
                    if line.startswith("data:"):
                        data_lines.append(line[5:].strip())
                    elif line == "" and data_lines:
                        payload = "\n".join(data_lines)
                        data_lines = []
                        try:
                            ev = json.loads(payload)
                        except json.JSONDecodeError:
                            continue
                        if isinstance(ev, dict) and ev.get("type"):
                            yield ev

    async def close(self) -> None:
        await self._http.aclose()
