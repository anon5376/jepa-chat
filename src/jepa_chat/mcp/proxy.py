"""jepa-mcp: stdio MCP server that forwards everything to the running jepa-core over HTTP.

For clients that only launch stdio servers (Claude Desktop). Tolerates daemon restarts.
"""

from __future__ import annotations

import argparse
import os
import sys

import anyio
from mcp import Client, types
from mcp.server import Server
from mcp.server.stdio import stdio_server

DEFAULT_URL = os.environ.get("JEPA_CORE_URL", "http://127.0.0.1:8765/mcp")


class Upstream:
    """One short-lived HTTP client per request.

    anyio cancel scopes are task-bound, so a client entered in one handler task cannot be
    reused from another; a fresh connection per call is cheap on localhost and robust to
    daemon restarts.
    """

    def __init__(self, url: str, mouth: str):
        self.url = url
        self.mouth = mouth

    def client(self) -> Client:
        return Client(self.url)


def build_proxy(url: str, mouth: str) -> Server:
    up = Upstream(url, mouth)

    def down(msg: str) -> types.CallToolResult:
        return types.CallToolResult(content=[types.TextContent(type="text", text=msg)], isError=True)

    async def on_list_tools(ctx, params):
        try:
            async with up.client() as c:
                return await c.list_tools()
        except Exception:  # noqa: BLE001
            return types.ListToolsResult(tools=[])

    async def on_call_tool(ctx, params: types.CallToolRequestParams):
        args = dict(params.arguments or {})
        if "mouth" in args or params.name in ("jepa_observe", "jepa_session"):
            args.setdefault("mouth", mouth)
        try:
            async with up.client() as c:
                return await c.call_tool(params.name, args)
        except Exception as e:  # noqa: BLE001
            return down(f"jepa-core not reachable at {url} ({e}). Start it with: uv run jepa-core serve")

    async def on_list_prompts(ctx, params):
        try:
            async with up.client() as c:
                return await c.list_prompts()
        except Exception:  # noqa: BLE001
            return types.ListPromptsResult(prompts=[])

    async def on_get_prompt(ctx, params: types.GetPromptRequestParams):
        async with up.client() as c:
            return await c.get_prompt(params.name, params.arguments)

    async def on_list_resources(ctx, params):
        try:
            async with up.client() as c:
                return await c.list_resources()
        except Exception:  # noqa: BLE001
            return types.ListResourcesResult(resources=[])

    async def on_read_resource(ctx, params: types.ReadResourceRequestParams):
        async with up.client() as c:
            return await c.read_resource(params.uri)

    return Server(
        "jepa",
        instructions="Proxy to the local ChatJEPA core. See the jepa_briefing prompt.",
        on_list_tools=on_list_tools,
        on_call_tool=on_call_tool,
        on_list_prompts=on_list_prompts,
        on_get_prompt=on_get_prompt,
        on_list_resources=on_list_resources,
        on_read_resource=on_read_resource,
    )


async def _run(url: str, mouth: str) -> None:
    server = build_proxy(url, mouth)
    async with stdio_server() as (read, write):
        await server.run(read, write, server.create_initialization_options())


def main() -> None:
    ap = argparse.ArgumentParser(prog="jepa-mcp")
    ap.add_argument("--url", default=DEFAULT_URL)
    ap.add_argument("--mouth", default=os.environ.get("JEPA_MOUTH", "claude-desktop"))
    args = ap.parse_args()
    try:
        anyio.run(_run, args.url, args.mouth)
    except KeyboardInterrupt:
        sys.exit(0)


if __name__ == "__main__":
    main()
