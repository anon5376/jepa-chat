import json

import pytest
from mcp import Client

from jepa_chat.mcp.server import build_mcp

pytestmark = pytest.mark.anyio


@pytest.fixture
def anyio_backend():
    return "asyncio"


async def test_tools_and_observe(runtime):
    cfg, memory, engine, bus, state, svc = runtime
    mcp = build_mcp(svc)
    async with Client(mcp) as c:
        tools = {t.name for t in (await c.list_tools()).tools}
        assert tools == {
            "jepa_observe",
            "jepa_state",
            "jepa_recall",
            "jepa_predict_next",
            "jepa_session",
            "jepa_stats",
            "jepa_control",
            "jepa_plan",
        }
        r = await c.call_tool("jepa_observe", {"role": "user", "text": "hello core", "mouth": "cursor"})
        assert not r.is_error
        sc = json.loads(r.content[0].text)
        assert sc["mouth"] == "cursor" and sc["surprise_label"] == "first"
        r2 = await c.call_tool("jepa_observe", {"role": "assistant", "text": "hi!", "mouth": "cursor"})
        assert isinstance(json.loads(r2.content[0].text)["surprise"], float)
        st = json.loads((await c.call_tool("jepa_state", {})).content[0].text)
        assert st["n_turns"] == 2 and "map" not in st
        rec = json.loads((await c.call_tool("jepa_recall", {"query": "hello", "k": 2})).content[0].text)
        assert rec["hits"] and "text" not in rec["hits"][0]
        prompts = {p.name for p in (await c.list_prompts()).prompts}
        assert "jepa_briefing" in prompts
        res = await c.read_resource("jepa://state")
        assert res.contents
