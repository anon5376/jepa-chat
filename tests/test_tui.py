import pytest

from jepa_chat.tui.app import JepaDashboard, LatentMap


@pytest.fixture
def anyio_backend():
    return "asyncio"


@pytest.mark.anyio
async def test_dashboard_composes_and_handles_events():
    app = JepaDashboard("http://127.0.0.1:1")  # nothing listening: exercises the offline path
    async with app.run_test(size=(140, 45)) as pilot:
        await pilot.pause(0.2)
        assert app.query_one("#map", LatentMap) is not None
        app.handle_event(
            {
                "type": "turn",
                "ts": 0,
                "turn_id": 1,
                "session_id": "s",
                "role": "user",
                "mouth": "cursor",
                "snippet": "hello",
                "surprise": 0.42,
                "label": "calm",
                "related": [{"score": 0.9, "mouth": "codex", "ts": None, "snippet": "old"}],
                "map": {
                    "points": [
                        {"turn_id": 1, "role": "user", "mouth": "cursor", "x": 0.0, "y": 0.0},
                        {"turn_id": 2, "role": "assistant", "mouth": "cursor", "x": 1.0, "y": 1.0},
                    ],
                    "predicted": {"x": 0.5, "y": 0.5, "role": "user"},
                },
            }
        )
        app.handle_event(
            {
                "type": "train_step",
                "ts": 0,
                "step": 3,
                "steps_per_s": 2.0,
                "loss1": 0.1,
                "loss2": 0.01,
                "latent_std": 0.5,
            }
        )
        app.handle_event({"type": "warning", "ts": 0, "message": "x"})
        await pilot.pause(0.2)
        assert app.surprises == [0.42] and app.loss2 == [0.01]
