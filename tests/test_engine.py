import time

from jepa_chat.core.checkpoint import load_checkpoint


def test_online_training_snapshots_and_checkpoint(runtime, tmp_path):
    cfg, memory, engine, bus, state, svc = runtime
    for sid in ("a", "b", "c"):
        svc.session("new", title=sid)
        for i in range(5):
            svc.observe("user" if i % 2 == 0 else "assistant", f"{sid} turn {i} about topic {i % 3}", mouth="t")
    v0 = engine.snapshot().version
    engine.start()
    engine.resume()
    deadline = time.time() + 60
    while engine.step < 6 and time.time() < deadline:
        time.sleep(0.1)
    assert engine.step >= 6, "training did not progress"
    assert engine.snapshot().version > v0
    engine.pause()
    time.sleep(0.3)  # let an in-flight step finish
    s = engine.step
    time.sleep(0.6)
    assert engine.step == s
    assert state.training.loss1 is not None and state.training.loss2 is not None
    ck = tmp_path / "core.pt"
    assert ck.exists()
    loaded = load_checkpoint(ck)
    assert loaded.step >= 4 and loaded.format_version == 2
    engine.request_reembed()
    engine.resume()
    time.sleep(0.5)
    engine.pause()
    assert any(e["type"] == "reembed" for e in bus.history)
