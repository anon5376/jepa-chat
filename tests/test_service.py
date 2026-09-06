def test_observe_surprise_and_mouths(runtime):
    cfg, memory, engine, bus, state, svc = runtime
    events = []
    bus.add_listener(events.append)
    r1 = svc.observe("user", "hello from claude", mouth="claude-code")
    assert r1["surprise"] is None and r1["surprise_label"] == "first"
    sid = r1["session_id"]
    r2 = svc.observe("assistant", "hi, how can I help?", mouth="claude-code", session_id=sid)
    assert isinstance(r2["surprise"], float)
    r3 = svc.observe("user", "continue where I left off", mouth="cursor", session_id=sid)
    assert isinstance(r3["surprise"], float) and r3["surprise_label"] in ("calm", "notable", "surprising")
    st = svc.state_snapshot(sid)
    assert st["n_turns"] == 3 and st["mouths_seen"] == ["claude-code", "cursor"]
    assert len(st["map"]["points"]) == 3 and st["map"]["predicted"] is not None
    assert any(e["type"] == "turn" for e in events)
    # dedupe within window
    r4 = svc.observe("user", "continue where I left off", mouth="cursor", session_id=sid)
    assert r4["deduped"] and svc.state_snapshot(sid)["n_turns"] == 3


def test_sessions_recall_predict(runtime):
    cfg, memory, engine, bus, state, svc = runtime
    a = svc.session("new", title="A", mouth="claude-code")["active"]["id"]
    svc.observe("user", "latent collapse in JEPA is scary", mouth="claude-code")
    svc.observe("assistant", "monitor latent_std", mouth="claude-code")
    b = svc.session("new", title="B", mouth="cursor")["active"]["id"]
    assert a != b and svc.session("list")["active"]["id"] == b
    svc.observe("user", "what did we say about collapse?", mouth="cursor")
    rec = svc.recall("latent collapse", k=3)
    assert rec["hits"] and {h["session_id"] for h in rec["hits"]} >= {a}
    pred = svc.predict_next()
    assert pred["session_id"] == b and isinstance(pred["hints"], list)
    resumed = svc.session("resume", session_id=a)
    assert resumed["active"]["id"] == a and resumed["active"]["n_turns"] == 2
    st = svc.state_snapshot(a)
    assert st["n_turns"] == 2 and st["mouths_seen"] == ["claude-code"]
    assert svc.stats()["memory_turns"] == 3
    assert svc.control("pause")["paused"] is True


def test_import_sessions(runtime):
    from jepa_chat.data.turns import Session, Turn

    cfg, memory, engine, bus, state, svc = runtime
    engine.start()
    sessions = [
        Session(
            id="chatgpt:1",
            title="old",
            source="chatgpt",
            created_ts=1.0,
            turns=[Turn("user", "old question", 1.0, "chatgpt"), Turn("assistant", "old answer", 2.0, "chatgpt")],
        )
    ]
    assert svc.import_sessions(sessions, log=lambda *_: None) == 1
    assert svc.import_sessions(sessions, log=lambda *_: None) == 0
    assert memory.count_turns() == 2
    assert svc.recall("old question", k=1)["hits"][0]["source"] == "chatgpt"
