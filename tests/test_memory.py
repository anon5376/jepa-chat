import numpy as np
from conftest import rand_latent

from jepa_chat.core.memory import MemoryStore


def test_add_search_dedupe(tmp_path):
    m = MemoryStore(tmp_path / "m.sqlite", 32)
    s = m.create_session("s1", title="t")
    z = rand_latent(seed=1)
    t = m.add_turn(s.id, "user", "m", "live", "hello world", 100.0, z, 0)
    hits = m.search(z, k=1)
    assert hits and hits[0].turn.id == t.id and hits[0].score > 0.99
    assert m.find_recent_duplicate("hello world", "user", within_s=60, now=120.0) is not None
    assert m.find_recent_duplicate("hello world", "user", within_s=60, now=1000.0) is None
    assert m.get_session("s1").n_turns == 1
    assert m.count_turns() == 1 and m.count_sessions() == 1


def test_reembed_and_windows(tmp_path):
    m = MemoryStore(tmp_path / "m.sqlite", 32)
    for sid in ("a", "b"):
        m.create_session(sid)
        for i in range(5):
            m.add_turn(
                sid, "user" if i % 2 == 0 else "assistant", "x", "live", f"{sid} {i}", float(i), rand_latent(seed=i), 0
            )
    assert len(m.stale_turn_ids(1)) == 10
    ids = m.stale_turn_ids(1, limit=3)
    m.update_latents(ids, np.ones((3, 32), np.float32), 5)
    assert len(m.stale_turn_ids(1)) == 7
    rng = np.random.default_rng(0)
    wins = m.sample_windows(4, 3, rng)
    assert len(wins) == 4 and all(len(w) == 3 for w in wins)
    assert all(len({t.session_id for t in w}) == 1 for w in wins)
    assert all(w[i + 1].idx == w[i].idx + 1 for w in wins for i in range(len(w) - 1))
    # persistence: reopen and search still works
    m.close()
    m2 = MemoryStore(tmp_path / "m.sqlite", 32)
    assert m2.count_turns() == 10 and len(m2.search(rand_latent(seed=2), k=3)) == 3
