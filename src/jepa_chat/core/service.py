"""CoreService: the one facade used by MCP tools, the REST API, hooks and the TUI."""

from __future__ import annotations

import time
from typing import Literal

import numpy as np

from jepa_chat.config import Config
from jepa_chat.core.engine import ModelThread, surprise_ratio
from jepa_chat.core.memory import Hit, MemoryStore, TurnRow
from jepa_chat.core.state import CoreState, EventBus, SessionState
from jepa_chat.data.turns import role_id
from jepa_chat.models.conversation import normalize_latent

DEDUPE_WINDOW_S = 60.0


def surprise_label(ratio: float | None) -> str:
    if ratio is None:
        return "first"
    if ratio < 0.5:
        return "calm"
    if ratio < 0.9:
        return "notable"
    return "surprising"


def _snippet(text: str, n: int = 160) -> str:
    text = " ".join(text.split())
    return text if len(text) <= n else text[: n - 1] + "…"


def _hit_dict(h: Hit, session_title: str = "") -> dict:
    t = h.turn
    return {
        "turn_id": t.id,
        "session_id": t.session_id,
        "session_title": session_title,
        "mouth": t.mouth,
        "source": t.source,
        "role": t.role,
        "ts": t.ts,
        "snippet": _snippet(t.text),
        "text": t.text,
        "score": round(h.score, 4),
    }


class CoreService:
    def __init__(self, cfg: Config, memory: MemoryStore, engine: ModelThread, bus: EventBus, state: CoreState):
        self.cfg = cfg
        self.memory = memory
        self.engine = engine
        self.bus = bus
        self.state = state
        self.rng = np.random.default_rng(0)
        self._pca_cache: tuple[tuple, dict] | None = None

    # ------------------------------------------------------------ sessions
    def _resolve_session(self, session_id: str | None, mouth: str = "") -> SessionState:
        with self.state.lock:
            sid = session_id or self.state.active_session_id
            row = self.memory.get_session(sid) if sid else None
            if row is None:
                row = self.memory.create_session(sid, title="", source="live", mouth=mouth)
                self.bus.publish("session", action="new", session_id=row.id, mouth=mouth)
            if self.state.active_session_id != row.id:
                self.state.active_session_id = row.id
                self.bus.publish("session", action="active", session_id=row.id, mouth=mouth)
            ss = self.state.session(row.id)
            if ss.n_turns == 0 and row.n_turns > 0:
                self._rehydrate(ss)
            return ss

    def _get_session_state(self, session_id: str | None) -> SessionState | None:
        """Session state for reads: rehydrates from memory if it is not in RAM yet."""
        sid = session_id or self.state.active_session_id
        if not sid:
            recent = self.memory.list_sessions(1)
            if not recent:
                return None
            sid = recent[0].id
            with self.state.lock:
                self.state.active_session_id = sid
        with self.state.lock:
            ss = self.state.sessions.get(sid)
            if ss is None:
                if self.memory.get_session(sid) is None:
                    return None
                ss = self.state.session(sid)
                self._rehydrate(ss)
            return ss

    def _rehydrate(self, ss: SessionState) -> None:
        turns = self.memory.session_turns(ss.session_id)
        ss.n_turns = len(turns)
        ss.mouths = {t.mouth for t in turns if t.mouth}
        ss.surprises = [t.surprise for t in turns if t.surprise is not None][-50:]
        ss.last_role = turns[-1].role if turns else None
        ss.last_ts = turns[-1].ts if turns else None
        self._update_state(ss, turns)

    def _update_state(self, ss: SessionState, turns: list[TurnRow]) -> None:
        rows = [t for t in turns if t.latent is not None]
        if not rows:
            return
        snap = self.engine.snapshot()
        z = np.stack([t.latent for t in rows])
        roles = [role_id(t.role) for t in rows]
        ss.h_last, ss.pending_pred = snap.encode_session(z, roles)

    def session(
        self,
        action: Literal["new", "resume", "list", "current"],
        session_id: str | None = None,
        title: str | None = None,
        mouth: str = "",
    ) -> dict:
        if action == "new":
            row = self.memory.create_session(None, title=title or "", source="live", mouth=mouth)
            with self.state.lock:
                self.state.active_session_id = row.id
                self.state.sessions.pop(row.id, None)
            self.bus.publish("session", action="new", session_id=row.id, mouth=mouth)
        elif action == "resume":
            if not session_id or self.memory.get_session(session_id) is None:
                raise ValueError(f"unknown session {session_id!r}")
            with self.state.lock:
                self.state.active_session_id = session_id
                self.state.sessions.pop(session_id, None)
            self.bus.publish("session", action="resume", session_id=session_id, mouth=mouth)
        active = self.state.active_session_id
        active_row = self.memory.get_session(active) if active else None
        sessions = [
            {
                "id": s.id,
                "title": s.title,
                "source": s.source,
                "mouth": s.mouth,
                "n_turns": s.n_turns,
                "last_ts": s.last_ts,
                "active": s.id == active,
            }
            for s in self.memory.list_sessions(30)
        ]
        return {
            "active": (
                {"id": active_row.id, "title": active_row.title, "n_turns": active_row.n_turns} if active_row else None
            ),
            "sessions": sessions,
        }

    # ------------------------------------------------------------ observe
    def observe(
        self,
        role: str,
        text: str,
        mouth: str = "unknown",
        session_id: str | None = None,
        ts: float | None = None,
    ) -> dict:
        role = "user" if role in ("user", "human") else "assistant" if role == "assistant" else "other"
        text = text.strip()
        if not text:
            raise ValueError("empty text")
        now = ts or time.time()
        dup = self.memory.find_recent_duplicate(text, role, DEDUPE_WINDOW_S, now)
        if dup is not None:
            return self._observe_result(dup, dup.surprise, None, [], deduped=True)

        ss = self._resolve_session(session_id, mouth)
        snap = self.engine.snapshot()
        z = snap.embed([text])[0]
        rid = role_id(role)

        ratio: float | None = None
        cos: float | None = None
        pred = ss.pending_pred.get(rid)
        if pred is not None:
            baseline = self.memory.random_latents(32, self.rng)
            ratio, cos = surprise_ratio(pred, z, baseline)

        related = self.memory.search(z, k=3, exclude_session=ss.session_id)
        row = self.memory.add_turn(ss.session_id, role, mouth, "live", text, now, z, snap.version, ratio, pred)
        self.memory.touch_session(ss.session_id, now, mouth)
        ss.n_turns += 1
        ss.mouths.add(mouth)
        ss.last_role, ss.last_ts = role, now
        if ratio is not None:
            ss.surprises = (ss.surprises + [ratio])[-50:]
        self._update_state(ss, self.memory.session_turns(ss.session_id, self.cfg.conversation.max_turns))

        result = self._observe_result(row, ratio, cos, related)
        self.bus.publish(
            "turn",
            turn_id=row.id,
            session_id=ss.session_id,
            role=role,
            mouth=mouth,
            snippet=_snippet(text, 100),
            surprise=ratio,
            label=result["surprise_label"],
            related=result["related"],
            map=self._map(ss),
        )
        return result

    def _observe_result(
        self,
        row: TurnRow,
        ratio: float | None,
        cos: float | None,
        related: list[Hit],
        deduped: bool = False,
    ) -> dict:
        ss = self.state.sessions.get(row.session_id)
        hint = self._hint(ss) if ss else None
        return {
            "turn_id": row.id,
            "session_id": row.session_id,
            "role": row.role,
            "mouth": row.mouth,
            "surprise": None if ratio is None else round(ratio, 3),
            "surprise_label": surprise_label(ratio),
            "cos_pred_actual": None if cos is None else round(cos, 3),
            "related": [_hit_dict(h, self._title(h.turn.session_id)) for h in related],
            "next_turn_hint": hint,
            "deduped": deduped,
        }

    def _title(self, session_id: str) -> str:
        row = self.memory.get_session(session_id)
        return row.title if row else ""

    def _hint(self, ss: SessionState, k: int = 1) -> str | None:
        pred = ss.pending_pred.get(0)
        if pred is None:
            return None
        hits = self.memory.search(pred, k=k, role="user", exclude_session=ss.session_id)
        return _snippet(hits[0].turn.text, 120) if hits else None

    # ------------------------------------------------------------ recall / predict
    def recall(
        self,
        query: str,
        k: int = 5,
        cross_session: bool = True,
        role: str | None = None,
        session_id: str | None = None,
    ) -> dict:
        z = self.engine.snapshot().embed([query])[0]
        exclude = None if cross_session else None
        hits = self.memory.search(z, k=k, role=role, exclude_session=exclude)
        if not cross_session:
            sid = session_id or self.state.active_session_id
            hits = [h for h in hits if h.turn.session_id == sid]
        out = [_hit_dict(h, self._title(h.turn.session_id)) for h in hits]
        self.bus.publish("recall", query=_snippet(query, 80), hits=[o["snippet"] for o in out])
        return {"hits": out}

    def predict_next(self, session_id: str | None = None, k: int = 5, role: str = "user") -> dict:
        sid = session_id or self.state.active_session_id
        ss = self._get_session_state(sid)
        if ss is None:
            return {"hints": [], "confidence": 0.0, "session_id": sid}
        pred = ss.pending_pred.get(role_id(role))
        if pred is None:
            return {"hints": [], "confidence": 0.0, "session_id": sid}
        hits = self.memory.search(pred, k=k, role=role)
        conf = float(np.mean([h.score for h in hits])) if hits else 0.0
        return {
            "session_id": sid,
            "role": role,
            "hints": [_hit_dict(h, self._title(h.turn.session_id)) for h in hits],
            "confidence": round(conf, 3),
        }

    # ------------------------------------------------------------ state / stats
    def _map(self, ss: SessionState) -> dict:
        turns = self.memory.session_turns(ss.session_id, self.cfg.core.map_points)
        rows = [t for t in turns if t.latent is not None]
        if len(rows) < 2:
            return {"points": [], "predicted": None}
        import torch

        z = normalize_latent(torch.from_numpy(np.stack([t.latent for t in rows]))).numpy()
        mean = z.mean(0, keepdims=True)
        zc = z - mean
        _, _, vt = np.linalg.svd(zc, full_matrices=False)
        proj = zc @ vt[:2].T
        pts = [
            {"turn_id": t.id, "role": t.role, "mouth": t.mouth, "x": float(p[0]), "y": float(p[1])}
            for t, p in zip(rows, proj, strict=True)
        ]
        predicted = None
        pp = ss.pending_pred.get(0)
        if pp is not None:
            q = (pp[None] - mean) @ vt[:2].T
            predicted = {"x": float(q[0, 0]), "y": float(q[0, 1]), "role": "user"}
        return {"points": pts, "predicted": predicted}

    def state_snapshot(self, session_id: str | None = None) -> dict:
        sid = session_id or self.state.active_session_id
        ss = self._get_session_state(sid)
        row = self.memory.get_session(sid) if sid else None
        top: list[dict] = []
        if ss is not None and ss.h_last is not None:
            last = self.memory.session_turns(sid, 1)
            if last and last[0].latent is not None:
                top = [
                    _hit_dict(h, self._title(h.turn.session_id))
                    for h in self.memory.search(last[0].latent, 3, exclude_session=sid)
                ]
        return {
            "session_id": sid,
            "title": row.title if row else "",
            "n_turns": ss.n_turns if ss else (row.n_turns if row else 0),
            "mouths_seen": sorted(ss.mouths) if ss else [],
            "surprise_recent": [round(s, 3) for s in ss.surprises] if ss else [],
            "surprise_mean": round(float(np.mean(ss.surprises)), 3) if ss and ss.surprises else None,
            "latent_norm": round(float(np.linalg.norm(ss.h_last)), 3) if ss and ss.h_last is not None else None,
            "top_memory": top,
            "map": self._map(ss) if ss else {"points": [], "predicted": None},
            "training": self.state.training.to_dict(),
            "sessions": self.session("list")["sessions"],
            "metrics": self.memory.recent_metrics(60),
        }

    def stats(self) -> dict:
        t = self.state.training.to_dict()
        t.update(
            {
                "memory_turns": self.memory.count_turns(),
                "memory_sessions": self.memory.count_sessions(),
                "uptime_s": round(time.time() - self.state.started_ts, 1),
                "active_session": self.state.active_session_id,
            }
        )
        return t

    def control(self, action: Literal["pause", "resume", "checkpoint", "reembed"]) -> dict:
        if action == "pause":
            self.engine.pause()
        elif action == "resume":
            self.engine.resume()
        elif action == "checkpoint":
            self.engine.request_checkpoint()
        elif action == "reembed":
            self.engine.request_reembed()
        else:
            raise ValueError(f"unknown action {action!r}")
        return {"ok": True, "paused": self.engine.paused, "action": action}

    # ------------------------------------------------------------ import
    def import_sessions(self, sessions, log=print) -> int:
        """Bulk-load ingested sessions into memory using the accelerator for embedding."""
        n = 0
        for s in sessions:
            if self.memory.get_session(s.id) is not None:
                continue
            self.memory.create_session(s.id, title=s.title, source=s.source, mouth=s.turns[0].mouth if s.turns else "")
            z = self.engine.embed_bulk([t.text for t in s.turns])
            for t, zt in zip(s.turns, z, strict=True):
                self.memory.add_turn(s.id, t.role, t.mouth, s.source, t.text, t.ts, zt, self.engine.enc_version)
            n += 1
            if n % 25 == 0:
                log(f"imported {n} sessions")
        self.bus.publish("import", sessions=n)
        return n
