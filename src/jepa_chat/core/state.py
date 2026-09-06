"""In-process state shared by the service, the engine and the HTTP layer."""

from __future__ import annotations

import asyncio
import collections
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np

Event = dict


class EventBus:
    """Thread-safe fan-out. Async subscribers get an asyncio.Queue; sync listeners get callbacks."""

    def __init__(self, history: int = 200):
        self._loop: asyncio.AbstractEventLoop | None = None
        self._queues: list[asyncio.Queue] = []
        self._listeners: list[Callable[[Event], None]] = []
        self._lock = threading.Lock()
        self.history: collections.deque[Event] = collections.deque(maxlen=history)

    def bind_loop(self, loop: asyncio.AbstractEventLoop) -> None:
        self._loop = loop

    def subscribe(self, maxsize: int = 500) -> asyncio.Queue:
        q: asyncio.Queue = asyncio.Queue(maxsize=maxsize)
        with self._lock:
            self._queues.append(q)
        return q

    def unsubscribe(self, q: asyncio.Queue) -> None:
        with self._lock:
            if q in self._queues:
                self._queues.remove(q)

    def add_listener(self, fn: Callable[[Event], None]) -> None:
        with self._lock:
            self._listeners.append(fn)

    def publish(self, kind: str, **data: object) -> Event:
        ev: Event = {"type": kind, "ts": time.time(), **data}
        with self._lock:
            self.history.append(ev)
            queues = list(self._queues)
            listeners = list(self._listeners)
        for fn in listeners:
            try:
                fn(ev)
            except Exception:  # noqa: BLE001 - listeners must never break the publisher
                pass
        if self._loop is not None:
            for q in queues:
                self._loop.call_soon_threadsafe(self._offer, q, ev)
        return ev

    @staticmethod
    def _offer(q: asyncio.Queue, ev: Event) -> None:
        try:
            q.put_nowait(ev)
        except asyncio.QueueFull:
            pass


@dataclass
class SessionState:
    session_id: str
    n_turns: int = 0
    mouths: set[str] = field(default_factory=set)
    surprises: list[float] = field(default_factory=list)
    h_last: np.ndarray | None = None
    pending_pred: dict[int, np.ndarray] = field(default_factory=dict)  # role_id -> predicted latent
    last_role: str | None = None
    last_ts: float | None = None


@dataclass
class TrainingStats:
    step: int = 0
    loss1: float | None = None
    loss2: float | None = None
    latent_std: float | None = None
    pretrain_latent_std: float | None = None
    steps_per_s: float = 0.0
    paused: bool = False
    enc_version: int = 0
    device: str = "cpu"
    warning: str | None = None
    last_checkpoint_ts: float | None = None

    def to_dict(self) -> dict:
        return dict(self.__dict__)


@dataclass
class CoreState:
    started_ts: float = field(default_factory=time.time)
    active_session_id: str | None = None
    sessions: dict[str, SessionState] = field(default_factory=dict)
    training: TrainingStats = field(default_factory=TrainingStats)
    lock: threading.RLock = field(default_factory=threading.RLock)

    def session(self, session_id: str) -> SessionState:
        with self.lock:
            if session_id not in self.sessions:
                self.sessions[session_id] = SessionState(session_id)
            return self.sessions[session_id]
