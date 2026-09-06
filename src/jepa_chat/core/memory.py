"""MemoryStore: SQLite persistence for sessions, turns, latents and metrics,
plus an in-memory cosine index over normalised turn latents."""

from __future__ import annotations

import hashlib
import math
import re
import sqlite3
import threading
import time
import uuid
from dataclasses import dataclass
from pathlib import Path

import numpy as np

_SCHEMA = """
CREATE TABLE IF NOT EXISTS sessions(
  id TEXT PRIMARY KEY, title TEXT, source TEXT, mouth TEXT,
  created_ts REAL, last_ts REAL);
CREATE TABLE IF NOT EXISTS turns(
  id INTEGER PRIMARY KEY AUTOINCREMENT, session_id TEXT NOT NULL, idx INTEGER NOT NULL,
  role TEXT, mouth TEXT, source TEXT, text TEXT, ts REAL,
  latent BLOB, enc_version INTEGER DEFAULT 0, surprise REAL, pred_latent BLOB, text_hash TEXT);
CREATE INDEX IF NOT EXISTS turns_session ON turns(session_id, idx);
CREATE INDEX IF NOT EXISTS turns_hash ON turns(text_hash, ts);
CREATE TABLE IF NOT EXISTS metrics(
  step INTEGER, ts REAL, loss1 REAL, loss2 REAL, latent_std REAL, ms REAL);
"""


_TOKEN = re.compile(r"[a-zA-Zа-яА-ЯёЁ0-9_]{2,}")


def tokenize(text: str) -> list[str]:
    return [t.lower() for t in _TOKEN.findall(text)]


class BM25:
    """Tiny in-memory BM25 over turn texts (lexical half of hybrid recall)."""

    def __init__(self, k1: float = 1.2, b: float = 0.75):
        self.k1, self.b = k1, b
        self.df: dict[str, int] = {}
        self.docs: dict[int, dict[str, int]] = {}
        self.lens: dict[int, int] = {}
        self.total_len = 0

    def add(self, doc_id: int, text: str) -> None:
        if doc_id in self.docs:
            self.remove(doc_id)
        toks = tokenize(text)
        tf: dict[str, int] = {}
        for t in toks:
            tf[t] = tf.get(t, 0) + 1
        for t in tf:
            self.df[t] = self.df.get(t, 0) + 1
        self.docs[doc_id] = tf
        self.lens[doc_id] = len(toks)
        self.total_len += len(toks)

    def remove(self, doc_id: int) -> None:
        tf = self.docs.pop(doc_id, None)
        if tf is None:
            return
        for t in tf:
            self.df[t] = self.df.get(t, 1) - 1
        self.total_len -= self.lens.pop(doc_id, 0)

    def scores(self, query: str) -> dict[int, float]:
        n = len(self.docs)
        if n == 0:
            return {}
        avg = self.total_len / n
        out: dict[int, float] = {}
        for q in set(tokenize(query)):
            df = self.df.get(q, 0)
            if df == 0:
                continue
            idf = math.log(1 + (n - df + 0.5) / (df + 0.5))
            for doc_id, tf in self.docs.items():
                f = tf.get(q)
                if not f:
                    continue
                denom = f + self.k1 * (1 - self.b + self.b * self.lens[doc_id] / max(avg, 1e-6))
                out[doc_id] = out.get(doc_id, 0.0) + idf * f * (self.k1 + 1) / denom
        return out


def text_hash(text: str) -> str:
    return hashlib.sha1(text.strip().encode("utf-8")).hexdigest()


def norm_for_index(z: np.ndarray, mu: np.ndarray | None = None, sd: np.ndarray | None = None) -> np.ndarray:
    """Whiten against the memory's global per-dimension mean/std, then L2-normalise.

    Without whitening, encoder latents that share one dominant direction all have cosine ~1.
    Falls back to per-row standardisation when no global stats exist yet.
    """
    z = np.asarray(z, dtype=np.float32)
    if mu is not None and sd is not None:
        z = (z - mu) / sd
    else:
        z = (z - z.mean(-1, keepdims=True)) / (z.std(-1, keepdims=True) + 1e-5)
    return z / (np.linalg.norm(z, axis=-1, keepdims=True) + 1e-8)


@dataclass
class TurnRow:
    id: int
    session_id: str
    idx: int
    role: str
    mouth: str
    source: str
    text: str
    ts: float | None
    latent: np.ndarray | None
    enc_version: int
    surprise: float | None
    pred_latent: np.ndarray | None


@dataclass
class SessionRow:
    id: str
    title: str
    source: str
    mouth: str
    created_ts: float | None
    last_ts: float | None
    n_turns: int


@dataclass
class Hit:
    turn: TurnRow
    score: float


def _blob(a: np.ndarray | None) -> bytes | None:
    return None if a is None else np.asarray(a, dtype=np.float32).tobytes()


def _unblob(b: bytes | None, dim: int) -> np.ndarray | None:
    if b is None:
        return None
    a = np.frombuffer(b, dtype=np.float32)
    return a if a.shape[0] == dim else None


class MemoryStore:
    def __init__(self, path: str | Path, dim: int):
        self.path = Path(path).expanduser()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.dim = dim
        self._lock = threading.RLock()
        self._db = sqlite3.connect(str(self.path), check_same_thread=False)
        self._db.row_factory = sqlite3.Row
        self._db.execute("PRAGMA journal_mode=WAL")
        self._db.executescript(_SCHEMA)
        self._db.commit()
        self._ids: list[int] = []
        self._lens: list[int] = []
        self._raw = np.zeros((0, dim), np.float32)
        self._index = np.zeros((0, dim), np.float32)
        self._pos: dict[int, int] = {}
        self._mu: np.ndarray | None = None
        self._sd: np.ndarray | None = None
        self._bucket_mu: dict[int, np.ndarray] = {}
        self._adds_since_refit = 0
        self._bm25 = BM25()
        self._rebuild_index()

    # ------------------------------------------------------------ index
    @staticmethod
    def length_bucket(n_chars: int) -> int:
        return int(np.log2(max(n_chars, 1)))

    def _rebuild_index(self) -> None:
        rows = self._db.execute(
            "SELECT id, latent, text, length(text) AS n FROM turns WHERE latent IS NOT NULL ORDER BY id"
        ).fetchall()
        ids, vecs, lens = [], [], []
        self._bm25 = BM25()
        for r in rows:
            v = _unblob(r["latent"], self.dim)
            if v is not None:
                ids.append(int(r["id"]))
                vecs.append(v)
                lens.append(int(r["n"] or 0))
                self._bm25.add(int(r["id"]), r["text"] or "")
        self._ids = ids
        self._lens = lens
        self._raw = np.stack(vecs) if vecs else np.zeros((0, self.dim), np.float32)
        self._pos = {tid: i for i, tid in enumerate(ids)}
        self._refit()

    def _refit(self) -> None:
        """Recompute whitening stats from all raw latents and re-normalise the index.

        Byte-level mean-pooled latents carry a strong text-length signature, so each latent is
        first centred against the mean of turns in its log2-length bucket (nuisance removal),
        then whitened by global per-dimension std.
        """
        n = self._raw.shape[0]
        self._bucket_mu = {}
        if n >= 8:
            buckets = np.array([self.length_bucket(x) for x in self._lens])
            for b in np.unique(buckets):
                sel = buckets == b
                if sel.sum() >= 5:
                    self._bucket_mu[int(b)] = self._raw[sel].mean(0)
            self._mu = self._raw.mean(0)
            centred = self._centre(self._raw, self._lens)
            self._sd = centred.std(0) + 1e-5
            self._index = norm_for_index(centred, np.zeros(self.dim, np.float32), self._sd)
        else:
            self._mu = self._sd = None
            self._index = norm_for_index(self._raw) if n else self._raw.copy()
        self._adds_since_refit = 0

    def _centre(self, z: np.ndarray, lens: list[int]) -> np.ndarray:
        if self._mu is None:
            return z
        out = np.empty_like(z, dtype=np.float32)
        for i, n in enumerate(lens):
            out[i] = z[i] - self._bucket_mu.get(self.length_bucket(n), self._mu)
        return out

    def _transform(self, z: np.ndarray, n_chars: int) -> np.ndarray:
        if self._mu is None:
            return norm_for_index(z)
        c = self._centre(np.asarray(z, np.float32)[None], [n_chars])
        return norm_for_index(c, np.zeros(self.dim, np.float32), self._sd)[0]

    def _index_add(self, tid: int, latent: np.ndarray, n_chars: int) -> None:
        raw = np.asarray(latent, np.float32)[None]
        v = self._transform(raw[0], n_chars)[None]
        if tid in self._pos:
            self._raw[self._pos[tid]] = raw[0]
            self._index[self._pos[tid]] = v[0]
            self._lens[self._pos[tid]] = n_chars
        else:
            self._pos[tid] = len(self._ids)
            self._ids.append(tid)
            self._lens.append(n_chars)
            self._raw = np.concatenate([self._raw, raw], 0)
            self._index = np.concatenate([self._index, v], 0)
        self._adds_since_refit += 1
        n = self._raw.shape[0]
        if self._mu is None and n >= 8 or self._adds_since_refit >= max(64, n // 10):
            self._refit()

    # --------------------------------------------------------- sessions
    def create_session(
        self, session_id: str | None = None, title: str = "", source: str = "live", mouth: str = ""
    ) -> SessionRow:
        with self._lock:
            sid = session_id or f"s_{time.strftime('%Y%m%d-%H%M%S')}_{uuid.uuid4().hex[:4]}"
            now = time.time()
            self._db.execute(
                "INSERT OR IGNORE INTO sessions(id,title,source,mouth,created_ts,last_ts) VALUES(?,?,?,?,?,?)",
                (sid, title, source, mouth, now, now),
            )
            self._db.commit()
            return self.get_session(sid)  # type: ignore[return-value]

    def get_session(self, session_id: str) -> SessionRow | None:
        with self._lock:
            r = self._db.execute(
                "SELECT s.*, (SELECT COUNT(*) FROM turns t WHERE t.session_id=s.id) AS n FROM sessions s WHERE id=?",
                (session_id,),
            ).fetchone()
            return self._session_row(r) if r else None

    def list_sessions(self, limit: int = 50) -> list[SessionRow]:
        with self._lock:
            rows = self._db.execute(
                "SELECT s.*, (SELECT COUNT(*) FROM turns t WHERE t.session_id=s.id) AS n "
                "FROM sessions s ORDER BY COALESCE(last_ts, created_ts, 0) DESC LIMIT ?",
                (limit,),
            ).fetchall()
            return [self._session_row(r) for r in rows]

    def touch_session(self, session_id: str, ts: float, mouth: str = "") -> None:
        with self._lock:
            if mouth:
                self._db.execute("UPDATE sessions SET last_ts=?, mouth=? WHERE id=?", (ts, mouth, session_id))
            else:
                self._db.execute("UPDATE sessions SET last_ts=? WHERE id=?", (ts, session_id))
            self._db.commit()

    @staticmethod
    def _session_row(r: sqlite3.Row) -> SessionRow:
        return SessionRow(
            id=r["id"],
            title=r["title"] or "",
            source=r["source"] or "",
            mouth=r["mouth"] or "",
            created_ts=r["created_ts"],
            last_ts=r["last_ts"],
            n_turns=int(r["n"]),
        )

    # ------------------------------------------------------------ turns
    def add_turn(
        self,
        session_id: str,
        role: str,
        mouth: str,
        source: str,
        text: str,
        ts: float | None,
        latent: np.ndarray | None,
        enc_version: int,
        surprise: float | None = None,
        pred_latent: np.ndarray | None = None,
    ) -> TurnRow:
        with self._lock:
            idx = self._db.execute(
                "SELECT COALESCE(MAX(idx),-1)+1 FROM turns WHERE session_id=?", (session_id,)
            ).fetchone()[0]
            cur = self._db.execute(
                "INSERT INTO turns(session_id,idx,role,mouth,source,text,ts,latent,enc_version,surprise,"
                "pred_latent,text_hash) VALUES(?,?,?,?,?,?,?,?,?,?,?,?)",
                (
                    session_id,
                    idx,
                    role,
                    mouth,
                    source,
                    text,
                    ts,
                    _blob(latent),
                    enc_version,
                    surprise,
                    _blob(pred_latent),
                    text_hash(text),
                ),
            )
            tid = int(cur.lastrowid)
            self._db.execute(
                "UPDATE sessions SET last_ts=MAX(COALESCE(last_ts,0),?) WHERE id=?",
                (ts or 0.0, session_id),
            )
            self._db.commit()
            if latent is not None:
                self._index_add(tid, latent, len(text))
                self._bm25.add(tid, text)
            return TurnRow(
                tid,
                session_id,
                idx,
                role,
                mouth,
                source,
                text,
                ts,
                latent,
                enc_version,
                surprise,
                pred_latent,
            )

    def _row(self, r: sqlite3.Row) -> TurnRow:
        return TurnRow(
            id=int(r["id"]),
            session_id=r["session_id"],
            idx=int(r["idx"]),
            role=r["role"] or "",
            mouth=r["mouth"] or "",
            source=r["source"] or "",
            text=r["text"] or "",
            ts=r["ts"],
            latent=_unblob(r["latent"], self.dim),
            enc_version=int(r["enc_version"] or 0),
            surprise=r["surprise"],
            pred_latent=_unblob(r["pred_latent"], self.dim),
        )

    def get_turns(self, ids: list[int]) -> list[TurnRow]:
        if not ids:
            return []
        with self._lock:
            q = ",".join("?" * len(ids))
            rows = self._db.execute(f"SELECT * FROM turns WHERE id IN ({q})", ids).fetchall()
            by_id = {int(r["id"]): self._row(r) for r in rows}
            return [by_id[i] for i in ids if i in by_id]

    def session_turns(self, session_id: str, limit: int | None = None) -> list[TurnRow]:
        with self._lock:
            if limit is None:
                rows = self._db.execute("SELECT * FROM turns WHERE session_id=? ORDER BY idx", (session_id,)).fetchall()
            else:
                rows = self._db.execute(
                    "SELECT * FROM (SELECT * FROM turns WHERE session_id=? ORDER BY idx DESC LIMIT ?) ORDER BY idx",
                    (session_id, limit),
                ).fetchall()
            return [self._row(r) for r in rows]

    def recent_turns(self, n: int) -> list[TurnRow]:
        with self._lock:
            rows = self._db.execute("SELECT * FROM turns ORDER BY id DESC LIMIT ?", (n,)).fetchall()
            return [self._row(r) for r in reversed(rows)]

    def find_recent_duplicate(self, text: str, role: str, within_s: float, now: float | None = None) -> TurnRow | None:
        now = now or time.time()
        with self._lock:
            r = self._db.execute(
                "SELECT * FROM turns WHERE text_hash=? AND role=? AND ts>=? ORDER BY id DESC LIMIT 1",
                (text_hash(text), role, now - within_s),
            ).fetchone()
            return self._row(r) if r else None

    def update_surprise(self, turn_id: int, surprise: float | None, pred_latent: np.ndarray | None) -> None:
        with self._lock:
            self._db.execute(
                "UPDATE turns SET surprise=?, pred_latent=? WHERE id=?",
                (surprise, _blob(pred_latent), turn_id),
            )
            self._db.commit()

    # ----------------------------------------------------------- search
    def search(
        self,
        query: np.ndarray,
        k: int = 5,
        exclude_session: str | None = None,
        role: str | None = None,
        exclude_ids: tuple[int, ...] = (),
        query_chars: int | None = None,
        min_chars: int = 0,
        query_text: str | None = None,
        lexical_weight: float = 0.5,
    ) -> list[Hit]:
        """Hybrid recall: latent cosine blended with BM25 when query_text is given.

        Both parts are min-max normalised over the candidate pool before blending, so the
        returned score is in [0, 1] and comparable across queries.
        """
        with self._lock:
            if self._index.shape[0] == 0:
                return []
            q = self._transform(np.asarray(query, np.float32), query_chars or 256)
            scores = self._index @ q
            if query_text and lexical_weight > 0:
                lex = self._bm25.scores(query_text)
                if lex:
                    lex_arr = np.array([lex.get(tid, 0.0) for tid in self._ids], dtype=np.float32)
                    lex_arr /= max(float(lex_arr.max()), 1e-6)
                    cos_n = (scores - scores.min()) / max(float(scores.max() - scores.min()), 1e-6)
                    scores = (1 - lexical_weight) * cos_n + lexical_weight * lex_arr
            order = np.argsort(-scores)
            hits: list[Hit] = []
            cand = [int(self._ids[i]) for i in order[: max(k * 8, 64)]]
            rows = {t.id: t for t in self.get_turns(cand)}
            for i in order[: max(k * 8, 64)]:
                tid = int(self._ids[i])
                t = rows.get(tid)
                if t is None or tid in exclude_ids:
                    continue
                if exclude_session and t.session_id == exclude_session:
                    continue
                if role and t.role != role:
                    continue
                if len(t.text) < min_chars:
                    continue
                hits.append(Hit(t, float(scores[i])))
                if len(hits) >= k:
                    break
            return hits

    def random_latents(self, n: int, rng: np.random.Generator) -> np.ndarray:
        with self._lock:
            if self._index.shape[0] == 0:
                return np.zeros((0, self.dim), np.float32)
            idx = rng.integers(0, self._index.shape[0], size=min(n, self._index.shape[0]))
            ids = [self._ids[i] for i in idx]
            rows = self.get_turns(ids)
            return (
                np.stack([r.latent for r in rows if r.latent is not None])
                if rows
                else np.zeros((0, self.dim), np.float32)
            )

    # --------------------------------------------------------- training
    def sample_windows(
        self,
        n_sessions: int,
        window: int,
        rng: np.random.Generator,
        recent_frac: float = 0.0,
        recent_since_ts: float | None = None,
    ) -> list[list[TurnRow]]:
        with self._lock:
            all_ids = [
                r["session_id"]
                for r in self._db.execute(
                    "SELECT session_id FROM turns GROUP BY session_id HAVING COUNT(*)>=2"
                ).fetchall()
            ]
            if not all_ids:
                return []
            recent_ids = all_ids
            if recent_since_ts is not None:
                recent_ids = [
                    r["id"]
                    for r in self._db.execute("SELECT id FROM sessions WHERE last_ts>=?", (recent_since_ts,)).fetchall()
                    if r["id"] in set(all_ids)
                ] or all_ids
            out: list[list[TurnRow]] = []
            for _ in range(n_sessions):
                pool = recent_ids if rng.random() < recent_frac else all_ids
                sid = pool[int(rng.integers(0, len(pool)))]
                turns = self.session_turns(sid)
                if len(turns) > window:
                    start = int(rng.integers(0, len(turns) - window + 1))
                    turns = turns[start : start + window]
                out.append(turns)
            return out

    def max_enc_version(self) -> int:
        with self._lock:
            v = self._db.execute("SELECT MAX(enc_version) FROM turns").fetchone()[0]
            return int(v or 0)

    def stale_turn_ids(self, min_version: int, limit: int = 256) -> list[int]:
        with self._lock:
            rows = self._db.execute(
                "SELECT id FROM turns WHERE enc_version<? ORDER BY id LIMIT ?", (min_version, limit)
            ).fetchall()
            return [int(r["id"]) for r in rows]

    def update_latents(self, ids: list[int], latents: np.ndarray, version: int) -> None:
        with self._lock:
            for tid, z in zip(ids, latents, strict=True):
                self._db.execute("UPDATE turns SET latent=?, enc_version=? WHERE id=?", (_blob(z), version, tid))
                n = self._lens[self._pos[tid]] if tid in self._pos else 0

                self._index_add(tid, z, n)
            self._db.commit()
            self._refit()

    # ---------------------------------------------------------- metrics
    def log_metric(self, step: int, loss1: float, loss2: float, latent_std: float, ms: float) -> None:
        with self._lock:
            self._db.execute(
                "INSERT INTO metrics(step,ts,loss1,loss2,latent_std,ms) VALUES(?,?,?,?,?,?)",
                (step, time.time(), loss1, loss2, latent_std, ms),
            )
            self._db.commit()

    def recent_metrics(self, n: int = 100) -> list[dict]:
        with self._lock:
            rows = self._db.execute("SELECT * FROM metrics ORDER BY step DESC LIMIT ?", (n,)).fetchall()
            return [dict(r) for r in reversed(rows)]

    def count_turns(self) -> int:
        with self._lock:
            return int(self._db.execute("SELECT COUNT(*) FROM turns").fetchone()[0])

    def count_sessions(self) -> int:
        with self._lock:
            return int(self._db.execute("SELECT COUNT(*) FROM sessions").fetchone()[0])

    def close(self) -> None:
        with self._lock:
            self._db.close()
