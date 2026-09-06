"""ModelThread: the single owner of the accelerator models, running online training,
publishing CPU inference snapshots, re-embedding stale memory and checkpointing."""

from __future__ import annotations

import copy
import queue
import threading
import time
from dataclasses import dataclass

import numpy as np
import torch

from jepa_chat.config import Config
from jepa_chat.core.checkpoint import save_checkpoint
from jepa_chat.core.memory import MemoryStore, TurnRow
from jepa_chat.core.state import CoreState, EventBus
from jepa_chat.data.masking import sample_masks
from jepa_chat.data.tokenizer import ByteTokenizer
from jepa_chat.data.turns import role_id, truncate_turn
from jepa_chat.models.conversation import ConversationPredictor, normalize_latent
from jepa_chat.models.jepa import TextJEPA


@dataclass
class InferenceSnapshot:
    """CPU, eval-mode copies used by every request path. Immutable once built."""

    version: int
    level1: TextJEPA
    level2: ConversationPredictor
    cfg: Config
    tokenizer: ByteTokenizer

    @torch.no_grad()
    def embed(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.cfg.model.d_model), np.float32)
        cc = self.cfg.conversation
        chunk = [truncate_turn(t, cc.head_bytes, cc.tail_bytes) for t in texts]
        ids, attn = self.tokenizer.batch_encode(chunk, self.cfg.model.max_len)
        return self.level1.embed(ids, attn).float().numpy()

    @torch.no_grad()
    def encode_session(self, latents: np.ndarray, roles: list[int]) -> tuple[np.ndarray, dict[int, np.ndarray]]:
        """Returns (h_last, {next_role: predicted normalised latent}) for a turn sequence."""
        k = self.cfg.conversation.max_turns
        z = torch.from_numpy(np.asarray(latents[-k:], np.float32))[None]
        r = torch.tensor(roles[-k:], dtype=torch.long)[None]
        attn = torch.ones_like(r, dtype=torch.bool)
        h = self.level2.encode(z, r, attn)[0, -1]
        preds = {rid: self.level2.predict_from_state(h, torch.tensor(rid)).numpy() for rid in (0, 1)}
        return h.numpy(), preds

    @torch.no_grad()
    def rollout(self, latents: np.ndarray, roles: list[int], future_roles: list[int]) -> np.ndarray:
        k = self.cfg.conversation.max_turns
        z = torch.from_numpy(np.asarray(latents[-k:], np.float32))[None]
        r = torch.tensor(roles[-k:], dtype=torch.long)[None]
        attn = torch.ones_like(r, dtype=torch.bool)
        return self.level2.rollout(z, r, attn, future_roles)[0].numpy()


def surprise_ratio(pred: np.ndarray, actual: np.ndarray, baseline: np.ndarray) -> tuple[float, float]:
    """(ratio, cosine). ratio = err(pred, actual) / mean err(pred, random). 0 = perfect, ~1 = chance."""
    a = normalize_latent(torch.from_numpy(np.asarray(actual, np.float32))).numpy()
    p = np.asarray(pred, np.float32)
    err = float(((p - a) ** 2).sum())
    cos = float(p @ a / (np.linalg.norm(p) * np.linalg.norm(a) + 1e-8))
    if baseline.shape[0] == 0:
        return err, cos
    b = normalize_latent(torch.from_numpy(np.asarray(baseline, np.float32))).numpy()
    base = float(((p[None] - b) ** 2).sum(-1).mean())
    return err / max(base, 1e-6), cos


class ModelThread(threading.Thread):
    def __init__(
        self,
        cfg: Config,
        level1: TextJEPA,
        level2: ConversationPredictor,
        memory: MemoryStore,
        bus: EventBus,
        state: CoreState,
        device: torch.device,
        checkpoint_path: str,
        step: int = 0,
        enc_version: int = 0,
        opt1_state: dict | None = None,
        opt2_state: dict | None = None,
        pretrain_latent_std: float | None = None,
        train: bool = True,
    ):
        super().__init__(name="jepa-model", daemon=True)
        self.cfg = cfg
        self.core = cfg.core
        self.device = device
        self.memory = memory
        self.bus = bus
        self.state = state
        self.checkpoint_path = checkpoint_path
        self.tokenizer = ByteTokenizer()
        self.level1 = level1.to(device)
        self.level2 = level2.to(device)
        p1 = [p for p in self.level1.parameters() if p.requires_grad]
        self.opt1 = torch.optim.AdamW(p1, lr=self.core.online_lr, weight_decay=0.05)
        self.opt2 = torch.optim.AdamW(self.level2.parameters(), lr=self.core.online_lr, weight_decay=0.05)
        if opt1_state:
            try:
                self.opt1.load_state_dict(opt1_state)
            except (ValueError, KeyError):
                pass
        if opt2_state:
            try:
                self.opt2.load_state_dict(opt2_state)
            except (ValueError, KeyError):
                pass
        self.step = step
        self.enc_version = enc_version
        self.pretrain_latent_std = pretrain_latent_std
        self.rng = np.random.default_rng(step)
        self._snapshot_lock = threading.Lock()
        self._snapshot: InferenceSnapshot | None = None
        self._pause = threading.Event()
        if not train:
            self._pause.set()
        self._stop_evt = threading.Event()
        self._jobs: queue.Queue = queue.Queue()
        self._last_event_ts = 0.0
        self._step_times: list[float] = []
        # learning budget: steps are earned by new experience (live turns) plus a slow idle trickle.
        # Replaying a fixed corpus for hours degrades the latents (measured: effective rank 18 -> 12
        # over 82k idle steps), so we do not train unboundedly.
        self._budget = 0.0
        self._last_budget_ts = time.time()
        self._budget_lock = threading.Lock()
        st = self.state.training
        st.step, st.enc_version, st.device = step, enc_version, str(device)
        st.pretrain_latent_std, st.paused = pretrain_latent_std, self._pause.is_set()
        self.publish_snapshot()

    # ------------------------------------------------------------ snapshot
    def publish_snapshot(self) -> InferenceSnapshot:
        l1 = copy.deepcopy(self.level1).cpu().eval()
        l2 = copy.deepcopy(self.level2).cpu().eval()
        snap = InferenceSnapshot(self.enc_version, l1, l2, self.cfg, self.tokenizer)
        with self._snapshot_lock:
            self._snapshot = snap
        self.state.training.enc_version = self.enc_version
        self.bus.publish("snapshot", version=self.enc_version)
        return snap

    def snapshot(self) -> InferenceSnapshot:
        with self._snapshot_lock:
            assert self._snapshot is not None
            return self._snapshot

    # ------------------------------------------------------------ budget
    def notify_new_turn(self, n: int = 1) -> None:
        """Called by the service whenever a live turn is observed: earns training steps."""
        with self._budget_lock:
            self._budget = min(self._budget + n * self.core.steps_per_new_turn, self.core.max_step_budget)

    def _take_step_budget(self) -> bool:
        now = time.time()
        with self._budget_lock:
            self._budget = min(
                self._budget + (now - self._last_budget_ts) * self.core.idle_steps_per_min / 60.0,
                self.core.max_step_budget,
            )
            self._last_budget_ts = now
            if self._budget >= 1.0:
                self._budget -= 1.0
                return True
            return False

    @property
    def step_budget(self) -> float:
        with self._budget_lock:
            return self._budget

    # ------------------------------------------------------------ control
    @property
    def paused(self) -> bool:
        return self._pause.is_set()

    def pause(self) -> None:
        self._pause.set()
        self.state.training.paused = True
        self.bus.publish("training", paused=True)

    def resume(self) -> None:
        self._pause.clear()
        self.state.training.paused = False
        self.state.training.warning = None
        self.bus.publish("training", paused=False)

    def request_checkpoint(self) -> None:
        self._jobs.put(("checkpoint", None))

    def request_reembed(self) -> None:
        self._jobs.put(("reembed", None))

    def stop(self) -> None:
        self._stop_evt.set()

    def embed_bulk(self, texts: list[str], timeout: float = 600.0) -> np.ndarray:
        """Embed on the accelerator from any thread; the model thread executes it between steps."""
        done = threading.Event()
        box: dict[str, np.ndarray] = {}
        self._jobs.put(("embed", (texts, box, done)))
        if not done.wait(timeout):
            raise TimeoutError("model thread did not answer embed request")
        return box["out"]

    # ------------------------------------------------------------ helpers
    @torch.no_grad()
    def _embed_on_device(self, texts: list[str], batch: int = 128) -> np.ndarray:
        cc = self.cfg.conversation
        self.level1.eval()
        out = []
        for i in range(0, len(texts), batch):
            chunk = [truncate_turn(t, cc.head_bytes, cc.tail_bytes) for t in texts[i : i + batch]]
            ids, attn = self.tokenizer.batch_encode(chunk, self.cfg.model.max_len)
            out.append(self.level1.embed(ids.to(self.device), attn.to(self.device)).float().cpu().numpy())
        self.level1.train()
        return np.concatenate(out) if out else np.zeros((0, self.cfg.model.d_model), np.float32)

    def _drain_jobs(self) -> None:
        while True:
            try:
                kind, payload = self._jobs.get_nowait()
            except queue.Empty:
                return
            if kind == "embed":
                texts, box, done = payload
                try:
                    box["out"] = self._embed_on_device(texts)
                finally:
                    done.set()
            elif kind == "checkpoint":
                self.checkpoint()
            elif kind == "reembed":
                self.reembed(force=True)

    def checkpoint(self) -> None:
        path = save_checkpoint(
            self.checkpoint_path,
            self.cfg,
            self.level1,
            self.level2,
            self.opt1,
            self.opt2,
            step=self.step,
            enc_version=self.enc_version,
            pretrain_latent_std=self.pretrain_latent_std,
        )
        self.state.training.last_checkpoint_ts = time.time()
        self.bus.publish("checkpoint", path=str(path), step=self.step)

    def reembed(self, force: bool = False, lag: int = 2) -> int:
        """Re-embed stale turns. force=True re-embeds *every* turn with the current encoder
        (used after swapping in a new checkpoint), in batches, bumping enc_version first so
        the stored version reflects the new encoder."""
        if force:
            self.enc_version += 1
            self.publish_snapshot()
        min_version = self.enc_version - (0 if force else lag)
        total = 0
        while True:
            ids = self.memory.stale_turn_ids(min_version, limit=512)
            if not ids:
                break
            rows = self.memory.get_turns(ids)
            z = self._embed_on_device([r.text for r in rows])
            self.memory.update_latents([r.id for r in rows], z, self.enc_version)
            total += len(rows)
            if not force:
                break
        if total:
            self.bus.publish("reembed", n=total, version=self.enc_version)
        return total

    # ------------------------------------------------------------ training
    def _batches(self) -> tuple[tuple[torch.Tensor, ...] | None, tuple[torch.Tensor, ...] | None]:
        windows = self.memory.sample_windows(
            self.core.online_batch_sessions,
            self.core.online_window,
            self.rng,
            recent_frac=self.core.live_ratio,
            recent_since_ts=time.time() - 3600,
        )
        windows = [w for w in windows if len(w) >= 2 and all(t.latent is not None for t in w)]
        if not windows:
            return None, None
        # Level 1: byte batch over the turns' text
        texts = [t.text for w in windows for t in w]
        pick = self.rng.choice(len(texts), size=min(32, len(texts)), replace=False)
        cc = self.cfg.conversation
        chunk = [truncate_turn(texts[i], cc.head_bytes, cc.tail_bytes) for i in pick]
        ids, attn = self.tokenizer.batch_encode(chunk, self.cfg.train.seq_len)
        masks = sample_masks(attn, self.cfg.mask)
        b1 = (ids, attn, masks.context_mask, masks.target_mask)
        # Level 2: stored latents
        k = self.core.online_window
        dim = self.cfg.model.d_model
        z = torch.zeros((len(windows), k, dim))
        roles = torch.full((len(windows), k), 2, dtype=torch.long)
        a = torch.zeros((len(windows), k), dtype=torch.bool)
        for i, w in enumerate(windows):
            n = min(len(w), k)
            z[i, :n] = torch.from_numpy(np.stack([t.latent for t in w[:n]]))
            roles[i, :n] = torch.tensor([role_id(t.role) for t in w[:n]])
            a[i, :n] = True
        return b1, (z, roles, a)

    def train_step(self) -> dict[str, float] | None:
        b1, b2 = self._batches()
        if b1 is None or b2 is None:
            return None
        t0 = time.time()
        d = self.device
        self.level1.train()
        out = self.level1(*(x.to(d) for x in b1))
        self.opt1.zero_grad(set_to_none=True)
        out.loss.backward()
        torch.nn.utils.clip_grad_norm_([p for p in self.level1.parameters() if p.requires_grad], 1.0)
        self.opt1.step()
        self.level1.update_target_encoder(self.core.ema_momentum)

        self.level2.train()
        loss2, stats2 = self.level2.loss(*(x.to(d) for x in b2))
        self.opt2.zero_grad(set_to_none=True)
        loss2.backward()
        torch.nn.utils.clip_grad_norm_(self.level2.parameters(), 1.0)
        self.opt2.step()
        ms = (time.time() - t0) * 1000
        return {
            "loss1": out.loss.item(),
            "latent_std": out.latent_std.item(),
            "loss2": stats2["loss2"],
            "ratio": stats2["ratio"],
            "ms": ms,
        }

    def run(self) -> None:
        while not self._stop_evt.is_set():
            self._drain_jobs()
            if self._pause.is_set() or not self._take_step_budget():
                time.sleep(0.2)
                continue
            m = self.train_step()
            if m is None:
                time.sleep(1.0)
                continue
            self.step += 1
            st = self.state.training
            st.step, st.loss1, st.loss2, st.latent_std = (
                self.step,
                m["loss1"],
                m["loss2"],
                m["latent_std"],
            )
            self._step_times.append(time.time())
            self._step_times = self._step_times[-50:]
            if len(self._step_times) > 1:
                st.steps_per_s = (len(self._step_times) - 1) / max(self._step_times[-1] - self._step_times[0], 1e-6)
            if self.step % 10 == 0:
                self.memory.log_metric(self.step, m["loss1"], m["loss2"], m["latent_std"], m["ms"])
            now = time.time()
            if now - self._last_event_ts > 0.25:
                self._last_event_ts = now
                self.bus.publish("train_step", step=self.step, steps_per_s=st.steps_per_s, **m)
            floor = self.core.collapse_floor * (self.pretrain_latent_std or 0.0)
            if self.pretrain_latent_std and m["latent_std"] < floor:
                st.warning = f"latent_std {m['latent_std']:.3f} below floor {floor:.3f}: training paused"
                self.bus.publish("warning", message=st.warning)
                self.pause()
                continue
            if self.step % self.core.snapshot_every == 0:
                self.enc_version += 1
                self.publish_snapshot()
            if self.step % self.core.reembed_every == 0:
                self.reembed()
            if self.step % self.core.checkpoint_every == 0:
                self.checkpoint()
        self.checkpoint()


def build_row_latents(rows: list[TurnRow]) -> np.ndarray:
    return np.stack([r.latent for r in rows if r.latent is not None])
