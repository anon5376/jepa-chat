"""jepa-tui: live terminal dashboard for the ChatJEPA core."""

from __future__ import annotations

import argparse
import asyncio
import time
from datetime import datetime

from rich.text import Text
from textual import work
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Vertical
from textual.widgets import DataTable, Footer, Header, RichLog, Sparkline, Static
from textual_plotext import PlotextPlot

from jepa_chat.tui.client import DEFAULT_BASE, CoreClient


def _ago(ts: float | None) -> str:
    if not ts:
        return "-"
    d = time.time() - ts
    if d < 90:
        return f"{int(d)}s"
    if d < 5400:
        return f"{int(d / 60)}m"
    if d < 172800:
        return f"{int(d / 3600)}h"
    return datetime.fromtimestamp(ts).strftime("%b %d")


class LatentMap(PlotextPlot):
    def update_map(self, m: dict) -> None:
        plt = self.plt
        plt.clear_data()
        plt.clear_figure()
        plt.theme("dark")
        pts = m.get("points") or []
        if len(pts) < 2:
            plt.title("latent map (waiting for turns)")
            self.refresh()
            return
        users = [p for p in pts if p["role"] == "user"]
        asst = [p for p in pts if p["role"] != "user"]
        if users:
            plt.scatter([p["x"] for p in users], [p["y"] for p in users], marker="dot", color="cyan", label="user")
        if asst:
            plt.scatter([p["x"] for p in asst], [p["y"] for p in asst], marker="dot", color="magenta", label="asst")
        tail = pts[-6:]
        if len(tail) > 1:
            plt.plot([p["x"] for p in tail], [p["y"] for p in tail], color="white", marker="braille")
        pred = m.get("predicted")
        if pred:
            plt.scatter([pred["x"]], [pred["y"]], marker="heart", color="yellow", label="predicted next")
            last = pts[-1]
            plt.plot([last["x"], pred["x"]], [last["y"], pred["y"]], color="yellow", marker="braille")
        plt.title(f"conversation in latent space (PCA of {len(pts)} turns)")
        plt.xlabel("pc1")
        plt.ylabel("pc2")
        self.refresh()


class JepaDashboard(App):
    CSS_PATH = "jepa.tcss"
    TITLE = "ChatJEPA core"
    BINDINGS = [
        Binding("n", "new_session", "new session"),
        Binding("s", "cycle_session", "next session"),
        Binding("p", "toggle_training", "pause/resume"),
        Binding("c", "checkpoint", "checkpoint"),
        Binding("r", "reembed", "re-embed"),
        Binding("q", "quit", "quit"),
    ]

    def __init__(self, base_url: str = DEFAULT_BASE):
        super().__init__()
        self.client = CoreClient(base_url)
        self.base_url = base_url
        self.connected = False
        self.session_ids: list[str] = []
        self.active_session: str | None = None
        self.surprises: list[float] = []
        self.loss2: list[float] = []
        self.lstd: list[float] = []

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        yield LatentMap(id="map")
        with Vertical(id="surprise"):
            yield Static("surprise", classes="title")
            yield Static("waiting for a turn", id="surprise_big", classes="big")
            yield Sparkline([], id="surprise_spark", min_color="green", max_color="red")
            yield Static("", id="surprise_note")
        with Vertical(id="sessions"):
            yield Static("sessions (mouths)", classes="title")
            yield DataTable(id="sessions_table", cursor_type="row", zebra_stripes=True)
        with Vertical(id="memory"):
            yield Static("memory hits for last turn", classes="title")
            yield DataTable(id="memory_table", zebra_stripes=True)
        with Vertical(id="training"):
            yield Static("online learning", classes="title")
            yield Static("", id="training_text")
            yield Sparkline([], id="loss_spark", min_color="blue", max_color="cyan")
            yield Sparkline([], id="std_spark", min_color="red", max_color="green")
        yield RichLog(id="log", markup=True, max_lines=400)
        yield Footer()

    def on_mount(self) -> None:
        self.query_one("#sessions_table", DataTable).add_columns("session", "turns", "mouth", "last")
        self.query_one("#memory_table", DataTable).add_columns("score", "mouth", "when", "snippet")
        self.log_line(f"connecting to {self.base_url}")
        self.stream_events()
        self.set_interval(3.0, self.refresh_state)

    # ---------------------------------------------------------------- data
    def log_line(self, msg: str) -> None:
        self.query_one("#log", RichLog).write(Text.from_markup(f"[dim]{datetime.now():%H:%M:%S}[/] {msg}"))

    @work(exclusive=True)
    async def refresh_state(self) -> None:
        try:
            st = await self.client.state()
        except Exception as e:  # noqa: BLE001
            if self.connected:
                self.log_line(f"[red]core offline[/] ({e.__class__.__name__})")
            self.connected = False
            self.sub_title = "core: offline"
            return
        if not self.connected:
            self.log_line("[green]connected[/]")
        self.connected = True
        self.apply_state(st)

    def apply_state(self, st: dict) -> None:
        tr = st.get("training") or {}
        self.sub_title = f"core: v{tr.get('enc_version', 0)} step {tr.get('step', 0)} {tr.get('device', '')}"
        self.active_session = st.get("session_id")
        self.surprises = list(st.get("surprise_recent") or [])
        self.update_surprise(self.surprises[-1] if self.surprises else None, None)
        self.query_one("#map", LatentMap).update_map(st.get("map") or {})
        self.update_sessions(st.get("sessions") or [])
        self.update_memory(st.get("top_memory") or [])
        metrics = st.get("metrics") or []
        if metrics:
            self.loss2 = [m["loss2"] for m in metrics if m.get("loss2") is not None][-60:]
            self.lstd = [m["latent_std"] for m in metrics if m.get("latent_std") is not None][-60:]
        self.update_training(tr)

    def update_surprise(self, value: float | None, label: str | None) -> None:
        big = self.query_one("#surprise_big", Static)
        if value is None:
            big.update("first turn")
        else:
            label = label or ("calm" if value < 0.5 else "notable" if value < 0.9 else "surprising")
            color = {"calm": "green", "notable": "yellow", "surprising": "red"}[label]
            big.update(Text.assemble((f"{value:.2f}", f"bold {color}"), f"  {label}"))
        spark = self.query_one("#surprise_spark", Sparkline)
        spark.display = bool(self.surprises)
        spark.data = self.surprises[-40:] or [0.0]
        mean = sum(self.surprises) / len(self.surprises) if self.surprises else None
        self.query_one("#surprise_note", Static).update(
            f"0 = predicted, 1 = chance | mean {mean:.2f} over {len(self.surprises)}" if mean is not None else ""
        )

    def update_sessions(self, sessions: list[dict]) -> None:
        t = self.query_one("#sessions_table", DataTable)
        t.clear()
        self.session_ids = []
        for s in sessions[:12]:
            mark = "▶ " if s.get("active") else "  "
            title = (s.get("title") or s["id"])[:18]
            t.add_row(f"{mark}{title}", str(s.get("n_turns", 0)), s.get("mouth") or "-", _ago(s.get("last_ts")))
            self.session_ids.append(s["id"])

    def update_memory(self, hits: list[dict]) -> None:
        t = self.query_one("#memory_table", DataTable)
        t.clear()
        for h in hits[:5]:
            t.add_row(
                f"{h.get('score', 0):.2f}",
                h.get("mouth") or h.get("source") or "-",
                _ago(h.get("ts")),
                (h.get("snippet") or "")[:60],
            )

    def update_training(self, tr: dict) -> None:
        def f(x, n=4):
            return "-" if x is None else f"{x:.{n}f}"

        paused = "PAUSED" if tr.get("paused") else "training"
        warn = f"\n[red]{tr['warning']}[/]" if tr.get("warning") else ""
        self.query_one("#training_text", Static).update(
            Text.from_markup(
                f"{paused}  step {tr.get('step', 0)}  {tr.get('steps_per_s', 0):.1f} step/s\n"
                f"loss1 {f(tr.get('loss1'))}  loss2 {f(tr.get('loss2'))}  latent_std {f(tr.get('latent_std'), 3)}"
                f"  (pretrain {f(tr.get('pretrain_latent_std'), 3)}){warn}"
            )
        )
        for wid, data in (("#loss_spark", self.loss2), ("#std_spark", self.lstd)):
            w = self.query_one(wid, Sparkline)
            w.display = len(data) > 1
            w.data = data or [0.0]

    # ---------------------------------------------------------------- events
    @work(exclusive=False)
    async def stream_events(self) -> None:
        backoff = 1.0
        while True:
            try:
                async for ev in self.client.events():
                    backoff = 1.0
                    self.handle_event(ev)
            except Exception:  # noqa: BLE001
                pass
            await asyncio.sleep(backoff)
            backoff = min(backoff * 2, 15.0)

    def handle_event(self, ev: dict) -> None:
        kind = ev.get("type")
        if kind == "turn":
            s = ev.get("surprise")
            if s is not None:
                self.surprises = (self.surprises + [s])[-50:]
            self.update_surprise(s, ev.get("label"))
            self.update_memory(ev.get("related") or [])
            if ev.get("map"):
                self.query_one("#map", LatentMap).update_map(ev["map"])
            tag = {"calm": "green", "notable": "yellow", "surprising": "red"}.get(ev.get("label"), "white")
            sv = f"[{tag}]{s:.2f}[/]" if s is not None else "[dim]first[/]"
            self.log_line(f"[bold]{ev.get('mouth')}[/] {ev.get('role')}: {ev.get('snippet')}  surprise {sv}")
        elif kind == "train_step":
            self.loss2 = (self.loss2 + [ev["loss2"]])[-60:]
            self.lstd = (self.lstd + [ev["latent_std"]])[-60:]
            self.update_training(
                {
                    "step": ev.get("step"),
                    "steps_per_s": ev.get("steps_per_s", 0),
                    "loss1": ev.get("loss1"),
                    "loss2": ev.get("loss2"),
                    "latent_std": ev.get("latent_std"),
                    "paused": False,
                }
            )
        elif kind == "recall":
            self.log_line(f"[cyan]recall[/] {ev.get('query')} -> {len(ev.get('hits') or [])} hits")
        elif kind in ("session", "checkpoint", "warning", "reembed", "import", "snapshot", "training"):
            detail = {k: v for k, v in ev.items() if k not in ("type", "ts")}
            color = "red" if kind == "warning" else "blue"
            self.log_line(f"[{color}]{kind}[/] {detail}")
            if kind in ("session", "import"):
                self.refresh_state()

    # ---------------------------------------------------------------- actions
    async def action_new_session(self) -> None:
        await self.client.session("new", title=f"tui {datetime.now():%H:%M}")
        self.refresh_state()

    async def action_cycle_session(self) -> None:
        if not self.session_ids:
            return
        cur = self.session_ids.index(self.active_session) if self.active_session in self.session_ids else -1
        nxt = self.session_ids[(cur + 1) % len(self.session_ids)]
        await self.client.session("resume", session_id=nxt)
        self.refresh_state()

    async def action_toggle_training(self) -> None:
        st = await self.client.stats()
        res = await self.client.control("resume" if st.get("paused") else "pause")
        self.log_line(f"training {'paused' if res.get('paused') else 'resumed'}")

    async def action_checkpoint(self) -> None:
        await self.client.control("checkpoint")

    async def action_reembed(self) -> None:
        await self.client.control("reembed")


def main() -> None:
    ap = argparse.ArgumentParser(prog="jepa-tui")
    ap.add_argument("--url", default=DEFAULT_BASE)
    args = ap.parse_args()
    JepaDashboard(args.url).run()


if __name__ == "__main__":
    main()
