"""Run-folder bookkeeping shared by train.py and train_clip.py.

Layout per run:
    <base_dir>/<timestamp>/
        config.json          # frozen Config at start of run
        metrics.csv          # appended every log/val event
        step_<N>.pt          # checkpoints (optimizer, scheduler, RNG inside)
        loss.png, acc.png    # written at end of run
"""
import csv
import json
import os
import random
import re
from dataclasses import asdict
from datetime import datetime

import torch

from config import Config


def make_run_dir(base_dir: str) -> str:
    name = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    path = os.path.join(base_dir, name)
    os.makedirs(path, exist_ok=True)
    return path


def save_config(cfg: Config, run_dir: str) -> None:
    with open(os.path.join(run_dir, "config.json"), "w") as f:
        json.dump(asdict(cfg), f, indent=2)


def load_config(run_dir: str) -> Config:
    with open(os.path.join(run_dir, "config.json")) as f:
        return Config(**json.load(f))


def find_latest_ckpt(run_dir: str):
    pat = re.compile(r"^step_(\d+)\.pt$")
    best, best_step = None, -1
    if not os.path.isdir(run_dir):
        return None
    for name in os.listdir(run_dir):
        m = pat.match(name)
        if m and int(m.group(1)) > best_step:
            best_step = int(m.group(1))
            best = os.path.join(run_dir, name)
    return best


def rng_snapshot():
    return {
        "torch": torch.get_rng_state(),
        "python": random.getstate(),
    }


def rng_restore(state):
    if not state:
        return
    # torch.load(..., map_location=device) moves all tensors in the blob to
    # `device`; the RNG state must be a CPU uint8 tensor regardless.
    t = state["torch"]
    if isinstance(t, torch.Tensor):
        t = t.detach().to(device="cpu", dtype=torch.uint8)
    torch.set_rng_state(t)
    random.setstate(state["python"])


class MetricsLogger:
    """Append-only CSV. One row per train log interval and per val event."""

    FIELDS = ["step", "event", "loss", "acc", "lr", "gnorm", "scale", "aux", "ce",
              "pc", "recon", "sat", "mid", "ms_per_step"]

    def __init__(self, run_dir: str):
        self.path = os.path.join(run_dir, "metrics.csv")
        is_new = not os.path.exists(self.path)
        self.f = open(self.path, "a", newline="")
        self.w = csv.DictWriter(self.f, fieldnames=self.FIELDS)
        if is_new:
            self.w.writeheader()
            self.f.flush()

    def log(self, **row):
        self.w.writerow({k: row.get(k, "") for k in self.FIELDS})
        self.f.flush()

    def close(self):
        try:
            self.f.close()
        except Exception:
            pass


def plot_metrics(run_dir: str) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:
        print(f"[plot] matplotlib unavailable, skipping: {e}")
        return

    path = os.path.join(run_dir, "metrics.csv")
    if not os.path.exists(path):
        return

    series = {"train": {"step": [], "loss": [], "acc": []},
              "val":   {"step": [], "loss": [], "acc": []}}
    with open(path) as f:
        for r in csv.DictReader(f):
            ev = r.get("event")
            if ev not in series:
                continue
            try:
                step = int(r["step"])
            except (ValueError, KeyError):
                continue
            series[ev]["step"].append(step)
            series[ev]["loss"].append(float(r["loss"]) if r.get("loss") else float("nan"))
            series[ev]["acc"].append(float(r["acc"]) if r.get("acc") else float("nan"))

    for metric, fname, title in [("loss", "loss.png", "loss"), ("acc", "acc.png", "accuracy")]:
        fig, ax = plt.subplots()
        plotted = False
        for ev, style in [("train", dict(alpha=0.7, linewidth=1)),
                          ("val", dict(marker="o", linewidth=1.5))]:
            xs = series[ev]["step"]
            ys = series[ev][metric]
            if xs:
                ax.plot(xs, ys, label=ev, **style)
                plotted = True
        ax.set_xlabel("step")
        ax.set_ylabel(metric)
        ax.set_title(title)
        if plotted:
            ax.legend()
        ax.grid(alpha=0.3)
        fig.savefig(os.path.join(run_dir, fname), dpi=120, bbox_inches="tight")
        plt.close(fig)
