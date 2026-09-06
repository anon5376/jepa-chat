"""Thin wrapper so the experiment is runnable as a script."""

from pathlib import Path

from jepa_chat.config import Config
from jepa_chat.training.train import train

if __name__ == "__main__":
    train(Config.from_yaml(Path(__file__).parent / "config.yaml"))
