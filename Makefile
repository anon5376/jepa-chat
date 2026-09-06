.PHONY: setup test lint train chat

setup:
	uv sync --extra dev

test:
	uv run pytest -q

lint:
	uv run ruff check src tests experiments

train:
	uv run jepa-train --config experiments/001_text_jepa_toy/config.yaml

chat:
	uv run jepa-chat
