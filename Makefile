.PHONY: setup test lint train chat ingest pretrain eval core tui install-clients

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

ingest:
	uv run jepa-ingest --all-local

pretrain:
	uv run jepa-pretrain --config experiments/002_chat_level1/config.yaml --stage all

eval:
	uv run jepa-eval

core:
	uv run jepa-core serve

tui:
	uv run jepa-tui

install-clients:
	uv run jepa-core install-client all --write
