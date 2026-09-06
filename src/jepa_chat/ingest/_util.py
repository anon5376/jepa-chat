"""Shared helpers for export parsers."""

from __future__ import annotations

import json
import zipfile
from collections.abc import Iterator
from datetime import datetime
from pathlib import Path


def iter_json_files(path: str | Path, name_suffix: str) -> Iterator[tuple[str, object]]:
    """Yield (name, parsed_json) for every file ending in `name_suffix`.

    `path` may be a .json file, a directory, or a .zip archive. Large ChatGPT exports
    ship numbered files (conversations-1.json, ...), so match on suffix.
    """
    path = Path(path)
    if path.is_dir():
        for f in sorted(path.rglob(f"*{name_suffix}")):
            yield str(f), json.loads(f.read_text(encoding="utf-8"))
    elif path.suffix == ".zip":
        with zipfile.ZipFile(path) as z:
            names = sorted(n for n in z.namelist() if n.endswith(name_suffix))
            for n in names:
                with z.open(n) as fh:
                    yield n, json.load(fh)
    else:
        yield str(path), json.loads(path.read_text(encoding="utf-8"))


def parse_iso_ts(value: object) -> float | None:
    if value is None:
        return None
    if isinstance(value, int | float):
        return float(value)
    try:
        return datetime.fromisoformat(str(value).replace("Z", "+00:00")).timestamp()
    except ValueError:
        return None
