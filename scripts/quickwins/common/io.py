"""I/O helpers for the quick-wins pipeline."""
from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Iterable, Iterator

ROOT = Path(__file__).resolve().parents[3]
RAW_DIR = ROOT / "cpt_prepared" / "quick_wins" / "_raw"
CACHE_DIR = ROOT / "cpt_prepared" / "quick_wins" / "_cache"
OUT_DIR = ROOT / "cpt_prepared" / "quick_wins"
CPT_OUT_DIR = ROOT / "cpt_prepared" / "quick_wins_cpt"
LOG_DIR = ROOT / "logs" / "quickwins"
EXISTING_SFT = ROOT / "rl_prepared" / "rl_sft.jsonl"

for d in (RAW_DIR, CACHE_DIR, OUT_DIR, CPT_OUT_DIR, LOG_DIR):
    d.mkdir(parents=True, exist_ok=True)


def write_jsonl(path: Path, rows: Iterable[dict]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            n += 1
    return n


def read_jsonl(path: Path) -> Iterator[dict]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


def load_env_keys(files: list[str] | None = None) -> list[str]:
    """Load GEMINI_API_KEY* from env files.

    `files` is a list of filenames (relative to ROOT) to read in order. Default
    reads `1.env` first (preferred) then `.env`. Pass `["1.env"]` to use only
    the working 6-key set when most other keys have been suspended.
    """
    keys: list[str] = []
    seen: set[str] = set()
    if files is None:
        files = ["1.env", ".env"]
    for env_name in files:
        env_path = ROOT / env_name
        if not env_path.exists():
            continue
        for line in env_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            m = re.match(r"^(?:export\s+)?(GEMINI_API_KEY\w*)\s*=\s*(.+?)\s*$", line)
            if not m:
                continue
            val = m.group(2).strip().strip('"').strip("'")
            if val and val not in seen:
                seen.add(val)
                keys.append(val)
    return keys
