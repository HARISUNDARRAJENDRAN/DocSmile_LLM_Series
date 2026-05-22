"""Question-hash deduper. Also dedups against existing SFT corpus."""
from __future__ import annotations

import hashlib
import re
from pathlib import Path

from .io import EXISTING_SFT, read_jsonl


def _norm(s: str) -> str:
    s = s.lower()
    s = re.sub(r"[^a-z0-9À-ɏ一-鿿\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def qhash(question: str) -> str:
    return hashlib.sha1(_norm(question).encode("utf-8")).hexdigest()


class Deduper:
    def __init__(self, seed_paths: list[Path] | None = None):
        self.seen: set[str] = set()
        if seed_paths:
            for p in seed_paths:
                if p.exists():
                    self.seed_from_jsonl(p)

    def seed_from_jsonl(self, path: Path, key: str = "question") -> int:
        n = 0
        for row in read_jsonl(path):
            q = row.get(key) or row.get("prompt") or ""
            if q:
                self.seen.add(qhash(q))
                n += 1
        return n

    def add(self, question: str) -> bool:
        """Return True if newly added, False if duplicate."""
        h = qhash(question)
        if h in self.seen:
            return False
        self.seen.add(h)
        return True


def existing_sft_deduper() -> Deduper:
    return Deduper(seed_paths=[EXISTING_SFT])
