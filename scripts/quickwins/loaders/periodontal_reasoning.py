"""Wildstash/periodontal-reasoning-40k — large dental reasoning QA set."""
from __future__ import annotations

from typing import Iterator

from datasets import load_dataset

from ..common.schema import make_sft

SOURCE = "wildstash_periodontal_reasoning"
CANDIDATES = [
    "WildStash/periodontal-reasoning-40k",
    "Wildstash/periodontal-reasoning-40k",
]


def _q_a(row: dict) -> tuple[str, str] | None:
    # Try a bunch of common keys to be schema-agnostic.
    q = (row.get("question") or row.get("prompt") or row.get("instruction")
         or row.get("query") or row.get("input") or "").strip()
    a = (row.get("answer") or row.get("response") or row.get("output")
         or row.get("completion") or row.get("reasoning") or "").strip()
    # If it's a {messages: [...]} chat schema:
    msgs = row.get("messages") or row.get("conversations")
    if (not q or not a) and isinstance(msgs, list) and msgs:
        u = next((m for m in msgs if (m.get("role") or m.get("from")) in ("user", "human")), None)
        s = next((m for m in msgs if (m.get("role") or m.get("from")) in ("assistant", "gpt", "model")), None)
        if u and not q:
            q = (u.get("content") or u.get("value") or "").strip()
        if s and not a:
            a = (s.get("content") or s.get("value") or "").strip()
    if q and a:
        return q, a
    return None


def load() -> Iterator[dict]:
    ds = None
    last_err = None
    for repo in CANDIDATES:
        try:
            ds = load_dataset(repo, split="train")
            break
        except Exception as e:
            last_err = e
    if ds is None:
        # try without split
        for repo in CANDIDATES:
            try:
                dsd = load_dataset(repo)
                # take first available split
                split = next(iter(dsd.keys()))
                ds = dsd[split]
                break
            except Exception as e:
                last_err = e
    if ds is None:
        raise RuntimeError(f"Could not load periodontal-reasoning: {last_err}")
    for row in ds:
        qa = _q_a(row)
        if not qa:
            continue
        q, a = qa
        topic = (row.get("topic") or row.get("category") or "Periodontics").strip() or "Periodontics"
        rec = make_sft(q, a, SOURCE, topic)
        if rec:
            yield rec
