"""PubMedQA filtered to dental-MeSH abstracts -> SFT.

We use the labelled (PQA-L, 1k) + artificial (PQA-A, 211k) configs. For each kept
row we emit one SFT example with the question, the context (abstract), and a
labelled yes/no/maybe answer plus the long answer if present.
"""
from __future__ import annotations

from typing import Iterator

from datasets import load_dataset

from ..common.dental_filter import is_dental, mesh_is_dental
from ..common.schema import make_sft

CONFIGS = [
    ("pqa_labeled", "pubmedqa_dental_labeled"),
    ("pqa_artificial", "pubmedqa_dental_artificial"),
]


def _context_text(ctx) -> tuple[str, list[str]]:
    """Return (concatenated context, mesh_list)."""
    if isinstance(ctx, dict):
        contexts = ctx.get("contexts") or []
        labels = ctx.get("labels") or []
        mesh = ctx.get("meshes") or ctx.get("mesh") or []
        if isinstance(contexts, list):
            chunks = []
            for lab, txt in zip(labels, contexts):
                if txt:
                    chunks.append(f"{lab}: {txt}" if lab else str(txt))
            return "\n".join(chunks), list(mesh) if isinstance(mesh, list) else [str(mesh)]
        return str(contexts), list(mesh) if isinstance(mesh, list) else [str(mesh)]
    return str(ctx or ""), []


def _row_to_sft(row: dict, source: str) -> dict | None:
    q = (row.get("question") or row.get("QUESTION") or "").strip()
    ctx, mesh = _context_text(row.get("context") or row.get("CONTEXTS") or row.get("contexts"))
    final = (row.get("final_decision") or row.get("FINAL_DECISION") or "").strip().lower()
    long_a = (row.get("long_answer") or row.get("LONG_ANSWER") or "").strip()

    if not q:
        return None
    if not (mesh_is_dental(mesh) or is_dental(q, ctx, long_a)):
        return None

    prompt = q
    if ctx:
        prompt += "\n\nContext:\n" + ctx[:6000]

    if long_a and final:
        answer = f"{final.capitalize()}. {long_a}"
    elif long_a:
        answer = long_a
    elif final:
        answer = final.capitalize()
    else:
        return None
    return make_sft(prompt, answer, source, "Dental Research QA")


def load() -> Iterator[dict]:
    for cfg, src in CONFIGS:
        try:
            ds = load_dataset("qiaojin/PubMedQA", cfg, split="train")
        except Exception:
            try:
                ds = load_dataset("bigbio/pubmed_qa", cfg, split="train")
            except Exception as e:
                print(f"[pubmedqa] {cfg} load failed: {e}")
                continue
        for row in ds:
            rec = _row_to_sft(row, src)
            if rec:
                yield rec
