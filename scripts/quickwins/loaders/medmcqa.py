"""MedMCQA dental subset (subject_name == 'Dental') -> SFT rows.

Each MCQ becomes one SFT example with the question + lettered options as the prompt
and the indexed correct option (plus explanation if present) as the answer.
"""
from __future__ import annotations

from typing import Iterator

from datasets import load_dataset

from ..common.schema import make_sft

LETTERS = ["A", "B", "C", "D"]
SOURCE = "medmcqa_dental"


def _format(row: dict) -> tuple[str, str] | None:
    q = (row.get("question") or "").strip()
    opts = [row.get("opa"), row.get("opb"), row.get("opc"), row.get("opd")]
    if not q or any(not o for o in opts):
        return None
    cop = row.get("cop")
    try:
        cop = int(cop)
    except (TypeError, ValueError):
        return None
    if cop < 0 or cop > 3:
        return None
    prompt = q + "\n\n" + "\n".join(f"{LETTERS[i]}. {opts[i]}" for i in range(4))
    correct_letter = LETTERS[cop]
    correct_text = (opts[cop] or "").strip()
    exp = (row.get("exp") or "").strip()
    if exp:
        answer = f"{correct_letter}. {correct_text}\n\nExplanation: {exp}"
    else:
        answer = f"{correct_letter}. {correct_text}"
    return prompt, answer


def load() -> Iterator[dict]:
    ds = load_dataset("openlifescienceai/medmcqa", split="train")
    ds = ds.filter(lambda r: (r.get("subject_name") or "").strip().lower() == "dental")
    for row in ds:
        fmt = _format(row)
        if not fmt:
            continue
        prompt, answer = fmt
        topic = (row.get("topic_name") or "Dentistry").strip() or "Dentistry"
        rec = make_sft(prompt, answer, SOURCE, topic)
        if rec:
            yield rec
