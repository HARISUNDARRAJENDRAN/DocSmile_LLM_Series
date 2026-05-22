"""Optional Gemini-backed topic inference, batched 20 per call.

Used only when a record lacks a topic. Cached by content hash; no repeat calls.
"""
from __future__ import annotations

import json
import re
from typing import Iterable

from .gemini import GeminiPool

TOPIC_TAXONOMY = [
    "Operative Dentistry", "Endodontics", "Periodontics", "Prosthodontics",
    "Orthodontics", "Oral Surgery", "Oral Medicine", "Oral Pathology",
    "Oral Radiology", "Pediatric Dentistry", "Preventive Dentistry",
    "Dental Public Health", "Dental Anatomy", "Dental Materials",
    "Dental Pharmacology", "Implantology", "Dental Hygiene", "Oral Cancer",
    "TMJ Disorders", "Dental Caries", "Dental Emergencies",
    "Oral Microbiology", "Dental Anesthesia", "Forensic Dentistry",
    "Geriatric Dentistry", "Dental Ethics", "Dentistry",
]

SYSTEM = (
    "You are a strict classifier for dental Q&A. For each item, return ONE topic "
    "label from this fixed list: " + ", ".join(TOPIC_TAXONOMY) + ". "
    "If none clearly applies, use 'Dentistry'. Respond with a JSON array of strings "
    "ONLY, one per item, same order."
)


def _build_prompt(items: list[dict]) -> str:
    payload = []
    for i, it in enumerate(items):
        q = (it.get("question") or "")[:600]
        a = (it.get("answer") or "")[:600]
        payload.append({"i": i, "q": q, "a": a})
    return (
        "Classify each of the following items. Return JSON array of length "
        f"{len(items)} of topic strings only.\n\n" + json.dumps(payload, ensure_ascii=False)
    )


_LIST_RE = re.compile(r"\[.*\]", re.DOTALL)


def _parse(out: str, n: int) -> list[str]:
    m = _LIST_RE.search(out or "")
    if not m:
        return ["Dentistry"] * n
    try:
        arr = json.loads(m.group(0))
    except Exception:
        return ["Dentistry"] * n
    out_list: list[str] = []
    for v in arr[:n]:
        v = str(v).strip()
        if v not in TOPIC_TAXONOMY:
            v = "Dentistry"
        out_list.append(v)
    while len(out_list) < n:
        out_list.append("Dentistry")
    return out_list


def tag_topics(pool: GeminiPool, items: list[dict], batch: int = 20) -> list[str]:
    out: list[str] = []
    for i in range(0, len(items), batch):
        chunk = items[i:i + batch]
        prompt = _build_prompt(chunk)
        est = 800 + sum(len((it.get('question') or '')) + len((it.get('answer') or '')) for it in chunk) // 3
        text = pool.generate(prompt, system=SYSTEM, est_tokens=min(est, 8000))
        out.extend(_parse(text, len(chunk)))
    return out


CLEAN_SYSTEM = (
    "You clean noisy doctor-patient dental dialogues into structured Q&A. "
    "For each input, extract a focused patient question (rephrased clinically if needed) "
    "and a clear, helpful clinical answer (preserve key clinical facts; remove greetings, "
    "salutations, signatures, names, and platform boilerplate). Return JSON array of "
    "objects {i, question, answer, topic} where topic is one of: "
    + ", ".join(TOPIC_TAXONOMY)
    + ". If a sample is not dental or not salvageable, return {i, skip: true}."
)


def clean_dialogues(pool: GeminiPool, items: list[dict], batch: int = 8) -> list[dict | None]:
    """items: [{q, a}]. Returns list aligned with items; None means skip."""
    results: list[dict | None] = []
    for i in range(0, len(items), batch):
        chunk = items[i:i + batch]
        payload = [{"i": j, "q": (it.get("q") or "")[:1500], "a": (it.get("a") or "")[:3000]}
                   for j, it in enumerate(chunk)]
        prompt = (
            "Clean the following dental doctor-patient dialogues into structured Q&A. "
            f"Return JSON array of length {len(chunk)} of objects keyed by 'i'.\n\n"
            + json.dumps(payload, ensure_ascii=False)
        )
        est = 1200 + sum(len(it["q"]) + len(it["a"]) for it in payload) // 3
        text = pool.generate(prompt, system=CLEAN_SYSTEM, est_tokens=min(est, 12000))
        m = _LIST_RE.search(text or "")
        parsed: list[dict | None] = [None] * len(chunk)
        if m:
            try:
                arr = json.loads(m.group(0))
                for obj in arr:
                    j = int(obj.get("i", -1))
                    if 0 <= j < len(chunk):
                        if obj.get("skip"):
                            parsed[j] = None
                        else:
                            q = (obj.get("question") or "").strip()
                            a = (obj.get("answer") or "").strip()
                            t = (obj.get("topic") or "Dentistry").strip()
                            if t not in TOPIC_TAXONOMY:
                                t = "Dentistry"
                            if q and a:
                                parsed[j] = {"question": q, "answer": a, "topic": t}
            except Exception:
                pass
        results.extend(parsed)
    return results
