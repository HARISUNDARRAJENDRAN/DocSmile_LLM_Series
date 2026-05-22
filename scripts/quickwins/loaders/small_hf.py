"""Small HuggingFace dental QA sets — same Q/A extraction, different schemas."""
from __future__ import annotations

import re
from typing import Iterator

from datasets import load_dataset

from ..common.schema import make_sft


def _extract_qa(row: dict) -> tuple[str, str] | None:
    q = (row.get("question") or row.get("instruction") or row.get("prompt")
         or row.get("input") or row.get("Question") or "").strip()
    a = (row.get("answer") or row.get("output") or row.get("response")
         or row.get("Answer") or row.get("completion") or "").strip()
    if (not q or not a):
        msgs = row.get("messages") or row.get("conversations")
        if isinstance(msgs, list) and msgs:
            u = next((m for m in msgs if (m.get("role") or m.get("from")) in ("user", "human")), None)
            s = next((m for m in msgs if (m.get("role") or m.get("from")) in ("assistant", "gpt", "model")), None)
            if u and not q:
                q = (u.get("content") or u.get("value") or "").strip()
            if s and not a:
                a = (s.get("content") or s.get("value") or "").strip()
    if q and a:
        return q, a
    return None


def _generic(repo: str, source_tag: str, topic: str, split: str = "train",
             config: str | None = None) -> Iterator[dict]:
    try:
        if config:
            ds = load_dataset(repo, config, split=split)
        else:
            ds = load_dataset(repo, split=split)
    except Exception:
        try:
            dsd = load_dataset(repo, config) if config else load_dataset(repo)
            split = next(iter(dsd.keys()))
            ds = dsd[split]
        except Exception as e:
            print(f"[small] {repo} ({config}/{split}) load failed: {e}")
            return
    for row in ds:
        qa = _extract_qa(row)
        if not qa:
            continue
        q, a = qa
        rec = make_sft(q, a, source_tag, topic)
        if rec:
            yield rec


def load_birdiebyte() -> Iterator[dict]:
    yield from _generic("BirdieByte1024/doctor_chat_dental_qa_alpaca",
                        "birdiebyte_dental_implants", "Implantology")


# ---- jonathankang: forum dialogues. Title is the patient Q; later speakers' utterances are the A.
_SPEAKER_RE = re.compile(r"\b([A-Z][A-Za-z0-9_.' \-]{1,40}):\s")


def _split_dialogue(dialogue: str) -> list[tuple[str, str]]:
    """Return list of (speaker, utterance) by scanning for 'Name: ' markers."""
    if not dialogue:
        return []
    matches = list(_SPEAKER_RE.finditer(dialogue))
    if not matches:
        return [("", dialogue.strip())]
    turns = []
    for i, m in enumerate(matches):
        speaker = m.group(1).strip()
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(dialogue)
        text = dialogue[start:end].strip()
        if text:
            turns.append((speaker, text))
    return turns


def _looks_clinician(name: str) -> bool:
    n = name.lower()
    return n.startswith("dr ") or n.startswith("dr.") or n.startswith("doctor ") or "dds" in n or "dmd" in n


def load_jonathankang() -> Iterator[dict]:
    from ..common.dental_filter import is_dental
    try:
        ds = load_dataset("jonathankang/dental_QA", split="train")
    except Exception as e:
        print(f"[small] jonathankang load failed: {e}")
        return
    for row in ds:
        title = (row.get("title") or "").strip()
        dialogue = (row.get("dialogue") or "").strip()
        if not dialogue:
            continue
        turns = _split_dialogue(dialogue)
        if not turns:
            continue
        first_speaker, first_text = turns[0]
        reply_speakers = [t for t in turns[1:] if t[0] and t[0] != first_speaker]
        clinician_reply = next((t for t in reply_speakers if _looks_clinician(t[0])), None)
        reply = clinician_reply or (reply_speakers[0] if reply_speakers else None)
        if not reply:
            continue
        q_parts = []
        if title:
            q_parts.append(title)
        q_parts.append(first_text)
        q = "\n\n".join(q_parts).strip()
        clinician_replies = [t[1] for t in reply_speakers if _looks_clinician(t[0])]
        if clinician_replies:
            a = "\n\n".join(clinician_replies).strip()
        else:
            a = reply[1].strip()
        if len(a) < 40:
            continue
        for ch_bad, ch_ok in [("�", "'"), ("’", "'"), ("‘", "'")]:
            q = q.replace(ch_bad, ch_ok)
            a = a.replace(ch_bad, ch_ok)
        # Strict clinical filter: title + first_text must hit a dental clinical keyword,
        # AND the answer must also hit one (filters business/career/loan/admin posts).
        if not is_dental(title, first_text):
            continue
        if not is_dental(a):
            continue
        rec = make_sft(q, a, "jonathankang_dental_forum", "Dental Consultation")
        if rec:
            yield rec


def load_emilykang() -> Iterator[dict]:
    """emilykang/dentistry_{train,test} are audio-only datasets — skip with note."""
    print("[small] emilykang/dentistry_* is audio-only; skipping (no text content).")
    return
    yield  # pragma: no cover


def load_vuha2003() -> Iterator[dict]:
    """vuha2003/medmcqa-Dental-responses has only a 'test' split; use sft-1.5b config."""
    yield from _generic("vuha2003/medmcqa-Dental-responses",
                        "vuha2003_medmcqa_dental", "Dentistry",
                        split="test", config="sft-1.5b")

