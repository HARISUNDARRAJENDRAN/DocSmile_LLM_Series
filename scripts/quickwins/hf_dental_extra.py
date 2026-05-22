"""Loader for additional HF dental/medical datasets not in the first batch.

Datasets:
  1. Lines/Open-Domain-Oral-Disease-QA-Dataset  (672 rows, oral disease Q&A)
  2. TachyHealth/ADA_Dental_Code_to_SBS_V2      (ADA dental code descriptions -> CPT)
  3. bigbio/med_qa dental subset                 (USMLE-style MCQ, dental filter)
  4. medalpaca/medical_meadow_medical_flashcards  (medical flashcards, dental filter)

Output:
  cpt_prepared/quick_wins/hf_dental_extra_sft.jsonl
  cpt_prepared/quick_wins/hf_dental_extra_cpt.jsonl
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "cpt_prepared" / "quick_wins"
OUT_DIR.mkdir(parents=True, exist_ok=True)

SFT_OUT = OUT_DIR / "hf_dental_extra_sft.jsonl"
CPT_OUT = OUT_DIR / "hf_dental_extra_cpt.jsonl"

DENTAL_RE = re.compile(
    r"\b(dent(?:al|ist|in|ition|ure)|tooth|teeth|molar|premolar|incisor|canine|"
    r"oral|gingiv|periodon|endodon|orthodon|prosthodon|pulp(?:itis|ectomy|otomy)|"
    r"caries|cavity|crown|bridge|implant|extraction|root canal|"
    r"mandib|maxill|alveol|occlus|malocclu|bruxism|"
    r"fluoride|plaque|tartar|calculus|scaling|"
    r"apical|periapical|abscess|fistula|granuloma|"
    r"amalgam|composite|ceramic|zirconia|"
    r"tmj|temporomandibular|stomatitis|glossitis|"
    r"leukoplakia|erythroplakia|lichen planus|"
    r"salivary|parotid|sublingual|submandibular|"
    r"cleft lip|cleft palate|oral surgery|oral pathol|"
    r"dental material|dental cement|dental anest|"
    r"trigeminal|inferior alveolar nerve|lingual nerve)\b",
    re.IGNORECASE,
)


def is_dental(text: str) -> bool:
    return len(DENTAL_RE.findall(text)) >= 2


def _make_sft(question: str, answer: str, source: str, topic: str = "Dentistry") -> dict | None:
    q = question.strip()
    a = answer.strip()
    if len(q) < 15 or len(a) < 30:
        return None
    return {"question": q, "answer": a, "source": source, "topic": topic}


def _make_cpt(text: str, source: str) -> dict | None:
    t = text.strip()
    if len(t) < 100:
        return None
    return {"text": t, "source": source}


def load_oral_disease_qa() -> tuple[list[dict], list[dict]]:
    """Lines/Open-Domain-Oral-Disease-QA-Dataset."""
    from datasets import load_dataset
    sft, cpt = [], []
    try:
        ds = load_dataset("Lines/Open-Domain-Oral-Disease-QA-Dataset", split="train")
    except Exception as e:
        print(f"[hf_extra] oral disease QA failed: {e}")
        return sft, cpt

    for row in ds:
        question = (row.get("question") or "").strip()
        answer = (row.get("Answer") or "").strip()
        disease = (row.get("disease") or "oral disease").strip()
        validity = (row.get("validity") or "").strip().lower()
        # Only keep correct/partially correct answers
        if validity in ("incorrect",):
            continue
        if not question or not answer:
            continue
        r = _make_sft(question, answer, "hf:Lines/Open-Domain-Oral-Disease-QA", disease)
        if r:
            sft.append(r)
        c = _make_cpt(f"Q: {question}\nA: {answer}", "hf:Lines/Open-Domain-Oral-Disease-QA")
        if c:
            cpt.append(c)
    print(f"[hf_extra] oral_disease_qa: sft={len(sft)} cpt={len(cpt)}")
    return sft, cpt


def load_ada_dental_codes() -> tuple[list[dict], list[dict]]:
    """TachyHealth/ADA_Dental_Code_to_SBS_V2 -- dental procedure codes -> CPT text."""
    from datasets import load_dataset
    sft, cpt = [], []
    try:
        ds = load_dataset("TachyHealth/ADA_Dental_Code_to_SBS_V2", split="train")
    except Exception as e:
        print(f"[hf_extra] ADA dental codes failed: {e}")
        return sft, cpt

    cols = ds.column_names
    print(f"[hf_extra] ADA dental codes columns: {cols}")

    for row in ds:
        # Try to extract text from available columns
        parts = []
        for col in cols:
            val = row.get(col)
            if val and isinstance(val, str) and len(val.strip()) > 5:
                parts.append(f"{col}: {val.strip()}")
        text = "\n".join(parts)
        if len(text) > 100:
            c = _make_cpt(text, "hf:TachyHealth/ADA_Dental_Code_to_SBS_V2")
            if c:
                cpt.append(c)
        # Also create SFT: "What dental procedure is described by code X?"
        code = row.get("CDT_Code") or row.get("code") or row.get("Code") or ""
        desc = row.get("Description") or row.get("description") or row.get("SBS") or ""
        if code and desc and len(desc) > 30:
            q = f"What dental procedure is described by the CDT code {code}?"
            r = _make_sft(q, desc, "hf:TachyHealth/ADA_Dental_Code_to_SBS_V2", "Dentistry")
            if r:
                sft.append(r)
    print(f"[hf_extra] ada_dental_codes: sft={len(sft)} cpt={len(cpt)}")
    return sft, cpt


def load_medqa_dental() -> tuple[list[dict], list[dict]]:
    """bigbio/med_qa -- USMLE-style MCQ, filter for dental content."""
    from datasets import load_dataset
    sft, cpt = [], []
    try:
        ds = load_dataset("bigbio/med_qa", "med_qa_en_source", split="train",
                          trust_remote_code=True)
    except Exception as e:
        print(f"[hf_extra] med_qa load failed: {e}")
        try:
            ds = load_dataset("bigbio/med_qa", split="train", trust_remote_code=True)
        except Exception as e2:
            print(f"[hf_extra] med_qa retry failed: {e2}")
            return sft, cpt

    cols = ds.column_names
    print(f"[hf_extra] med_qa columns: {cols}, rows: {len(ds)}")

    for row in ds:
        question = (row.get("question") or "").strip()
        answer = (row.get("answer") or "").strip()
        options = row.get("options") or row.get("choices") or {}
        explanation = (row.get("explanation") or "").strip()

        combined = f"{question} {answer} {explanation}"
        if not is_dental(combined):
            continue

        if isinstance(options, dict):
            correct_key = row.get("answer_idx") or row.get("answer")
            answer_text = options.get(correct_key, answer)
        elif isinstance(options, list):
            answer_text = answer
        else:
            answer_text = answer

        full_answer = answer_text
        if explanation:
            full_answer = f"{answer_text}\n\n{explanation}"

        r = _make_sft(question, full_answer, "hf:bigbio/med_qa", "Dentistry")
        if r:
            sft.append(r)
    print(f"[hf_extra] medqa_dental: sft={len(sft)}")
    return sft, cpt


def load_medical_flashcards() -> tuple[list[dict], list[dict]]:
    """medalpaca/medical_meadow_medical_flashcards -- dental subset."""
    from datasets import load_dataset
    sft, cpt = [], []
    try:
        ds = load_dataset("medalpaca/medical_meadow_medical_flashcards", split="train")
    except Exception as e:
        print(f"[hf_extra] medical flashcards failed: {e}")
        return sft, cpt

    cols = ds.column_names
    print(f"[hf_extra] medical flashcards columns: {cols}, rows: {len(ds)}")

    for row in ds:
        question = ""
        answer = ""
        # Try different column name patterns
        for qk in ("input", "question", "instruction", "prompt"):
            if qk in cols and row.get(qk):
                question = row[qk].strip()
                break
        for ak in ("output", "answer", "response"):
            if ak in cols and row.get(ak):
                answer = row[ak].strip()
                break

        combined = f"{question} {answer}"
        if not is_dental(combined):
            continue

        r = _make_sft(question, answer, "hf:medalpaca/medical_flashcards", "Dentistry")
        if r:
            sft.append(r)
    print(f"[hf_extra] medical_flashcards_dental: sft={len(sft)}")
    return sft, cpt


def main(argv=None) -> int:
    all_sft: list[dict] = []
    all_cpt: list[dict] = []

    loaders = [
        ("oral_disease_qa", load_oral_disease_qa),
        ("ada_dental_codes", load_ada_dental_codes),
        ("medqa_dental", load_medqa_dental),
        ("medical_flashcards", load_medical_flashcards),
    ]

    for name, loader in loaders:
        print(f"\n[hf_extra] --- {name} ---")
        try:
            sft, cpt = loader()
            all_sft.extend(sft)
            all_cpt.extend(cpt)
        except Exception as e:
            print(f"[hf_extra] {name} FAILED: {e}")

    # Write outputs
    n_sft = 0
    with SFT_OUT.open("w", encoding="utf-8") as f:
        for row in all_sft:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            n_sft += 1

    n_cpt = 0
    with CPT_OUT.open("w", encoding="utf-8") as f:
        for row in all_cpt:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            n_cpt += 1

    print(f"\n[hf_extra] DONE: sft={n_sft} cpt={n_cpt}")
    print(f"  -> {SFT_OUT}")
    print(f"  -> {CPT_OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
