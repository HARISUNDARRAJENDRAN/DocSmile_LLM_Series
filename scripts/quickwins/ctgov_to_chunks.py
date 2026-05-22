"""Convert CT.gov raw dump -> CPT chunks + optional SFT Q&A pairs."""
from __future__ import annotations

import json
import random
import sys
from pathlib import Path
from typing import Iterator

ROOT = Path(__file__).resolve().parents[2]
CT_DIR = ROOT / "cpt_prepared" / "clinicaltrials_dental"
RAW = CT_DIR / "clinicaltrials_dental.jsonl"

OUT_CPT = CT_DIR / "ctgov_dental_cpt.jsonl"
OUT_SFT = CT_DIR / "ctgov_dental_sft.jsonl"


def _g(d: dict, *path):
    cur = d
    for p in path:
        if not isinstance(cur, dict):
            return None
        cur = cur.get(p)
    return cur


def _flatten_outcomes(outcomes: list) -> str:
    if not outcomes:
        return ""
    out = []
    for o in outcomes:
        if isinstance(o, dict):
            m = o.get("measure") or ""
            d = o.get("description") or ""
            tf = o.get("timeFrame") or ""
            line = f"- {m}".rstrip()
            if d:
                line += f": {d}"
            if tf:
                line += f" (timeframe: {tf})"
            out.append(line)
    return "\n".join(out)


def study_to_cpt(study: dict) -> dict | None:
    ps = study.get("protocolSection") or {}
    ident = ps.get("identificationModule") or {}
    nct = ident.get("nctId") or ""
    title = ident.get("officialTitle") or ident.get("briefTitle") or ""
    desc_mod = ps.get("descriptionModule") or {}
    brief = desc_mod.get("briefSummary") or ""
    detail = desc_mod.get("detailedDescription") or ""
    cond_mod = ps.get("conditionsModule") or {}
    conditions = cond_mod.get("conditions") or []
    keywords = cond_mod.get("keywords") or []
    elig_mod = ps.get("eligibilityModule") or {}
    elig = elig_mod.get("eligibilityCriteria") or ""
    arms_mod = ps.get("armsInterventionsModule") or {}
    interventions = arms_mod.get("interventions") or []
    out_mod = ps.get("outcomesModule") or {}
    primary = out_mod.get("primaryOutcomes") or []
    secondary = out_mod.get("secondaryOutcomes") or []

    parts = []
    if title:
        parts.append(f"Title: {title}")
    if conditions:
        parts.append("Conditions: " + ", ".join(str(c) for c in conditions))
    if brief:
        parts.append(f"Summary: {brief}")
    if detail:
        parts.append(f"Detailed Description: {detail}")
    if interventions:
        ints = []
        for i in interventions:
            if isinstance(i, dict):
                name = i.get("name") or i.get("type") or ""
                d = i.get("description") or ""
                ints.append(f"- {name}: {d}" if d else f"- {name}")
        if ints:
            parts.append("Interventions:\n" + "\n".join(ints))
    if elig:
        parts.append("Eligibility Criteria:\n" + elig)
    if primary:
        parts.append("Primary Outcomes:\n" + _flatten_outcomes(primary))
    if secondary:
        parts.append("Secondary Outcomes:\n" + _flatten_outcomes(secondary))
    text = "\n\n".join(p for p in parts if p).strip()
    if len(text) < 300:
        return None
    return {"text": text, "source": f"ctgov:{nct}"}


_Q_BRIEF = [
    "What is the dental clinical trial '{t}' about?",
    "Summarise the dental clinical trial titled '{t}'.",
    "Describe the purpose and rationale of the clinical trial '{t}'.",
    "What is the aim of the dental study '{t}'?",
]
_Q_ELIG = [
    "What are the eligibility criteria for the dental clinical trial '{t}'?",
    "Who is eligible to participate in the clinical trial '{t}'?",
    "List the inclusion and exclusion criteria for the dental study '{t}'.",
    "What patient population does the dental trial '{t}' enroll?",
]
_Q_OUTCOMES = [
    "What are the primary outcome measures of the dental trial '{t}'?",
    "How is success measured in the dental clinical trial '{t}'?",
    "What primary endpoints are assessed in the trial '{t}'?",
]
_Q_METHODS = [
    "Describe the methodology of the clinical trial on {c}: '{t}'.",
    "How is the dental clinical trial '{t}' on {c} designed and conducted?",
    "Explain the study design for the trial '{t}' investigating {c}.",
]


def study_to_sft(study: dict) -> Iterator[dict]:
    """Extract structured Q&A from CT.gov record. NCT id used as RNG seed for stability."""
    ps = study.get("protocolSection") or {}
    ident = ps.get("identificationModule") or {}
    nct = ident.get("nctId") or ""
    title = ident.get("officialTitle") or ident.get("briefTitle") or ""
    desc_mod = ps.get("descriptionModule") or {}
    brief = desc_mod.get("briefSummary") or ""
    detail = desc_mod.get("detailedDescription") or ""
    elig_mod = ps.get("eligibilityModule") or {}
    elig = elig_mod.get("eligibilityCriteria") or ""
    cond_mod = ps.get("conditionsModule") or {}
    conditions = cond_mod.get("conditions") or []
    out_mod = ps.get("outcomesModule") or {}
    primary = out_mod.get("primaryOutcomes") or []

    rng = random.Random(nct or title)
    source = f"ctgov:{nct}"
    topic = "Clinical Research"

    if title and brief and len(brief) > 80:
        yield {
            "question": rng.choice(_Q_BRIEF).format(t=title),
            "answer": brief,
            "source": source,
            "topic": topic,
        }
    if title and elig and len(elig) > 80:
        yield {
            "question": rng.choice(_Q_ELIG).format(t=title),
            "answer": elig,
            "source": source,
            "topic": topic,
        }
    if title and primary:
        prim_text = _flatten_outcomes(primary)
        if len(prim_text) > 50:
            yield {
                "question": rng.choice(_Q_OUTCOMES).format(t=title),
                "answer": prim_text,
                "source": source,
                "topic": topic,
            }
    if title and conditions and detail and len(detail) > 80:
        cond_str = ", ".join(str(c) for c in conditions[:4])
        yield {
            "question": rng.choice(_Q_METHODS).format(t=title, c=cond_str),
            "answer": detail,
            "source": source,
            "topic": topic,
        }


def main() -> int:
    if not RAW.exists():
        print(f"[ctgov->cpt] missing input: {RAW}")
        return 1
    n_in = n_cpt = n_sft = 0
    with RAW.open(encoding="utf-8") as f_in, \
            OUT_CPT.open("w", encoding="utf-8") as f_cpt, \
            OUT_SFT.open("w", encoding="utf-8") as f_sft:
        for line in f_in:
            line = line.strip()
            if not line:
                continue
            try:
                study = json.loads(line)
            except Exception:
                continue
            n_in += 1
            cpt = study_to_cpt(study)
            if cpt:
                f_cpt.write(json.dumps(cpt, ensure_ascii=False) + "\n")
                n_cpt += 1
            for sft in study_to_sft(study):
                f_sft.write(json.dumps(sft, ensure_ascii=False) + "\n")
                n_sft += 1
    print(f"[ctgov->cpt] read {n_in} -> CPT={n_cpt} SFT={n_sft}")
    print(f"  -> {OUT_CPT}")
    print(f"  -> {OUT_SFT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
