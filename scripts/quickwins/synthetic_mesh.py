"""Synthetic dental Q&A from MeSH descriptors (no LLM; deterministic templates).

For each dental MeSH descriptor we generate 1-3 structured Q&A using the descriptor's
scope note (definition) and tree position. Output schema matches existing SFT.

We fetch MeSH descriptor data via NCBI E-utilities (efetch db=mesh) for known
dental MeSH UIDs. List is curated for top-level dental concepts; can be expanded.

This uses ZERO Gemini quota — useful when API budget is tight.
"""
from __future__ import annotations

import json
import os
import random
import sys
import time
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "cpt_prepared" / "synthetic_mesh_dental"
OUT_DIR.mkdir(parents=True, exist_ok=True)

EUTILS = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
TOOL = "docsmile-quickwins"
EMAIL = os.environ.get("NCBI_EMAIL", "docsmile@example.com")
API_KEY = os.environ.get("NCBI_API_KEY", "")

# Dental MeSH descriptors — curated dental specialty + condition + procedure terms.
DENTAL_MESH_TERMS = [
    # Specialties / fields
    "Dentistry", "Endodontics", "Periodontics", "Orthodontics", "Prosthodontics",
    "Pediatric Dentistry", "Preventive Dentistry", "Operative Dentistry",
    "Forensic Dentistry", "Geriatric Dentistry", "Dental Hygiene",
    "Esthetic Dentistry", "Public Health Dentistry",
    # Major condition groups
    "Tooth Diseases", "Mouth Diseases", "Periodontal Diseases",
    "Dental Caries", "Periodontitis", "Gingivitis", "Pulpitis",
    "Periapical Diseases", "Tooth Abscess", "Tooth Loss", "Tooth Avulsion",
    "Tooth Erosion", "Tooth Wear", "Tooth Resorption", "Dental Fluorosis",
    "Malocclusion", "Cross Bite", "Open Bite", "Overbite",
    "Temporomandibular Joint Disorders", "Bruxism", "Halitosis",
    "Xerostomia", "Trismus",
    # Tissues / anatomy
    "Tooth", "Tooth, Deciduous", "Molar", "Molar, Third", "Incisor",
    "Bicuspid", "Cuspid", "Dental Enamel", "Dentin", "Dental Pulp",
    "Dental Cementum", "Gingiva", "Periodontal Ligament",
    "Dental Plaque", "Dental Calculus", "Mouth", "Mouth Mucosa",
    "Tongue", "Salivary Glands", "Mandible", "Maxilla",
    "Alveolar Process", "Temporomandibular Joint",
    # Procedures
    "Tooth Extraction", "Root Canal Therapy", "Dental Restoration, Permanent",
    "Dental Restoration, Temporary", "Crowns", "Dental Bonding",
    "Dental Veneers", "Dentures", "Dental Implants", "Tooth Bleaching",
    "Dental Prophylaxis", "Dental Scaling", "Root Planing",
    "Orthodontic Appliances", "Pit and Fissure Sealants",
    "Dental Cavity Preparation", "Dental Pulp Capping",
    # Materials
    "Composite Resins", "Dental Amalgam", "Dental Cements",
    "Glass Ionomer Cements", "Dental Materials", "Dental Alloys",
    # Cancers / lesions
    "Mouth Neoplasms", "Tongue Neoplasms", "Lip Neoplasms",
    "Oral Submucous Fibrosis", "Leukoplakia, Oral", "Erythroplasia",
    "Oral Lichen Planus", "Candidiasis, Oral", "Stomatitis",
    "Cleft Lip", "Cleft Palate",
]


def _qs(params: dict) -> str:
    base = {"tool": TOOL, "email": EMAIL}
    if API_KEY:
        base["api_key"] = API_KEY
    base.update(params)
    return urllib.parse.urlencode(base)


def _http_get(url: str, max_retries: int = 4) -> bytes:
    backoff = 1.0
    for attempt in range(max_retries):
        try:
            with urllib.request.urlopen(url, timeout=60) as resp:
                return resp.read()
        except Exception:
            if attempt == max_retries - 1:
                raise
            time.sleep(backoff)
            backoff *= 2
    raise RuntimeError("unreachable")


def _rate_sleep() -> None:
    time.sleep(0.35 if not API_KEY else 0.12)


def _search_uid(term: str) -> str | None:
    params = {"db": "mesh", "term": f'"{term}"[MeSH Major Topic]',
              "retmode": "json", "retmax": 5}
    url = f"{EUTILS}/esearch.fcgi?{_qs(params)}"
    try:
        data = json.loads(_http_get(url).decode("utf-8"))
        ids = data.get("esearchresult", {}).get("idlist", [])
        return ids[0] if ids else None
    except Exception:
        return None


def _fetch_descriptor(uid: str) -> dict | None:
    params = {"db": "mesh", "id": uid, "retmode": "xml"}
    url = f"{EUTILS}/efetch.fcgi?{_qs(params)}"
    try:
        raw = _http_get(url).decode("utf-8", errors="replace")
        # The mesh efetch returns plain text (not XML) for db=mesh. Parse it.
        return _parse_mesh_text(raw)
    except Exception:
        return None


def _parse_mesh_text(raw: str) -> dict | None:
    """efetch db=mesh returns plain text. Format:

        <N>: <Heading>
        <Scope note line 1>
        <Scope note line 2 ...>
        <blank>
        Subheadings:
            ...
        Tree Number(s): A14, C07
        See Also:
            ...
        Entry Term: ...

    We extract: heading, scope_note (joined), tree_numbers, entry_terms.
    """
    if not raw:
        return None
    lines = raw.splitlines()
    # Find heading line: starts with a digit and a colon
    heading = ""
    heading_idx = -1
    for i, ln in enumerate(lines):
        m = ln.strip()
        if m and m[0].isdigit() and ": " in m:
            head_part = m.split(": ", 1)[1].strip()
            if head_part and not head_part.endswith(")"):
                heading = head_part
                heading_idx = i
                break
    if not heading:
        return None
    # Scope note: lines after heading until blank or section keyword
    scope_lines: list[str] = []
    j = heading_idx + 1
    section_starts = (
        "Subheadings:", "Tree Number", "See Also", "Entry Term:",
        "Year introduced", "All MeSH Categories", "Allowable Qualifiers",
        "Pharm Action", "Public MeSH Note", "Online Note",
        "History Note", "Date of Entry", "Unique ID", "Annotation:",
    )
    while j < len(lines):
        ln = lines[j].rstrip()
        if not ln.strip():
            break
        if any(ln.startswith(p) for p in section_starts):
            break
        scope_lines.append(ln.strip())
        j += 1
    scope_note = " ".join(scope_lines).strip()
    # Tree numbers
    tree_numbers: list[str] = []
    entry_terms: list[str] = []
    for ln in lines:
        if ln.startswith("Tree Number"):
            after = ln.split(":", 1)[1].strip() if ":" in ln else ""
            for tn in after.split(","):
                tn = tn.strip()
                if tn:
                    tree_numbers.append(tn)
        elif ln.startswith("Entry Term:"):
            term = ln.split(":", 1)[1].strip()
            if term:
                entry_terms.append(term)
    return {
        "heading": heading,
        "scope_note": scope_note,
        "tree_numbers": tree_numbers,
        "entry_terms": entry_terms,
        "raw": raw,
    }


_Q_TEMPLATES_DEF = [
    "What is {term}?",
    "Define {term} in dental terminology.",
    "Provide a clinical description of {term}.",
    "Explain {term} as used in dentistry.",
    "What does the dental term {term} mean?",
]
_Q_TEMPLATES_USE = [
    "How is the concept of {term} relevant in dental practice?",
    "When would a dentist consider {term} during patient assessment?",
    "Describe the clinical significance of {term}.",
]


def _sft_for_descriptor(d: dict) -> list[dict]:
    heading = d.get("heading") or ""
    scope = d.get("scope_note") or ""
    if not heading or len(scope) < 60:
        return []
    rng = random.Random(heading)
    rows = []
    q1 = rng.choice(_Q_TEMPLATES_DEF).format(term=heading)
    rows.append({
        "question": q1,
        "answer": scope,
        "source": f"mesh:{heading}",
        "topic": "Dental Terminology",
    })
    # Include entry-term synonyms in a second prompt if available
    entries = d.get("entry_terms") or []
    if entries:
        syns = ", ".join(e.strip() for e in entries[:8] if e.strip())
        rows.append({
            "question": f"What other names or synonyms are used for {heading} in MeSH?",
            "answer": f"In MeSH, {heading} is also referenced by entry terms: {syns}. " + scope,
            "source": f"mesh:{heading}",
            "topic": "Dental Terminology",
        })
    # Clinical significance prompt
    rows.append({
        "question": rng.choice(_Q_TEMPLATES_USE).format(term=heading),
        "answer": scope,
        "source": f"mesh:{heading}",
        "topic": "Dental Terminology",
    })
    return rows


def main() -> int:
    out_path = OUT_DIR / "synthetic_mesh_dental_sft.jsonl"
    state_path = OUT_DIR / "state.json"
    state = {}
    if state_path.exists():
        try:
            state = json.loads(state_path.read_text())
        except Exception:
            state = {}
    done_terms = set(state.get("done_terms", []))
    descriptors = state.get("descriptors", {})

    # Resolve UIDs and fetch records for terms not yet done.
    for term in DENTAL_MESH_TERMS:
        if term in done_terms:
            continue
        uid = _search_uid(term)
        _rate_sleep()
        if not uid:
            done_terms.add(term)
            continue
        d = _fetch_descriptor(uid)
        _rate_sleep()
        if d:
            descriptors[term] = d
        done_terms.add(term)
        if len(done_terms) % 5 == 0:
            state["done_terms"] = sorted(done_terms)
            state["descriptors"] = descriptors
            state_path.write_text(json.dumps(state)[:5_000_000])
            print(f"[mesh] fetched {len(descriptors)}/{len(DENTAL_MESH_TERMS)}")
    state["done_terms"] = sorted(done_terms)
    state["descriptors"] = descriptors
    state_path.write_text(json.dumps(state)[:5_000_000])

    # Emit SFT rows: always re-parse raw to pick up parser improvements.
    n = 0
    with out_path.open("w", encoding="utf-8") as f:
        for term, d in descriptors.items():
            raw = d.get("raw") or ""
            if not raw:
                continue
            parsed = _parse_mesh_text(raw)
            if not parsed:
                continue
            for rec in _sft_for_descriptor(parsed):
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                n += 1
    print(f"[mesh] wrote {n} synthetic SFT rows -> {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
