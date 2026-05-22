"""Print current dental training corpus stats.

Run anytime to see what's available across SFT and CPT, broken down by source.
"""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def _stats(path: Path, key: str = "question"):
    if not path.exists():
        return None
    n = 0
    src = Counter()
    topic = Counter()
    chars = 0
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except Exception:
                continue
            n += 1
            src[r.get("source", "?")] += 1
            topic[r.get("topic", "?")] += 1
            chars += len(r.get(key, "")) + len(r.get("answer", "")) + len(r.get("text", ""))
    return {"n": n, "src": src, "topic": topic, "chars": chars}


def main() -> int:
    paths = {
        # Existing baseline
        "rl_sft.jsonl (existing baseline)": ROOT / "rl_prepared" / "rl_sft.jsonl",
        "rl_dpo.jsonl (existing baseline)": ROOT / "rl_prepared" / "rl_dpo.jsonl",
        # Final merged
        "dental_sft_mega.jsonl (FINAL SFT)": ROOT / "rl_prepared" / "dental_sft_mega.jsonl",
        "dental_sft_mega_train.jsonl": ROOT / "rl_prepared" / "dental_sft_mega_train.jsonl",
        "dental_sft_mega_val.jsonl": ROOT / "rl_prepared" / "dental_sft_mega_val.jsonl",
        "dental_cpt_mega.jsonl (FINAL CPT)": ROOT / "cpt_prepared" / "dental_cpt_mega.jsonl",
        # Raw quick-wins per-dataset
        "quick_wins/medmcqa_dental.jsonl": ROOT / "cpt_prepared" / "quick_wins" / "medmcqa_dental.jsonl",
        "quick_wins/headqa_dental.jsonl": ROOT / "cpt_prepared" / "quick_wins" / "headqa_dental.jsonl",
        "quick_wins/chatdoctor_dental.jsonl": ROOT / "cpt_prepared" / "quick_wins" / "chatdoctor_dental.jsonl",
        "quick_wins/periodontal_reasoning_40k.jsonl": ROOT / "cpt_prepared" / "quick_wins" / "periodontal_reasoning_40k.jsonl",
        "quick_wins/pubmedqa_dental.jsonl": ROOT / "cpt_prepared" / "quick_wins" / "pubmedqa_dental.jsonl",
        "quick_wins/jonathankang_dental_qa.jsonl": ROOT / "cpt_prepared" / "quick_wins" / "jonathankang_dental_qa.jsonl",
        # CT.gov / PubMed
        "clinicaltrials_dental/ctgov_dental_sft.jsonl": ROOT / "cpt_prepared" / "clinicaltrials_dental" / "ctgov_dental_sft.jsonl",
        "clinicaltrials_dental/ctgov_dental_cpt.jsonl": ROOT / "cpt_prepared" / "clinicaltrials_dental" / "ctgov_dental_cpt.jsonl",
        "pubmed_dental/pubmed_dental_abstracts.jsonl (raw)": ROOT / "cpt_prepared" / "pubmed_dental" / "pubmed_dental_abstracts.jsonl",
        "pubmed_dental/pubmed_dental_sft.jsonl": ROOT / "cpt_prepared" / "pubmed_dental" / "pubmed_dental_sft.jsonl",
        "pubmed_dental/pubmed_dental_cpt.jsonl": ROOT / "cpt_prepared" / "pubmed_dental" / "pubmed_dental_cpt.jsonl",
        "statpearls_dental/statpearls_dental_chapters.jsonl": ROOT / "cpt_prepared" / "statpearls_dental" / "statpearls_dental_chapters.jsonl",
    }

    print("=" * 80)
    print("DOCSMILE DENTAL CORPUS — FILE INVENTORY")
    print("=" * 80)
    grand_n = 0
    for label, p in paths.items():
        if not p.exists():
            print(f"  [missing] {label}")
            continue
        s = _stats(p, key="question")
        if s is None:
            continue
        size_mb = p.stat().st_size / (1024 * 1024)
        print(f"  {label}: {s['n']:>7}  rows  ({size_mb:>6.1f} MB,  {s['chars']/1_000_000:.1f}M chars)")
        if "FINAL" in label:
            print(f"     top topics: {', '.join(f'{t}={c}' for t,c in s['topic'].most_common(8))}")
            print(f"     unique sources: {len(s['src'])}")
        if "FINAL SFT" in label:
            grand_n = s["n"]
    print()
    # PubMed in-progress
    state = ROOT / "cpt_prepared" / "pubmed_dental" / "state.json"
    if state.exists():
        try:
            st = json.loads(state.read_text())
            print(f"  PubMed scraper state: next_idx={st.get('next_idx', 0)}/{st.get('total', '?')}  parsed={st.get('n_parsed', 0)}")
        except Exception:
            pass
    print()
    print(f"GRAND TOTAL FINAL SFT: {grand_n} rows")
    return 0


if __name__ == "__main__":
    sys.exit(main())
