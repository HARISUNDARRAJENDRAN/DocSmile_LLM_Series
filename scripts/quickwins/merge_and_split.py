"""Merge all quick-wins SFT JSONLs into a single train/val split.

Output:
  rl_prepared/rl_sft_v2_quickwins.jsonl                (combined, full)
  rl_prepared/rl_sft_v2_quickwins_train.jsonl
  rl_prepared/rl_sft_v2_quickwins_val.jsonl

CPT chunks are merged separately into:
  cpt_prepared/quick_wins_cpt/combined_cpt.jsonl
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from .common.io import CPT_OUT_DIR, OUT_DIR, ROOT, read_jsonl, write_jsonl
from .common.schema import is_valid_sft


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--val-frac", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out-name", default="rl_sft_v2_quickwins")
    args = parser.parse_args(argv)

    rl_dir = ROOT / "rl_prepared"
    rl_dir.mkdir(parents=True, exist_ok=True)

    sft_files = sorted(p for p in OUT_DIR.glob("*.jsonl") if p.is_file())
    print(f"[merge] SFT files: {len(sft_files)}")
    sft_rows: list[dict] = []
    for p in sft_files:
        rows_before = len(sft_rows)
        for row in read_jsonl(p):
            if is_valid_sft(row):
                sft_rows.append(row)
        print(f"  + {p.name}: +{len(sft_rows)-rows_before}")

    print(f"[merge] total SFT rows: {len(sft_rows)}")
    random.seed(args.seed)
    random.shuffle(sft_rows)
    n_val = max(1, int(len(sft_rows) * args.val_frac))
    val = sft_rows[:n_val]
    train = sft_rows[n_val:]

    out_full = rl_dir / f"{args.out_name}.jsonl"
    out_train = rl_dir / f"{args.out_name}_train.jsonl"
    out_val = rl_dir / f"{args.out_name}_val.jsonl"
    write_jsonl(out_full, sft_rows)
    write_jsonl(out_train, train)
    write_jsonl(out_val, val)
    print(f"[merge] wrote {out_full} ({len(sft_rows)})")
    print(f"[merge] wrote {out_train} ({len(train)})")
    print(f"[merge] wrote {out_val} ({len(val)})")

    # CPT merge
    cpt_files = sorted(p for p in CPT_OUT_DIR.glob("*.jsonl") if p.is_file())
    cpt_rows: list[dict] = []
    for p in cpt_files:
        for row in read_jsonl(p):
            if row.get("text") and row.get("source"):
                cpt_rows.append(row)
    if cpt_rows:
        out_cpt = CPT_OUT_DIR / "combined_cpt.jsonl"
        write_jsonl(out_cpt, cpt_rows)
        print(f"[merge] wrote {out_cpt} ({len(cpt_rows)})")

    # Source/topic breakdown
    from collections import Counter
    src = Counter(r["source"] for r in sft_rows)
    topic = Counter(r["topic"] for r in sft_rows)
    print("\n[merge] sources:")
    for s, c in src.most_common():
        print(f"  {s:<36} {c:>6}")
    print("\n[merge] top topics:")
    for t, c in topic.most_common(20):
        print(f"  {t:<36} {c:>6}")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
