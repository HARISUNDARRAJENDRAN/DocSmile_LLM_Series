"""Orchestrator for the quick-wins phase.

Runs each loader, normalises, dedups (in-batch + against existing SFT corpus),
writes a per-dataset JSONL, and emits a summary report.

Usage:
  python -m scripts.quickwins.run_all [--only NAME[,NAME...]]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from pathlib import Path

from .common.dedup import Deduper, existing_sft_deduper
from .common.io import CPT_OUT_DIR, EXISTING_SFT, LOG_DIR, OUT_DIR, write_jsonl
from .common.schema import is_valid_sft


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", help="Comma-separated list of loader names to run.")
    parser.add_argument("--skip", help="Comma-separated list of loader names to skip.")
    parser.add_argument("--no-existing-dedup", action="store_true",
                        help="Skip seeding the deduper from the existing SFT corpus.")
    args = parser.parse_args(argv)

    # Build the loader registry lazily so a broken loader doesn't kill imports.
    loaders: list[tuple[str, callable, str]] = []

    from .loaders import (
        medmcqa, headqa, periodontal_reasoning, chatdoctor, lexicon_shift,
        small_hf, pubmedqa,
    )

    loaders.append(("medmcqa_dental", medmcqa.load, "sft"))
    loaders.append(("headqa_dental", headqa.load, "sft"))
    loaders.append(("periodontal_reasoning_40k", periodontal_reasoning.load, "sft"))
    loaders.append(("chatdoctor_dental", chatdoctor.load, "sft"))
    loaders.append(("lexicon_shift_qna", lexicon_shift.load_qna_sft, "sft"))
    loaders.append(("lexicon_shift_fb_articles", lexicon_shift.load_fb_sft, "cpt"))
    loaders.append(("lexicon_shift_sinhala_cpt", lexicon_shift.load_sinhala_cpt, "cpt"))
    loaders.append(("birdiebyte_dental_implants", small_hf.load_birdiebyte, "sft"))
    loaders.append(("jonathankang_dental_qa", small_hf.load_jonathankang, "sft"))
    loaders.append(("emilykang_dentistry", small_hf.load_emilykang, "sft"))
    loaders.append(("vuha2003_medmcqa_dental", small_hf.load_vuha2003, "sft"))
    loaders.append(("pubmedqa_dental", pubmedqa.load, "sft"))

    only = {x.strip() for x in (args.only or "").split(",") if x.strip()}
    skip = {x.strip() for x in (args.skip or "").split(",") if x.strip()}

    deduper = Deduper() if args.no_existing_dedup else existing_sft_deduper()
    print(f"[run_all] dedup seed size: {len(deduper.seen)}")

    summary: list[dict] = []
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    run_log = LOG_DIR / f"run_{time.strftime('%Y%m%d_%H%M%S')}.log"

    for name, fn, mode in loaders:
        if only and name not in only:
            continue
        if name in skip:
            continue
        out_path = (CPT_OUT_DIR if mode == "cpt" else OUT_DIR) / f"{name}.jsonl"
        if out_path.exists():
            # If already generated, count and skip to allow resuming
            try:
                n = sum(1 for _ in out_path.open(encoding="utf-8"))
                summary.append({"name": name, "mode": mode, "kept": n, "status": "cached"})
                # seed dedup with cached questions
                if mode == "sft":
                    from .common.dedup import qhash
                    with out_path.open(encoding="utf-8") as f:
                        for line in f:
                            try:
                                obj = json.loads(line)
                                deduper.add(obj["question"])
                            except Exception:
                                pass
                print(f"[run_all] {name}: cached -> {n} rows (skipping fresh fetch)")
                continue
            except Exception:
                pass

        print(f"[run_all] === {name} ({mode}) ===")
        start = time.time()
        kept = 0
        dropped_invalid = 0
        dropped_dup = 0
        errors = 0
        try:
            rows_out: list[dict] = []
            for row in fn():
                if mode == "sft":
                    if not is_valid_sft(row):
                        dropped_invalid += 1
                        continue
                    if not deduper.add(row["question"]):
                        dropped_dup += 1
                        continue
                    rows_out.append(row)
                else:
                    # CPT: dedup by text hash
                    text = row.get("text", "")
                    if not text:
                        dropped_invalid += 1
                        continue
                    if not deduper.add(text[:500]):
                        dropped_dup += 1
                        continue
                    rows_out.append(row)
                kept += 1
            write_jsonl(out_path, rows_out)
        except Exception as e:
            errors += 1
            print(f"[run_all] {name} FAILED: {e}")
            traceback.print_exc()
            with run_log.open("a", encoding="utf-8") as f:
                f.write(f"{name}: FAILED: {e}\n{traceback.format_exc()}\n")
        elapsed = time.time() - start
        rec = {"name": name, "mode": mode, "kept": kept,
               "dropped_invalid": dropped_invalid, "dropped_dup": dropped_dup,
               "errors": errors, "elapsed_sec": round(elapsed, 1)}
        summary.append(rec)
        with run_log.open("a", encoding="utf-8") as f:
            f.write(json.dumps(rec) + "\n")
        print(f"[run_all] {name}: kept={kept} dup={dropped_dup} inv={dropped_invalid} "
              f"err={errors} t={elapsed:.1f}s -> {out_path}")

    print("\n=== SUMMARY ===")
    total = 0
    for r in summary:
        print(f"  {r['name']:<32} kept={r.get('kept',0):>6}  "
              f"dup={r.get('dropped_dup',0):>5}  inv={r.get('dropped_invalid',0):>5}")
        total += r.get("kept", 0)
    print(f"  TOTAL kept: {total}")
    (LOG_DIR / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
