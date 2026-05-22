#!/usr/bin/env python3
"""
Restore destroyed cleaned files from raw source backups.

Files where cleaned size is < 5% of raw size are considered destroyed by
API failures and are restored from the raw .txt source files.

Also resets the cleaning_progress.json checkpoint so they get re-cleaned
on the next run.
"""

import json
from pathlib import Path
import argparse


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--cleaned-dir", default="cpt_prepared/core_cpt_text_cleaned")
    p.add_argument("--raw-dirs", nargs="+",
                   default=["cpt_prepared/core_cpt_text",
                            "cpt_prepared/selective_cpt_text"])
    p.add_argument("--ratio-threshold", type=float, default=0.05,
                   help="Files kept < this fraction of original are restored")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    cleaned_dir = Path(args.cleaned_dir)
    raw_lookup = {}
    for d in args.raw_dirs:
        for f in Path(d).glob("*.txt"):
            raw_lookup[f.name] = f

    restored = []
    skipped = []
    no_backup = []

    for cf in cleaned_dir.glob("*.txt"):
        cs = cf.stat().st_size
        raw = raw_lookup.get(cf.name)
        if not raw:
            no_backup.append(cf.name)
            continue
        rs = raw.stat().st_size
        if rs == 0:
            continue
        ratio = cs / rs
        if ratio < args.ratio_threshold:
            restored.append((cf.name, rs, cs, ratio))
            if not args.dry_run:
                # Read raw, write to cleaned dir (overwrite)
                raw_text = raw.read_text(encoding="utf-8", errors="ignore")
                tmp = cf.with_suffix(cf.suffix + ".tmp")
                tmp.write_text(raw_text, encoding="utf-8")
                tmp.replace(cf)
        else:
            skipped.append(cf.name)

    print(f"Restored: {len(restored)} files")
    print(f"Kept as-is: {len(skipped)} files")
    print(f"No raw backup: {len(no_backup)} files")

    if restored:
        print("\n=== Restored files (sorted by damage) ===")
        for name, rs, cs, r in sorted(restored, key=lambda x: x[3]):
            print(f"  {r*100:5.1f}%  was {cs:>8,} / {rs:>10,}  {name[:70]}")

    if not args.dry_run and restored:
        # Reset progress for restored files so they get re-cleaned
        ckpt_path = cleaned_dir / "cleaning_progress.json"
        if ckpt_path.exists():
            ckpt = json.loads(ckpt_path.read_text(encoding="utf-8"))
            restored_names = set(name for name, _, _, _ in restored)
            ckpt["done_files"] = [
                n for n in ckpt.get("done_files", []) if n not in restored_names
            ]
            ckpt_path.write_text(json.dumps(ckpt, indent=2), encoding="utf-8")
            print(f"\nReset checkpoint: {len(restored_names)} files removed from done_files")
            print(f"  Remaining done_files: {len(ckpt['done_files'])}")


if __name__ == "__main__":
    main()
