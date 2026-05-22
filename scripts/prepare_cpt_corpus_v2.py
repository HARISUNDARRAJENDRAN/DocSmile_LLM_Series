#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from prepare_cpt_text import clean_markdown


def safe_name(group: str, stem: str) -> str:
    name = f"{group}__{stem}"
    name = re.sub(r'[<>:"/\\|?*\x00-\x1f]', "_", name)
    name = re.sub(r"\s+", " ", name).strip()
    return name[:230] + ".txt"


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a larger raw CPT text corpus from markdown book folders.")
    parser.add_argument("--input-dirs", nargs="+", default=["core_cpt", "rl", "selective_cpt"])
    parser.add_argument("--output-dir", default="cpt_prepared/cpt_raw_text_v2")
    parser.add_argument("--manifest-json", default="cpt_prepared/cpt_raw_text_v2/manifest.json")
    parser.add_argument("--keep-figures", action="store_true")
    parser.add_argument("--min-chars", type=int, default=2000)
    parser.add_argument("--max-files", type=int, default=0)
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = Path(args.manifest_json)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    records: list[dict] = []
    files: list[tuple[str, Path]] = []
    for input_dir in args.input_dirs:
        root = Path(input_dir)
        group = root.name
        files.extend((group, path) for path in sorted(root.glob("*.md")))
    if args.max_files:
        files = files[: args.max_files]

    for group, md_path in files:
        raw = md_path.read_text(encoding="utf-8", errors="ignore")
        cleaned = clean_markdown(raw, keep_figures=args.keep_figures)
        record = {
            "group": group,
            "source_path": str(md_path),
            "source_chars": len(raw),
            "prepared_chars": len(cleaned),
            "status": "written",
        }
        if len(cleaned) < args.min_chars:
            record["status"] = "skipped_too_short"
            records.append(record)
            continue
        out_name = safe_name(group, md_path.stem)
        out_path = out_dir / out_name
        out_path.write_text(cleaned.strip() + "\n", encoding="utf-8")
        record["output_path"] = str(out_path)
        record["output_name"] = out_name
        records.append(record)

    summary = {
        "input_dirs": args.input_dirs,
        "output_dir": str(out_dir),
        "files_seen": len(files),
        "files_written": sum(1 for r in records if r["status"] == "written"),
        "files_skipped": sum(1 for r in records if r["status"] != "written"),
        "total_prepared_chars": sum(r["prepared_chars"] for r in records if r["status"] == "written"),
    }
    manifest_path.write_text(json.dumps({"summary": summary, "files": records}, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
