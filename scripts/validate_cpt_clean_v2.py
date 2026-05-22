#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path


BAD_PHRASES = [
    "here is the cleaned",
    "i have cleaned",
    "as an ai",
    "output only",
    "text to clean",
]


def repeated_line_fraction(text: str) -> float:
    lines = [re.sub(r"\s+", " ", line.strip()).lower() for line in text.splitlines()]
    lines = [line for line in lines if 5 <= len(line) <= 120]
    if not lines:
        return 0.0
    counts = Counter(lines)
    repeated = sum(count for count in counts.values() if count >= 3)
    return round(repeated / len(lines), 4)


def validate_pair(src: str, cleaned: str) -> dict:
    low = cleaned.lower()
    warnings: list[str] = []
    ratio = len(cleaned) / max(1, len(src))
    rep_frac = repeated_line_fraction(cleaned)
    if not cleaned.strip():
        warnings.append("empty")
    if ratio < 0.35:
        warnings.append("very_low_retention")
    elif ratio < 0.55:
        warnings.append("low_retention_review")
    if ratio > 1.08:
        warnings.append("unexpected_growth")
    if rep_frac > 0.08:
        warnings.append("repeated_layout_leftover")
    if any(phrase in low[:1000] for phrase in BAD_PHRASES):
        warnings.append("assistant_or_prompt_leak")
    if cleaned.count("<SKIP>"):
        warnings.append("skip_token_leak")
    return {
        "input_chars": len(src),
        "output_chars": len(cleaned),
        "retention_ratio": round(ratio, 4),
        "repeated_line_fraction": rep_frac,
        "warnings": warnings,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate cleaned CPT text corpus.")
    parser.add_argument("--input-dir", default="cpt_prepared/core_cpt_text")
    parser.add_argument("--cleaned-dir", default="cpt_prepared/core_cpt_text_gemini_clean_v2")
    parser.add_argument("--report-json", default="cpt_prepared/core_cpt_text_gemini_clean_v2/_reports/validation_report.json")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    cleaned_dir = Path(args.cleaned_dir)
    report_path = Path(args.report_json)
    report_path.parent.mkdir(parents=True, exist_ok=True)

    report: dict[str, dict] = {}
    warning_counts: Counter[str] = Counter()
    missing = 0
    for src_path in sorted(input_dir.glob("*.txt")):
        cleaned_path = cleaned_dir / src_path.name
        if not cleaned_path.exists():
            missing += 1
            report[src_path.name] = {"missing_cleaned": True, "warnings": ["missing_cleaned"]}
            warning_counts["missing_cleaned"] += 1
            continue
        src = src_path.read_text(encoding="utf-8", errors="ignore")
        cleaned = cleaned_path.read_text(encoding="utf-8", errors="ignore")
        row = validate_pair(src, cleaned)
        report[src_path.name] = row
        warning_counts.update(row["warnings"])

    summary = {
        "input_dir": str(input_dir),
        "cleaned_dir": str(cleaned_dir),
        "files": len(report),
        "missing": missing,
        "warning_counts": dict(warning_counts),
        "review_files": [
            name for name, row in report.items()
            if row.get("warnings") and row.get("warnings") != []
        ][:100],
    }
    payload = {"summary": summary, "files": report}
    report_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"Report: {report_path}")


if __name__ == "__main__":
    main()
