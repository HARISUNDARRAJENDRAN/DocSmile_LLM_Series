import json
from pathlib import Path


def validate_pair(src_text: str, cleaned_text: str):
    stats = {}
    stats["input_chars"] = len(src_text)
    stats["output_chars"] = len(cleaned_text)
    stats["ratio"] = round((len(cleaned_text) / max(1, len(src_text))), 4)
    stats["empty"] = cleaned_text.strip() == ""

    warnings = []
    if stats["ratio"] < 0.7:
        warnings.append("low_retention")
    if stats["ratio"] > 1.1:
        warnings.append("unexpected_growth")
    if stats["empty"]:
        warnings.append("empty_output")
    if cleaned_text.strip().lower().startswith("sure"):
        warnings.append("assistant_preface_detected")

    stats["warnings"] = warnings
    return stats


def main():
    root = Path(__file__).resolve().parents[1]
    src_dir = root / "cpt_prepared" / "core_cpt_text"
    cleaned_dir = root / "cpt_prepared" / "core_cpt_text_cleaned"

    report = {}
    for src_path in sorted(src_dir.glob("*.txt")):
        cleaned_path = cleaned_dir / src_path.name
        if not cleaned_path.exists():
            report[src_path.name] = {"missing_cleaned": True}
            continue
        src_text = src_path.read_text(encoding="utf-8", errors="ignore")
        cleaned_text = cleaned_path.read_text(encoding="utf-8", errors="ignore")
        report[src_path.name] = validate_pair(src_text, cleaned_text)

    out_path = cleaned_dir / "validation_all.json"
    out_path.write_text(json.dumps(report, indent=2, ensure_ascii=True), encoding="utf-8")
    print(f"Validation report: {out_path}")


if __name__ == "__main__":
    main()
