import argparse
import json
import os
import time
from pathlib import Path
from urllib import request
from urllib.error import HTTPError, URLError


SYSTEM_PROMPT = (
    "You are a careful copy editor for dental textbooks. "
    "Fix only obvious OCR/spelling errors and spacing issues. "
    "Do NOT paraphrase, summarize, or change meaning. "
    "Preserve all technical terms, abbreviations, numbers, and units. "
    "Preserve paragraph breaks. Output only the corrected text."
)


def load_env_key(env_path: Path):
    candidates = ["GEMINI_API_KEY", "GOOGLE_API_KEY", "GENAI_API_KEY", "API_KEY"]

    for name in candidates:
        if os.environ.get(name):
            return os.environ.get(name), name

    if env_path.exists():
        for line in env_path.read_text(encoding="utf-8", errors="ignore").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            if key in candidates and value:
                return value, key

    return None, None


def split_paragraphs(text: str):
    paras = [p.strip() for p in text.split("\n\n") if p.strip()]
    return paras


def chunk_paragraphs(paragraphs, max_words):
    chunks = []
    buf = []
    count = 0
    for p in paragraphs:
        w = len(p.split())
        if buf and count + w > max_words:
            chunks.append("\n\n".join(buf))
            buf = [p]
            count = w
        else:
            buf.append(p)
            count += w
    if buf:
        chunks.append("\n\n".join(buf))
    return chunks


def call_gemini(api_key, model, text, max_output_tokens, retries=3, sleep_s=2.0):
    url = (
        "https://generativelanguage.googleapis.com/v1beta/models/"
        f"{model}:generateContent?key={api_key}"
    )

    payload = {
        "systemInstruction": {"parts": [{"text": SYSTEM_PROMPT}]},
        "contents": [
            {
                "role": "user",
                "parts": [
                    {
                        "text": "Clean the text below. Return only corrected text.\n\n" + text
                    }
                ],
            }
        ],
        "generationConfig": {
            "temperature": 0.1,
            "topP": 0.95,
            "topK": 40,
            "maxOutputTokens": max_output_tokens,
        },
    }

    data = json.dumps(payload).encode("utf-8")
    req = request.Request(url, data=data, headers={"Content-Type": "application/json"})

    for attempt in range(1, retries + 1):
        try:
            with request.urlopen(req, timeout=120) as resp:
                raw = resp.read().decode("utf-8")
            parsed = json.loads(raw)
            parts = parsed.get("candidates", [])[0].get("content", {}).get("parts", [])
            if not parts:
                raise ValueError("Empty response parts")
            return parts[0].get("text", "").strip()
        except (HTTPError, URLError, ValueError, IndexError, KeyError) as exc:
            if attempt == retries:
                raise exc
            time.sleep(sleep_s * attempt)

    return ""


def validate_chunk(input_text, output_text):
    warnings = []
    if not output_text:
        warnings.append("empty_output")
        return warnings

    ratio = len(output_text) / max(1, len(input_text))
    if ratio < 0.8 or ratio > 1.2:
        warnings.append(f"length_ratio_{ratio:.2f}")

    lower = output_text.strip().lower()
    if lower.startswith("sure") or lower.startswith("here"):
        warnings.append("assistant_preface_detected")

    if "As an AI" in output_text:
        warnings.append("assistant_disclaimer_detected")

    return warnings


def main():
    parser = argparse.ArgumentParser(description="Minimal Gemini-based cleaning for CPT text.")
    parser.add_argument("--input-dir", default="cpt_prepared/core_cpt_text", help="Input .txt directory")
    parser.add_argument("--output-dir", default="cpt_prepared/core_cpt_text_min", help="Output directory")
    parser.add_argument("--allowlist", default="", help="Optional file with one .txt filename per line")
    parser.add_argument("--max-files", type=int, default=0, help="Process only first N files")
    parser.add_argument("--model", default=os.environ.get("GEMINI_MODEL", "gemini-1.5-flash"))
    parser.add_argument("--chunk-words", type=int, default=600)
    parser.add_argument("--max-output-tokens", type=int, default=2048)
    parser.add_argument("--sleep", type=float, default=1.0)
    parser.add_argument("--dry-run", action="store_true")

    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    env_path = root / ".env"
    api_key, key_name = load_env_key(env_path)
    if not api_key and not args.dry_run:
        raise SystemExit("Missing API key. Set GEMINI_API_KEY in .env or environment.")

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.allowlist:
        allow_path = Path(args.allowlist)
        names = [
            line.strip() for line in allow_path.read_text(encoding="utf-8").splitlines() if line.strip()
        ]
        files = [input_dir / name for name in names]
    else:
        files = sorted([p for p in input_dir.glob("*.txt") if p.is_file()])

    if args.max_files > 0:
        files = files[: args.max_files]

    if args.dry_run:
        print(f"Dry run. Files: {len(files)}")
        for p in files:
            print(p.name)
        return

    report = {"files": [], "warnings": []}

    for path in files:
        text = path.read_text(encoding="utf-8", errors="ignore")
        paragraphs = split_paragraphs(text)
        chunks = chunk_paragraphs(paragraphs, args.chunk_words)

        cleaned_chunks = []
        file_warnings = []

        for idx, chunk in enumerate(chunks):
            cleaned = call_gemini(
                api_key,
                args.model,
                chunk,
                max_output_tokens=args.max_output_tokens,
            )
            cleaned_chunks.append(cleaned)

            warnings = validate_chunk(chunk, cleaned)
            if warnings:
                file_warnings.append({"chunk": idx, "warnings": warnings})

            time.sleep(args.sleep)

        cleaned_text = "\n\n".join([c for c in cleaned_chunks if c])
        out_path = output_dir / path.name
        out_path.write_text(cleaned_text, encoding="utf-8")

        report["files"].append(
            {
                "file": path.name,
                "chunks": len(chunks),
                "warnings": file_warnings,
            }
        )

        if file_warnings:
            report["warnings"].append({"file": path.name, "issues": file_warnings})

    (output_dir / "gemini_clean_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=True), encoding="utf-8"
    )

    print(f"Cleaned files: {len(files)}")
    print(f"Output: {output_dir}")
    print(f"Report: {output_dir / 'gemini_clean_report.json'}")


if __name__ == "__main__":
    main()
