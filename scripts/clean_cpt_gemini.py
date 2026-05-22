import argparse
import json
import os
import time
import urllib.error
import urllib.request
from datetime import datetime
from pathlib import Path


def load_api_key(env_path: Path) -> str:
    if not env_path.exists():
        return os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY") or ""

    key = ""
    for line in env_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export "):].lstrip()
        if "=" not in line:
            continue
        name, value = line.split("=", 1)
        name = name.strip()
        value = value.strip().strip('"').strip("'")
        if name in ("GEMINI_API_KEY", "GOOGLE_API_KEY"):
            key = value
            break
    if not key:
        key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY") or ""
    return key


def build_prompt(text: str) -> str:
    return (
        "You are a careful text cleaner for medical textbooks.\n"
        "Task: fix obvious OCR and spelling errors only.\n"
        "Rules:\n"
        "- Do NOT paraphrase or rewrite sentences.\n"
        "- Keep medical terms, names, and numbers exactly unless clearly misspelled.\n"
        "- Fix broken words caused by spacing or line breaks.\n"
        "- Normalize spacing, but preserve paragraph breaks.\n"
        "- Remove stray artifacts like repeated page markers if present.\n"
        "Return ONLY the cleaned text.\n\n"
        "TEXT:\n"
        f"{text}"
    )


def call_gemini(api_key: str, model: str, text: str, max_retries: int = 3, sleep_sec: float = 0.5) -> str:
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={api_key}"
    payload = {
        "contents": [
            {
                "role": "user",
                "parts": [{"text": build_prompt(text)}],
            }
        ],
        "generationConfig": {
            "temperature": 0.1,
            "topP": 0.95,
        },
    }

    data = json.dumps(payload).encode("utf-8")
    headers = {"Content-Type": "application/json"}

    for attempt in range(max_retries):
        try:
            req = urllib.request.Request(url, data=data, headers=headers)
            with urllib.request.urlopen(req, timeout=120) as resp:
                raw = resp.read().decode("utf-8", errors="ignore")
                parsed = json.loads(raw)
                parts = parsed.get("candidates", [{}])[0].get("content", {}).get("parts", [])
                if not parts:
                    raise RuntimeError("Empty response from model")
                return "".join(p.get("text", "") for p in parts).strip()
        except (urllib.error.HTTPError, urllib.error.URLError, RuntimeError) as exc:
            if attempt == max_retries - 1:
                raise exc
            time.sleep(sleep_sec * (2 ** attempt))

    raise RuntimeError("Gemini API call failed after retries")


def chunk_by_paragraphs(text: str, max_words: int):
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks = []
    buf = []
    word_count = 0

    for p in paragraphs:
        p_words = p.split()
        if word_count + len(p_words) > max_words and buf:
            chunks.append("\n\n".join(buf))
            buf = []
            word_count = 0
        buf.append(p)
        word_count += len(p_words)

    if buf:
        chunks.append("\n\n".join(buf))

    return chunks


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

    stats["warnings"] = warnings
    return stats


def load_validation(report_path: Path) -> dict:
    if not report_path.exists():
        return {}
    try:
        payload = json.loads(report_path.read_text(encoding="utf-8", errors="ignore"))
    except json.JSONDecodeError:
        return {}
    return payload if isinstance(payload, dict) else {}


def write_validation(report_path: Path, validation: dict) -> None:
    report_path.write_text(json.dumps(validation, indent=2, ensure_ascii=True), encoding="utf-8")


def cleanup_partial_files(output_dir: Path) -> None:
    for partial_path in output_dir.glob("*.partial"):
        try:
            partial_path.unlink()
        except OSError:
            pass


def write_progress(progress_path: Path, progress: dict) -> None:
    progress_path.write_text(json.dumps(progress, indent=2, ensure_ascii=True), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description="Minimal Gemini-based cleaning for CPT text files.")
    parser.add_argument("--input-dir", required=True, help="Directory with .txt files")
    parser.add_argument("--output-dir", required=True, help="Directory for cleaned .txt files")
    parser.add_argument("--model", default="gemini-3.1-flash-lite-preview", help="Gemini model name")
    parser.add_argument("--include-list", default="", help="Optional path to a file list to process")
    parser.add_argument("--max-files", type=int, default=0, help="Limit number of files for a test run")
    parser.add_argument("--chunk-words", type=int, default=1200, help="Max words per request")
    parser.add_argument("--sleep-sec", type=float, default=0.5, help="Sleep between requests")
    parser.add_argument("--validate", action="store_true", help="Write validation report")
    parser.add_argument("--resume", action="store_true", help="Skip files already cleaned in output dir")
    parser.add_argument("--no-resume", action="store_true", help="Do not skip existing files")
    parser.add_argument("--continue-on-error", action="store_true", help="Continue on API errors; keep original chunk")

    args = parser.parse_args()

    root_dir = Path(__file__).resolve().parents[1]
    env_path = root_dir / ".env"
    api_key = load_api_key(env_path)
    if not api_key:
        raise RuntimeError("Missing GEMINI_API_KEY or GOOGLE_API_KEY in .env or environment")

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    cleanup_partial_files(output_dir)

    include = None
    if args.include_list:
        include = set()
        for line in Path(args.include_list).read_text(encoding="utf-8", errors="ignore").splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            include.add(line)

    txt_files = sorted([p for p in input_dir.glob("*.txt") if p.is_file()])
    if include is not None:
        txt_files = [p for p in txt_files if p.name in include]

    if args.max_files > 0:
        txt_files = txt_files[: args.max_files]

    resume = True
    if args.no_resume:
        resume = False
    elif args.resume:
        resume = True

    report_path = output_dir / "validation.json"
    validation = load_validation(report_path) if args.validate and resume else {}

    total_files = len(txt_files)
    progress_path = output_dir / "progress.json"
    start_time = time.time()
    start_iso = datetime.now().isoformat(timespec="seconds")
    progress = {
        "status": "running",
        "total_files": total_files,
        "done_files": 0,
        "skipped_files": 0,
        "error_files": 0,
        "start_time": start_iso,
        "last_file": "",
        "last_status": "",
        "last_update": start_iso,
        "elapsed_minutes": 0.0,
        "current_file": "",
        "current_chunk": 0,
        "current_chunk_total": 0,
        "current_file_progress": 0.0,
    }
    write_progress(progress_path, progress)

    done_count = 0
    skipped_count = 0
    error_count = 0

    try:
        for idx, src_path in enumerate(txt_files, start=1):
            out_path = output_dir / src_path.name
            if resume and out_path.exists() and out_path.stat().st_size > 0:
                skipped_count += 1
                progress["done_files"] = done_count
                progress["skipped_files"] = skipped_count
                progress["error_files"] = error_count
                progress["last_file"] = src_path.name
                progress["last_status"] = "skipped"
                progress["current_file"] = src_path.name
                progress["current_chunk"] = 0
                progress["current_chunk_total"] = 0
                progress["current_file_progress"] = 0.0
                progress["last_update"] = datetime.now().isoformat(timespec="seconds")
                progress["elapsed_minutes"] = round((time.time() - start_time) / 60.0, 2)
                write_progress(progress_path, progress)
                print(f"[{idx}/{total_files}] skipped (exists): {src_path.name}")
                continue

            src_text = src_path.read_text(encoding="utf-8", errors="ignore")
            chunks = chunk_by_paragraphs(src_text, args.chunk_words)
            cleaned_chunks = []
            file_errors = []

            progress["current_file"] = src_path.name
            progress["current_chunk"] = 0
            progress["current_chunk_total"] = len(chunks)
            progress["current_file_progress"] = 0.0
            progress["last_status"] = "processing"
            progress["last_update"] = datetime.now().isoformat(timespec="seconds")
            progress["elapsed_minutes"] = round((time.time() - start_time) / 60.0, 2)
            write_progress(progress_path, progress)

            for chunk_idx, chunk in enumerate(chunks):
                try:
                    cleaned = call_gemini(api_key, args.model, chunk)
                except Exception as exc:  # noqa: BLE001
                    if args.continue_on_error:
                        cleaned = chunk
                        file_errors.append({"chunk": chunk_idx, "error": str(exc)})
                    else:
                        raise
                cleaned_chunks.append(cleaned)
                progress["current_chunk"] = chunk_idx + 1
                progress["current_file_progress"] = round(
                    100.0 * (chunk_idx + 1) / max(1, len(chunks)),
                    2,
                )
                progress["last_update"] = datetime.now().isoformat(timespec="seconds")
                progress["elapsed_minutes"] = round((time.time() - start_time) / 60.0, 2)
                write_progress(progress_path, progress)
                time.sleep(args.sleep_sec)

            cleaned_text = "\n\n".join(c for c in cleaned_chunks if c.strip())

            temp_path = out_path.with_suffix(out_path.suffix + ".partial")
            temp_path.write_text(cleaned_text, encoding="utf-8")
            if out_path.exists():
                out_path.unlink()
            temp_path.replace(out_path)

            if args.validate:
                stats = validate_pair(src_text, cleaned_text)
                if file_errors:
                    stats["api_errors"] = file_errors
                validation[src_path.name] = stats
                write_validation(report_path, validation)

            if file_errors:
                error_count += 1
                print(f"[{idx}/{total_files}] cleaned with warnings: {src_path.name}")
                last_status = "cleaned_with_warnings"
            else:
                print(f"[{idx}/{total_files}] cleaned: {src_path.name}")
                last_status = "cleaned"

            done_count += 1
            progress["done_files"] = done_count
            progress["skipped_files"] = skipped_count
            progress["error_files"] = error_count
            progress["last_file"] = src_path.name
            progress["last_status"] = last_status
            progress["current_file"] = src_path.name
            progress["current_chunk"] = len(chunks)
            progress["current_chunk_total"] = len(chunks)
            progress["current_file_progress"] = 100.0
            progress["last_update"] = datetime.now().isoformat(timespec="seconds")
            progress["elapsed_minutes"] = round((time.time() - start_time) / 60.0, 2)
            write_progress(progress_path, progress)
    except Exception as exc:
        progress["status"] = "error"
        progress["last_error"] = str(exc)
        progress["last_update"] = datetime.now().isoformat(timespec="seconds")
        progress["elapsed_minutes"] = round((time.time() - start_time) / 60.0, 2)
        write_progress(progress_path, progress)
        raise
    else:
        progress["status"] = "completed"
        progress["last_update"] = datetime.now().isoformat(timespec="seconds")
        progress["elapsed_minutes"] = round((time.time() - start_time) / 60.0, 2)
        write_progress(progress_path, progress)

    if args.validate:
        print(f"Validation report: {report_path}")


if __name__ == "__main__":
    main()
