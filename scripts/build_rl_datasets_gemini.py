import argparse
import html
import json
import os
import random
import re
import time
import urllib.error
import urllib.request
from datetime import datetime
from pathlib import Path


MD_IMAGE_RE = re.compile(r"!\[.*?\]\(.*?\)")
MD_LINK_RE = re.compile(r"\[(.*?)\]\((.*?)\)")
TABLE_ROW_RE = re.compile(r"^\s*\|.*\|\s*$")
TABLE_DIV_RE = re.compile(r"^\s*\|?\s*-{2,}(\s*\|\s*-{2,})+\s*\|?\s*$")
CODE_FENCE_RE = re.compile(r"^\s*```")
HTML_TAG_RE = re.compile(r"<[^>]+>")
API_KEY_NAME_RE = re.compile(r"^(?:GEMINI_API_KEY|GOOGLE_API_KEY)(?:\d+)?$")


def _natural_key(value: str) -> list[object]:
    return [int(part) if part.isdigit() else part.lower() for part in re.split(r"(\d+)", value)]


def load_api_keys(env_path: Path) -> list[tuple[str, str]]:
    candidates: list[tuple[str, str]] = []

    if env_path.exists():
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
            if API_KEY_NAME_RE.match(name) and value:
                candidates.append((name, value))

    for name, value in os.environ.items():
        if API_KEY_NAME_RE.match(name) and value:
            candidates.append((name, value))

    candidates.sort(key=lambda item: _natural_key(item[0]))

    seen_values = set()
    api_keys: list[tuple[str, str]] = []
    for name, value in candidates:
        if value in seen_values:
            continue
        seen_values.add(value)
        api_keys.append((name, value))
    return api_keys


def call_gemini_with_fallback(
    api_keys: list[tuple[str, str]],
    start_index: int,
    model: str,
    text: str,
    max_retries: int = 6,
    sleep_sec: float = 0.6,
    retry_max_sleep: float = 60.0,
) -> tuple[str, int]:
    if not api_keys:
        raise RuntimeError("Missing Gemini API keys")

    errors = []
    total = len(api_keys)
    for offset in range(total):
        index = (start_index + offset) % total
        key_name, api_key = api_keys[index]
        try:
            result = call_gemini(
                api_key,
                model,
                text,
                max_retries=1,
                sleep_sec=sleep_sec,
                retry_max_sleep=retry_max_sleep,
            )
            return result, index
        except urllib.error.HTTPError as exc:
            errors.append(f"{key_name}: HTTP {exc.code} {exc.reason}")
            if exc.code not in {400, 401, 403, 429, 500, 503}:
                raise
        except (urllib.error.URLError, RuntimeError) as exc:
            errors.append(f"{key_name}: {exc}")

    summary = " | ".join(errors[-total:])
    raise RuntimeError(f"All Gemini API keys failed. Recent errors: {summary}")


def _parse_retry_after(headers) -> float:
    if not headers:
        return 0.0
    try:
        value = headers.get("Retry-After")
    except AttributeError:
        value = None
    if not value:
        return 0.0
    try:
        return float(value)
    except ValueError:
        return 0.0


def call_gemini(
    api_key: str,
    model: str,
    text: str,
    max_retries: int = 6,
    sleep_sec: float = 0.6,
    retry_max_sleep: float = 60.0,
) -> str:
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={api_key}"
    payload = {
        "contents": [
            {
                "role": "user",
                "parts": [{"text": text}],
            }
        ],
        "generationConfig": {
            "temperature": 0.2,
            "topP": 0.9,
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

            if isinstance(exc, urllib.error.HTTPError) and exc.code in {429, 500, 503}:
                retry_after = _parse_retry_after(getattr(exc, "headers", None))
                backoff = max(retry_after, sleep_sec * (2 ** attempt))
            else:
                backoff = sleep_sec * (2 ** attempt)

            backoff = min(retry_max_sleep, backoff)
            jitter = random.uniform(0.0, min(5.0, 0.25 * backoff))
            time.sleep(backoff + jitter)

    raise RuntimeError("Gemini API call failed after retries")


def _is_table_line(line: str) -> bool:
    if TABLE_DIV_RE.match(line):
        return True
    if TABLE_ROW_RE.match(line) and line.count("|") >= 2:
        return True
    return False


def clean_markdown(text: str) -> str:
    lines = text.splitlines()
    cleaned_lines = []
    in_code_block = False

    for raw in lines:
        line = raw.rstrip("\n")

        if CODE_FENCE_RE.match(line):
            in_code_block = not in_code_block
            continue
        if in_code_block:
            continue

        stripped = line.strip()
        if stripped == "":
            cleaned_lines.append("")
            continue

        if stripped.startswith("<!--") and stripped.endswith("-->"):
            continue

        lowered = stripped.lower()
        if "terms and conditions" in lowered:
            continue
        if "all rights reserved" in lowered:
            continue
        if "copyright" in lowered:
            continue

        if _is_table_line(stripped):
            continue

        if re.match(r"^\s*(figure|fig\.|table)\b", stripped, re.IGNORECASE):
            continue

        line = MD_IMAGE_RE.sub("", line)
        line = MD_LINK_RE.sub(r"\1", line)
        line = HTML_TAG_RE.sub("", line)
        line = html.unescape(line)
        line = re.sub(r"https?://\S+|www\.\S+", "", line)

        heading_match = re.match(r"^\s*#{1,6}\s+(.*)$", line)
        if heading_match:
            heading_text = heading_match.group(1).strip()
            if heading_text:
                if cleaned_lines and cleaned_lines[-1] != "":
                    cleaned_lines.append("")
                cleaned_lines.append(heading_text)
                cleaned_lines.append("")
            continue

        bullet_match = re.match(r"^\s*[-*+]\s+\[[ xX]\]\s+(.*)$", line)
        if bullet_match:
            item = bullet_match.group(1).strip()
            if item:
                cleaned_lines.append(item)
                cleaned_lines.append("")
            continue

        bullet_match = re.match(r"^\s*[-*+]\s+(.*)$", line)
        if bullet_match:
            item = bullet_match.group(1).strip()
            if item:
                cleaned_lines.append(item)
                cleaned_lines.append("")
            continue

        line = re.sub(r"^\s*>\s+", "", line)
        line = re.sub(r"\*\*(.*?)\*\*", r"\1", line)
        line = re.sub(r"__(.*?)__", r"\1", line)
        line = re.sub(r"\*(.*?)\*", r"\1", line)
        line = re.sub(r"_(.*?)_", r"\1", line)

        line = line.replace("\t", " ")
        line = re.sub(r"\s+", " ", line).strip()

        if line.isdigit():
            continue

        if line:
            cleaned_lines.append(line)

    collapsed = []
    last_blank = False
    for line in cleaned_lines:
        is_blank = line == ""
        if is_blank and last_blank:
            continue
        collapsed.append(line)
        last_blank = is_blank

    paragraphs = []
    buf = []
    for line in collapsed:
        if line == "":
            if buf:
                paragraphs.append(" ".join(buf))
                buf = []
            continue
        buf.append(line)
    if buf:
        paragraphs.append(" ".join(buf))

    return "\n\n".join(paragraphs).strip()


def chunk_text(text: str, chunk_words: int, overlap_words: int) -> list[str]:
    words = text.split()
    if not words:
        return []

    chunks = []
    step = max(1, chunk_words - overlap_words)
    for start in range(0, len(words), step):
        end = start + chunk_words
        chunk = " ".join(words[start:end]).strip()
        if chunk:
            chunks.append(chunk)
        if end >= len(words):
            break
    return chunks


def normalize_text(text: str) -> str:
    lowered = text.lower()
    lowered = re.sub(r"[^a-z0-9]+", " ", lowered)
    return re.sub(r"\s+", " ", lowered).strip()


def strip_json_fence(text: str) -> str:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?", "", cleaned).strip()
    if cleaned.endswith("```"):
        cleaned = cleaned[: -3].strip()
    return cleaned


def parse_json_payload(text: str) -> dict:
    cleaned = strip_json_fence(text)
    try:
        payload = json.loads(cleaned)
        if isinstance(payload, dict):
            return payload
    except json.JSONDecodeError:
        pass

    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start != -1 and end != -1 and end > start:
        payload = json.loads(cleaned[start : end + 1])
        if isinstance(payload, dict):
            return payload

    raise ValueError("Could not parse JSON payload")


def payload_items(payload: dict, key: str) -> list[dict]:
    value = payload.get(key, [])
    if value is None:
        return []
    if isinstance(value, list):
        return [item for item in value if isinstance(item, dict)]
    if isinstance(value, dict):
        return [value]
    return []


def calc_targets(word_count: int, sft_min: int, sft_max: int, dpo_min: int, dpo_max: int) -> tuple[int, int]:
    sft_target = max(sft_min, min(sft_max, int(round(word_count / 2500))))
    dpo_target = max(dpo_min, min(dpo_max, int(round(word_count / 5000))))
    return sft_target, dpo_target


TASK_MIX_CYCLE = (
    ["clinical_student"] * 8
    + ["patient_friendly"] * 5
    + ["exam_reasoning"] * 4
    + ["differential_comparison"] * 2
    + ["safety_referral"] * 1
)


TASK_STYLE_GUIDES = {
    "clinical_student": (
        "Clinical/dental student Q&A. Ask a clinically useful dental-student level question. "
        "Answer with precise terminology, relevant mechanisms, and practical clinical meaning."
    ),
    "patient_friendly": (
        "Patient-friendly explanation. Ask a realistic patient-style question and answer in plain language. "
        "Avoid jargon where possible, explain uncertainty, and encourage dental evaluation when appropriate."
    ),
    "exam_reasoning": (
        "Exam-style reasoning. Ask a board/exam-style question and answer with a concise final answer plus "
        "brief rationale, key clues, and why the closest alternative is less suitable. Do not expose long chain-of-thought."
    ),
    "differential_comparison": (
        "Differential diagnosis/comparison. Ask a comparison or distinction question. "
        "Answer by contrasting features, clues, and decision points supported by the text."
    ),
    "safety_referral": (
        "Safety/referral guidance. Ask about risk, red flags, contraindications, urgent symptoms, or when to refer. "
        "Answer cautiously, avoid definitive diagnosis from limited information, and include appropriate escalation."
    ),
}


def task_style_for_chunk(chunk_idx: int) -> str:
    return TASK_MIX_CYCLE[chunk_idx % len(TASK_MIX_CYCLE)]


def build_prompt(chunk: str, want_sft: int, want_dpo: int, file_title: str, task_style: str) -> str:
    style_guide = TASK_STYLE_GUIDES.get(task_style, TASK_STYLE_GUIDES["clinical_student"])
    return (
        "You are creating high-quality dental-domain SFT and DPO training data.\n"
        "Use ONLY facts from the provided text. Do not add external knowledge.\n"
        "Prioritize quality over quantity. Avoid generic or duplicated questions.\n"
        "Ignore front matter (publisher info, copyright, dedications, TOC) unless clinically relevant.\n\n"
        "Target task style for this chunk:\n"
        f"{style_guide}\n\n"
        "Return STRICT JSON (no markdown) with keys: sft, dpo.\n"
        "Each sft item: {\"instruction\": str, \"output\": str, \"topic\": str}.\n"
        "Each dpo item: {\"prompt\": str, \"chosen\": str, \"rejected\": str, \"topic\": str}.\n"
        "Rules:\n"
        "- All items must be unique and grounded in the text.\n"
        "- Keep SFT outputs concise but complete (about 60-200 words).\n"
        "- For patient-friendly items, use clear everyday language and avoid overconfident diagnosis.\n"
        "- For exam reasoning items, include brief rationale/key clues, not hidden chain-of-thought.\n"
        "- For differential/comparison items, explicitly compare the relevant entities or conditions.\n"
        "- For safety/referral items, include red flags or referral/urgent-care guidance only when supported by the text.\n"
        "- DPO rejected should be plausible but wrong/incomplete; no dangerous advice.\n"
        "- DPO chosen should be more grounded, safer, and more clinically useful than rejected.\n"
        "- Do NOT repeat wording across items.\n\n"
        f"File title: {file_title}\n"
        f"Generate up to {want_sft} SFT items and up to {want_dpo} DPO items.\n"
        f"Use task style label: {task_style}.\n"
        "TEXT:\n"
        f"{chunk}"
    )


def load_existing_keys(path: Path, key_fn) -> set[str]:
    keys = set()
    if not path.exists():
        return keys
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        if not line.strip():
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue
        key = key_fn(obj)
        if key:
            keys.add(key)
    return keys


def write_progress(progress_path: Path, progress: dict) -> None:
    progress_path.write_text(json.dumps(progress, indent=2, ensure_ascii=True), encoding="utf-8")


def load_state(state_path: Path) -> dict:
    if not state_path.exists():
        return {"processed_files": [], "file_chunks": {}, "total_sft": 0, "total_dpo": 0}
    try:
        payload = json.loads(state_path.read_text(encoding="utf-8", errors="ignore"))
    except json.JSONDecodeError:
        return {"processed_files": [], "file_chunks": {}, "total_sft": 0, "total_dpo": 0}
    if isinstance(payload, dict):
        if "file_chunks" not in payload:
            payload["file_chunks"] = {}
        return payload
    return {"processed_files": [], "file_chunks": {}, "total_sft": 0, "total_dpo": 0}


def save_state(state_path: Path, state: dict) -> None:
    state["updated"] = datetime.now().isoformat(timespec="seconds")
    state_path.write_text(json.dumps(state, indent=2, ensure_ascii=True), encoding="utf-8")


def add_sft_item(
    item: dict,
    source: str,
    seen_keys: set[str],
    min_output_words: int,
    max_output_words: int,
) -> dict | None:
    instruction = str(item.get("instruction", "")).strip()
    output = str(item.get("output", "")).strip()
    topic = str(item.get("topic", "")).strip()

    if len(instruction) < 12 or len(output) < 40:
        return None

    output_words = output.split()
    if len(output_words) < min_output_words or len(output_words) > max_output_words:
        return None

    key = f"{normalize_text(instruction)}||{normalize_text(output)}"
    if not key or key in seen_keys:
        return None
    seen_keys.add(key)

    return {
        "question": instruction,
        "answer": output,
        "source": source,
        "topic": topic,
    }


def add_dpo_item(
    item: dict,
    source: str,
    seen_keys: set[str],
    min_chosen_words: int,
    max_chosen_words: int,
) -> dict | None:
    prompt = str(item.get("prompt", "")).strip()
    chosen = str(item.get("chosen", "")).strip()
    rejected = str(item.get("rejected", "")).strip()
    topic = str(item.get("topic", "")).strip()

    if len(prompt) < 12 or len(chosen) < 40 or len(rejected) < 20:
        return None

    if normalize_text(chosen) == normalize_text(rejected):
        return None

    chosen_words = chosen.split()
    if len(chosen_words) < min_chosen_words or len(chosen_words) > max_chosen_words:
        return None

    key = f"{normalize_text(prompt)}||{normalize_text(chosen)}"
    if not key or key in seen_keys:
        return None
    seen_keys.add(key)

    return {
        "prompt": prompt,
        "chosen": chosen,
        "rejected": rejected,
        "source": source,
        "topic": topic,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Create SFT and DPO datasets from RL markdown using Gemini.")
    parser.add_argument("--input-dir", required=True, help="Directory with RL .md files")
    parser.add_argument("--output-dir", required=True, help="Directory for dataset outputs")
    parser.add_argument("--model", default="gemini-3.1-flash-lite-preview", help="Gemini model name")
    parser.add_argument("--chunk-words", type=int, default=1200, help="Words per chunk")
    parser.add_argument("--overlap-words", type=int, default=120, help="Overlap words between chunks")
    parser.add_argument("--max-files", type=int, default=0, help="Limit number of files")
    parser.add_argument(
        "--max-completed-files",
        type=int,
        default=0,
        help="Stop before starting a new file once this many total files are completed in state.json",
    )
    parser.add_argument("--max-chunks", type=int, default=0, help="Limit chunks per file")
    parser.add_argument("--sft-min", type=int, default=2, help="Minimum SFT items per file")
    parser.add_argument("--sft-max", type=int, default=8, help="Maximum SFT items per file")
    parser.add_argument("--dpo-min", type=int, default=1, help="Minimum DPO items per file")
    parser.add_argument("--dpo-max", type=int, default=4, help="Maximum DPO items per file")
    parser.add_argument("--resume", action="store_true", help="Resume using state.json and existing outputs")
    parser.add_argument("--no-resume", action="store_true", help="Ignore existing outputs and state")
    parser.add_argument("--sleep-sec", type=float, default=0.6, help="Sleep between requests")
    parser.add_argument("--min-request-gap", type=float, default=1.2, help="Min seconds between API calls")
    parser.add_argument("--max-retries", type=int, default=6, help="Max retries for Gemini calls")
    parser.add_argument("--retry-max-sleep", type=float, default=60.0, help="Max backoff sleep on retry")
    parser.add_argument("--continue-on-error", action="store_true", help="Continue on API errors")
    parser.add_argument("--min-output-words", type=int, default=60, help="Min words for SFT output")
    parser.add_argument("--max-output-words", type=int, default=200, help="Max words for SFT output")
    parser.add_argument("--min-chosen-words", type=int, default=80, help="Min words for DPO chosen")
    parser.add_argument("--max-chosen-words", type=int, default=220, help="Max words for DPO chosen")
    args = parser.parse_args()

    root_dir = Path(__file__).resolve().parents[1]
    env_path = root_dir / ".env"
    api_keys = load_api_keys(env_path)
    if not api_keys:
        raise RuntimeError("Missing GEMINI_API_KEY/GOOGLE_API_KEY values in .env or environment")
    print(f"Loaded {len(api_keys)} Gemini API key(s): {', '.join(name for name, _ in api_keys)}", flush=True)

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    sft_path = output_dir / "rl_sft.jsonl"
    dpo_path = output_dir / "rl_dpo.jsonl"
    state_path = output_dir / "state.json"
    progress_path = output_dir / "progress.json"

    resume = True
    if args.no_resume:
        resume = False
    elif args.resume:
        resume = True

    state = load_state(state_path) if resume else {"processed_files": [], "file_chunks": {}, "total_sft": 0, "total_dpo": 0}
    processed = set(state.get("processed_files", []))
    file_chunks = state.get("file_chunks", {})

    seen_sft = load_existing_keys(sft_path, lambda obj: normalize_text(obj.get("question", ""))
                                  + "||" + normalize_text(obj.get("answer", "")))
    seen_dpo = load_existing_keys(dpo_path, lambda obj: normalize_text(obj.get("prompt", ""))
                                  + "||" + normalize_text(obj.get("chosen", "")))

    md_files = sorted([p for p in input_dir.glob("*.md") if p.is_file()])
    if args.max_files > 0:
        md_files = md_files[: args.max_files]

    total_files = len(md_files)
    start_time = time.time()
    start_iso = datetime.now().isoformat(timespec="seconds")
    progress = {
        "status": "running",
        "total_files": total_files,
        "done_files": 0,
        "skipped_files": 0,
        "error_files": 0,
        "completed_files_existing": len(processed),
        "completed_files_total": len(processed),
        "max_completed_files": args.max_completed_files,
        "total_sft": state.get("total_sft", 0),
        "total_dpo": state.get("total_dpo", 0),
        "start_time": start_iso,
        "last_file": "",
        "last_status": "",
        "last_update": start_iso,
        "elapsed_minutes": 0.0,
        "current_file": "",
        "current_chunk": 0,
        "current_chunk_total": 0,
        "current_file_progress": 0.0,
        "api_key_count": len(api_keys),
        "active_api_key": api_keys[0][0],
    }
    write_progress(progress_path, progress)

    done_count = 0
    skipped_count = 0
    error_count = 0
    last_request_time = 0.0
    api_key_index = 0

    with sft_path.open("a", encoding="utf-8") as sft_out, dpo_path.open("a", encoding="utf-8") as dpo_out:
        try:
            for idx, md_path in enumerate(md_files, start=1):
                if args.max_completed_files > 0 and len(processed) >= args.max_completed_files:
                    progress["status"] = "paused_limit"
                    progress["last_status"] = (
                        f"reached max completed files: {len(processed)}/{args.max_completed_files}"
                    )
                    progress["completed_files_total"] = len(processed)
                    progress["last_update"] = datetime.now().isoformat(timespec="seconds")
                    progress["elapsed_minutes"] = round((time.time() - start_time) / 60.0, 2)
                    write_progress(progress_path, progress)
                    print(
                        f"Reached max completed files ({len(processed)}/{args.max_completed_files}). "
                        "Stopping before the next book.",
                        flush=True,
                    )
                    break

                if md_path.name in processed:
                    skipped_count += 1
                    progress["skipped_files"] = skipped_count
                    progress["completed_files_total"] = len(processed)
                    progress["last_file"] = md_path.name
                    progress["last_status"] = "skipped"
                    progress["current_file"] = md_path.name
                    progress["current_chunk"] = 0
                    progress["current_chunk_total"] = 0
                    progress["current_file_progress"] = 0.0
                    progress["last_update"] = datetime.now().isoformat(timespec="seconds")
                    progress["elapsed_minutes"] = round((time.time() - start_time) / 60.0, 2)
                    write_progress(progress_path, progress)
                    print(f"[{idx}/{total_files}] skipped (state): {md_path.name}")
                    continue

                src_text = md_path.read_text(encoding="utf-8", errors="ignore")
                clean_text = clean_markdown(src_text)
                if not clean_text:
                    skipped_count += 1
                    progress["skipped_files"] = skipped_count
                    progress["completed_files_total"] = len(processed)
                    progress["last_file"] = md_path.name
                    progress["last_status"] = "empty"
                    progress["last_update"] = datetime.now().isoformat(timespec="seconds")
                    progress["elapsed_minutes"] = round((time.time() - start_time) / 60.0, 2)
                    write_progress(progress_path, progress)
                    print(f"[{idx}/{total_files}] skipped (empty): {md_path.name}")
                    continue

                words = clean_text.split()
                sft_target, dpo_target = calc_targets(
                    len(words),
                    args.sft_min,
                    args.sft_max,
                    args.dpo_min,
                    args.dpo_max,
                )

                chunks = chunk_text(clean_text, args.chunk_words, args.overlap_words)
                if args.max_chunks > 0:
                    chunks = chunks[: args.max_chunks]

                progress["current_file"] = md_path.name
                progress["current_chunk"] = 0
                progress["current_chunk_total"] = len(chunks)
                progress["current_file_progress"] = 0.0
                progress["last_status"] = "processing"
                progress["last_update"] = datetime.now().isoformat(timespec="seconds")
                progress["elapsed_minutes"] = round((time.time() - start_time) / 60.0, 2)
                write_progress(progress_path, progress)

                file_sft = 0
                file_dpo = 0
                start_chunk_idx = file_chunks.get(md_path.name, 0)

                for chunk_idx, chunk in enumerate(chunks):
                    if chunk_idx < start_chunk_idx:
                        continue
                    
                    want_sft = 1
                    want_dpo = 1
                    task_style = task_style_for_chunk(chunk_idx)

                    prompt = build_prompt(chunk, want_sft, want_dpo, md_path.stem, task_style)

                    try:
                        if args.min_request_gap > 0:
                            wait_for = args.min_request_gap - (time.time() - last_request_time)
                            if wait_for > 0:
                                time.sleep(wait_for)
                        raw, api_key_index = call_gemini_with_fallback(
                            api_keys,
                            api_key_index,
                            args.model,
                            prompt,
                            max_retries=args.max_retries,
                            sleep_sec=args.sleep_sec,
                            retry_max_sleep=args.retry_max_sleep,
                        )
                        progress["active_api_key"] = api_keys[api_key_index][0]
                        api_key_index = (api_key_index + 1) % len(api_keys)
                        last_request_time = time.time()
                        payload = parse_json_payload(raw)
                    except Exception as exc:  # noqa: BLE001
                        error_count += 1
                        progress["error_files"] = error_count
                        progress["completed_files_total"] = len(processed)
                        progress["last_file"] = md_path.name
                        progress["last_status"] = f"error: {exc}"
                        progress["last_update"] = datetime.now().isoformat(timespec="seconds")
                        progress["elapsed_minutes"] = round((time.time() - start_time) / 60.0, 2)
                        write_progress(progress_path, progress)
                        if args.continue_on_error:
                            continue
                        raise

                    for item in payload_items(payload, "sft")[:want_sft]:
                        record = add_sft_item(
                            item,
                            md_path.name,
                            seen_sft,
                            args.min_output_words,
                            args.max_output_words,
                        )
                        if record:
                            sft_out.write(json.dumps(record, ensure_ascii=True) + "\n")
                            file_sft += 1
                            progress["total_sft"] += 1

                    for item in payload_items(payload, "dpo")[:want_dpo]:
                        record = add_dpo_item(
                            item,
                            md_path.name,
                            seen_dpo,
                            args.min_chosen_words,
                            args.max_chosen_words,
                        )
                        if record:
                            dpo_out.write(json.dumps(record, ensure_ascii=True) + "\n")
                            file_dpo += 1
                            progress["total_dpo"] += 1

                    progress["current_chunk"] = chunk_idx + 1
                    progress["current_file_progress"] = round(
                        100.0 * (chunk_idx + 1) / max(1, len(chunks)),
                        2,
                    )
                    progress["last_update"] = datetime.now().isoformat(timespec="seconds")
                    progress["elapsed_minutes"] = round((time.time() - start_time) / 60.0, 2)
                    write_progress(progress_path, progress)
                    
                    file_chunks[md_path.name] = chunk_idx + 1
                    state["file_chunks"] = file_chunks
                    save_state(state_path, state)
                    
                    time.sleep(args.sleep_sec)

                done_count += 1
                progress["done_files"] = done_count
                progress["skipped_files"] = skipped_count
                progress["error_files"] = error_count
                progress["last_file"] = md_path.name
                progress["last_status"] = "completed"
                progress["current_file"] = md_path.name
                progress["current_chunk"] = len(chunks)
                progress["current_chunk_total"] = len(chunks)
                progress["current_file_progress"] = 100.0
                progress["last_update"] = datetime.now().isoformat(timespec="seconds")
                progress["elapsed_minutes"] = round((time.time() - start_time) / 60.0, 2)
                write_progress(progress_path, progress)

                processed.add(md_path.name)
                state["processed_files"] = sorted(processed)
                state["total_sft"] = progress["total_sft"]
                state["total_dpo"] = progress["total_dpo"]
                file_chunks.pop(md_path.name, None)
                state["file_chunks"] = file_chunks
                save_state(state_path, state)
                progress["completed_files_total"] = len(processed)
                write_progress(progress_path, progress)

                print(
                    f"[{idx}/{total_files}] done: {md_path.name} "
                    f"(SFT {file_sft}/{sft_target}, DPO {file_dpo}/{dpo_target})"
                )
        except Exception as exc:
            progress["status"] = "error"
            progress["last_status"] = f"error: {exc}"
            progress["last_update"] = datetime.now().isoformat(timespec="seconds")
            progress["elapsed_minutes"] = round((time.time() - start_time) / 60.0, 2)
            write_progress(progress_path, progress)
            raise
        else:
            progress["status"] = "completed"
            progress["last_update"] = datetime.now().isoformat(timespec="seconds")
            progress["elapsed_minutes"] = round((time.time() - start_time) / 60.0, 2)
            write_progress(progress_path, progress)


if __name__ == "__main__":
    main()
