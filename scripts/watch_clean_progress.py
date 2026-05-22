import argparse
import json
import time
from datetime import datetime
from pathlib import Path


def resolve_progress_path(output_dir: str | None, progress_path: str | None) -> Path:
    if progress_path:
        return Path(progress_path)

    root_dir = Path(__file__).resolve().parents[1]
    if output_dir:
        return Path(output_dir) / "progress.json"

    return root_dir / "cpt_prepared" / "core_cpt_text_cleaned" / "progress.json"


def load_progress(path: Path) -> dict | None:
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8", errors="ignore"))
    except json.JSONDecodeError:
        return {"status": "invalid_json"}
    return payload if isinstance(payload, dict) else {"status": "invalid_json"}


def format_status(progress: dict | None) -> str:
    now = datetime.now().isoformat(timespec="seconds")
    if progress is None:
        return f"[{now}] waiting for progress.json"

    status = progress.get("status", "unknown")
    total = progress.get("total_files", "?")
    done = progress.get("done_files", 0)
    skipped = progress.get("skipped_files", 0)
    errors = progress.get("error_files", 0)
    last_file = progress.get("last_file", "")
    last_status = progress.get("last_status", "")
    last_update = progress.get("last_update", "")
    current_file = progress.get("current_file", "")
    current_chunk = progress.get("current_chunk", 0)
    current_chunk_total = progress.get("current_chunk_total", 0)
    current_progress = progress.get("current_file_progress", 0.0)

    age_note = ""
    if last_update:
        try:
            last_dt = datetime.fromisoformat(last_update)
            minutes = (datetime.now() - last_dt).total_seconds() / 60.0
            age_note = f"; {minutes:.1f} min since last update"
        except ValueError:
            age_note = ""

    current_note = ""
    if current_file and current_chunk_total:
        current_note = (
            f"; current={current_file} {current_chunk}/{current_chunk_total} "
            f"({current_progress:.2f}%)"
        )

    return (
        f"[{now}] status={status}; done={done}/{total}; skipped={skipped}; errors={errors}; "
        f"last={last_file} ({last_status}) at {last_update}{age_note}{current_note}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Watch progress.json for Gemini cleaning.")
    parser.add_argument("--output-dir", default="", help="Output dir containing progress.json")
    parser.add_argument("--progress-path", default="", help="Explicit progress.json path")
    parser.add_argument("--interval", type=int, default=60, help="Seconds between checks")
    parser.add_argument("--once", action="store_true", help="Print one snapshot and exit")
    args = parser.parse_args()

    progress_path = resolve_progress_path(
        output_dir=args.output_dir or None,
        progress_path=args.progress_path or None,
    )

    while True:
        progress = load_progress(progress_path)
        print(format_status(progress), flush=True)
        if args.once:
            break
        time.sleep(max(1, args.interval))


if __name__ == "__main__":
    main()
