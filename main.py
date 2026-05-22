from __future__ import annotations

import json
import os
import sys
import threading
import time
import traceback
from contextlib import contextmanager
from datetime import datetime, timezone
from multiprocessing import get_context
from pathlib import Path

# -------------------------
# CONFIG
# -------------------------

BASE_DIR = Path(__file__).resolve().parent
DEFAULT_BOOKS_DIR = BASE_DIR / "batch_2_l-20260408T052104Z-3-002" / "batch_2_l"

BOOKS_INPUT_DIR = Path(
    os.environ.get("BOOKS_DIR") or os.environ.get("DENTAL_BOOKS_DIR") or DEFAULT_BOOKS_DIR
)

OUTPUT_DIR = Path(os.environ.get("OUTPUT_DIR", "docling_output_md"))
OUTPUT_IMAGES_DIR = Path(os.environ.get("OUTPUT_IMAGES_DIR", "docling_output_images"))

SAVE_PICTURE_IMAGES = os.environ.get("SAVE_PICTURE_IMAGES", "1").strip() in ("1", "true", "yes")
RESUME = os.environ.get("RESUME", "1").strip() in ("1", "true", "yes")

IMAGE_PLACEHOLDER = os.environ.get("IMAGE_PLACEHOLDER", "<!--image-->")

def _max_books_from_env() -> int:
    try:
        return max(0, int(os.environ.get("MAX_BOOKS", "0")))
    except:
        return 0

MAX_BOOKS = _max_books_from_env()

def _convert_heartbeat_seconds() -> float:
    try:
        return max(0.0, float(os.environ.get("CONVERT_HEARTBEAT_SEC", "30")))
    except:
        return 30.0

# -------------------------
# Resume Helpers
# -------------------------

def _safe_stem(stem: str) -> str:
    return "".join(c if c.isalnum() or c in " -_." else "_" for c in stem)[:180]

def _output_md_path(out_root: Path, pdf_path: Path) -> Path:
    return out_root / f"{_safe_stem(pdf_path.stem)}.md"

def _is_complete_output(out_md: Path) -> bool:
    try:
        return out_md.is_file() and out_md.stat().st_size > 0
    except:
        return False

# -------------------------
# Heartbeat Thread
# -------------------------

@contextmanager
def _heartbeat_while_converting(gpu_id: int, pdf_name: str, interval_sec: float):
    if interval_sec <= 0:
        yield
        return

    stop = threading.Event()
    t0 = time.perf_counter()

    def run():
        while not stop.wait(interval_sec):
            elapsed = time.perf_counter() - t0
            print(
                f"[GPU {gpu_id}] ... still converting {pdf_name} ({elapsed:.0f}s elapsed)",
                flush=True,
            )

    th = threading.Thread(target=run, daemon=True)
    th.start()

    try:
        yield
    finally:
        stop.set()
        th.join()

# -------------------------
# Atomic Write
# -------------------------

def _atomic_write_text(path: Path, text: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(path)

# -------------------------
# Worker
# -------------------------

def _worker_gpu(gpu_id: int, pdf_paths: list[str], out_root: str, images_root: str):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    from docling.document_converter import DocumentConverter

    converter = DocumentConverter()

    hb_sec = _convert_heartbeat_seconds()

    out_root_p = Path(out_root)
    images_root_p = Path(images_root)

    rows = []

    print(f"[GPU {gpu_id}] Worker started | jobs={len(pdf_paths)}", flush=True)

    for i, pdf_str in enumerate(pdf_paths, 1):
        pdf_path = Path(pdf_str)
        safe = _safe_stem(pdf_path.stem)
        out_md = out_root_p / f"{safe}.md"

        row = {
            "gpu": gpu_id,
            "pdf": str(pdf_path),
            "output": str(out_md),
            "ok": False,
        }

        try:
            print(f"[GPU {gpu_id}] [{i}/{len(pdf_paths)}] START {pdf_path.name}", flush=True)

            t0 = time.perf_counter()

            with _heartbeat_while_converting(gpu_id, pdf_path.name, hb_sec):
                result = converter.convert(str(pdf_path))

            convert_time = time.perf_counter() - t0

            doc = result.document

            print(f"[GPU {gpu_id}] convert done in {convert_time:.2f}s", flush=True)

            md = doc.export_to_markdown()

            _atomic_write_text(out_md, md)

            row["ok"] = True
            row["chars"] = len(md)
            row["convert_time"] = round(convert_time, 2)

            print(f"[GPU {gpu_id}] DONE {pdf_path.name}", flush=True)

        except Exception as e:
            row["error"] = str(e)
            print(f"[GPU {gpu_id}] FAIL {pdf_path.name}: {e}", flush=True)

        rows.append(row)

    return rows

# -------------------------
# Split
# -------------------------

def _split_evenly(items: list, n: int):
    out = [[] for _ in range(n)]
    for i, x in enumerate(items):
        out[i % n].append(x)
    return out

# -------------------------
# MAIN
# -------------------------

def main():
    root = BASE_DIR

    books_dir = BOOKS_INPUT_DIR.expanduser().resolve()
    out_dir = (root / OUTPUT_DIR).resolve()
    images_dir = (root / OUTPUT_IMAGES_DIR).resolve()

    if not books_dir.exists():
        print("Books folder not found")
        return 1

    pdfs = sorted(books_dir.glob("*.pdf"))

    if not pdfs:
        print("No PDFs found")
        return 1

    out_dir.mkdir(parents=True, exist_ok=True)

    # -------------------------
    # Resume filtering
    # -------------------------
    pending = []
    skipped = 0

    for pdf in pdfs:
        out_md = _output_md_path(out_dir, pdf)

        if RESUME and _is_complete_output(out_md):
            skipped += 1
            continue

        pending.append(pdf)

    print(f"Skipped {skipped} books | Remaining {len(pending)}")

    if not pending:
        print("All books already processed")
        return 0

    if MAX_BOOKS > 0:
        pending = pending[:MAX_BOOKS]

    paths = [str(p) for p in pending]

    # -------------------------
    # GPU Setup
    # -------------------------
    try:
        import torch
        num_gpus = torch.cuda.device_count()
    except:
        num_gpus = 0

    if num_gpus <= 0:
        chunks = [paths]
        gpu_ids = [0]
        print("Running on CPU")
    else:
        n = min(num_gpus, len(paths))
        chunks = _split_evenly(paths, n)
        gpu_ids = list(range(n))
        print(f"Using {n} GPUs")

    ctx = get_context("spawn")
    all_rows = []

    jobs = [
        (gpu_ids[i], chunks[i], str(out_dir), str(images_dir))
        for i in range(len(chunks)) if chunks[i]
    ]

    if len(jobs) == 1:
        all_rows.extend(_worker_gpu(*jobs[0]))
    else:
        with ctx.Pool(len(jobs)) as pool:
            for rows in pool.starmap(_worker_gpu, jobs):
                all_rows.extend(rows)

    # -------------------------
    # Save manifest
    # -------------------------
    manifest = out_dir / "manifest.jsonl"

    with manifest.open("w", encoding="utf-8") as f:
        for row in all_rows:
            f.write(json.dumps(row) + "\n")

    ok = sum(1 for r in all_rows if r.get("ok"))

    print(f"Done {ok}/{len(all_rows)}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())