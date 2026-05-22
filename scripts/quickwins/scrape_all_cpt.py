"""Master CPT scraper orchestrator — runs all data collection tasks sequentially.

Order (to avoid NCBI rate-limit conflicts):
  1. Wikipedia dental (uses MediaWiki API, no NCBI conflict)
  2. OpenAlex dental (independent API)
  3. HF extra datasets (local download)
  4. PMC full-text (NCBI - runs after PubMed extension finishes)
  5. PubMed extension (NCBI - extend 1990-2021)

Each scraper is resumable, so interrupting and restarting is safe.
"""
from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PYTHON = str(ROOT / ".venv" / "Scripts" / "python.exe")


def run_script(module: str, args: list[str] | None = None, label: str = "") -> int:
    cmd = [PYTHON, "-u", "-m", module]
    if args:
        cmd.extend(args)
    print(f"\n{'='*70}")
    print(f"[orchestrator] STARTING: {label or module}")
    print(f"  cmd: {' '.join(cmd)}")
    print(f"{'='*70}\n")
    started = time.time()
    result = subprocess.run(cmd, cwd=str(ROOT))
    elapsed = time.time() - started
    status = "OK" if result.returncode == 0 else f"FAILED (rc={result.returncode})"
    print(f"\n[orchestrator] {label or module}: {status} ({elapsed:.0f}s)\n")
    return result.returncode


def main() -> int:
    print("[orchestrator] DocSmile CPT data collection pipeline")
    print(f"  root: {ROOT}")
    print(f"  python: {PYTHON}")
    print()

    tasks = [
        ("Wikipedia dental articles", "scripts.quickwins.wikipedia_dental", ["--max-articles", "5000"]),
        ("OpenAlex dental (200k target)", "scripts.quickwins.openalex_dental", ["--target", "200000", "--pub-year-min", "2000"]),
        ("HF extra datasets", "scripts.quickwins.hf_dental_extra", []),
        ("PMC full-text dental", "scripts.quickwins.pmc_fulltext", ["--target", "50000", "--max-articles", "25000"]),
        ("PubMed extension (1990-2021)", "scripts.quickwins.pubmed_dental", ["--target", "300000", "--start-year", "2021", "--end-year", "1990", "--out", "pubmed_dental_1990_2021.jsonl"]),
    ]

    results = {}
    for label, module, args in tasks:
        rc = run_script(module, args, label)
        results[label] = rc
        if rc != 0:
            print(f"[orchestrator] WARNING: {label} failed, continuing...")

    print("\n" + "="*70)
    print("[orchestrator] SUMMARY")
    print("="*70)
    for label, rc in results.items():
        status = "OK" if rc == 0 else "FAILED"
        print(f"  [{status}] {label}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
