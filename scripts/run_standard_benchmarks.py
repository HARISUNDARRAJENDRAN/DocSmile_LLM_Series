"""
Standard benchmark evaluation for base models before CPT.
Runs MMLU, HellaSwag, ARC, TruthfulQA, and GSM8K using lm-evaluation-harness.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from datetime import datetime


STANDARD_BENCHMARKS = {
    "mmlu": {
        "tasks": "mmlu",
        "num_fewshot": 5,
        "description": "Massive Multitask Language Understanding (57 subjects)"
    },
    "hellaswag": {
        "tasks": "hellaswag",
        "num_fewshot": 10,
        "description": "Commonsense reasoning about physical situations"
    },
    "arc_challenge": {
        "tasks": "arc_challenge",
        "num_fewshot": 25,
        "description": "AI2 Reasoning Challenge (hard subset)"
    },
    "truthfulqa": {
        "tasks": "truthfulqa_mc2",
        "num_fewshot": 0,
        "description": "Truthfulness in answering questions"
    },
    "gsm8k": {
        "tasks": "gsm8k",
        "num_fewshot": 5,
        "description": "Grade school math word problems"
    },
    "winogrande": {
        "tasks": "winogrande",
        "num_fewshot": 5,
        "description": "Commonsense reasoning (pronoun resolution)"
    }
}


def check_lm_eval_installed():
    """Check if lm-evaluation-harness is installed."""
    try:
        result = subprocess.run(
            ["lm_eval", "--help"],
            capture_output=True,
            text=True,
            timeout=10
        )
        return result.returncode == 0
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return False


def install_lm_eval():
    """Install lm-evaluation-harness."""
    print("Installing lm-evaluation-harness...")
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "-q", "lm-eval[api]>=0.4.0"],
        check=True
    )
    print("Installation complete.")


def run_benchmark(
    model_name: str,
    benchmark_name: str,
    output_dir: Path,
    batch_size: int = 8,
    device: str = "cuda",
    dtype: str = "auto",
    limit: int | None = None
):
    """Run a single benchmark using lm-evaluation-harness."""

    benchmark_config = STANDARD_BENCHMARKS[benchmark_name]

    print(f"\n{'='*80}")
    print(f"Running: {benchmark_name}")
    print(f"Description: {benchmark_config['description']}")
    print(f"Few-shot: {benchmark_config['num_fewshot']}")
    print(f"{'='*80}\n")

    output_path = output_dir / f"{benchmark_name}_results.json"

    cmd = [
        "lm_eval",
        "--model", "hf",
        "--model_args", f"pretrained={model_name},dtype={dtype},device_map=auto",
        "--tasks", benchmark_config["tasks"],
        "--num_fewshot", str(benchmark_config["num_fewshot"]),
        "--batch_size", str(batch_size),
        "--output_path", str(output_path),
        "--log_samples"
    ]

    if limit:
        cmd.extend(["--limit", str(limit)])

    print(f"Command: {' '.join(cmd)}\n")

    try:
        result = subprocess.run(cmd, check=True, text=True)
        print(f"\n✓ {benchmark_name} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n✗ {benchmark_name} failed with error: {e}")
        return False


def run_all_benchmarks(
    model_name: str,
    output_dir: Path,
    benchmarks: list[str] | None = None,
    batch_size: int = 8,
    limit: int | None = None
):
    """Run all specified benchmarks."""

    output_dir.mkdir(parents=True, exist_ok=True)

    if benchmarks is None:
        benchmarks = list(STANDARD_BENCHMARKS.keys())

    results = {}
    start_time = datetime.now()

    for benchmark in benchmarks:
        if benchmark not in STANDARD_BENCHMARKS:
            print(f"Warning: Unknown benchmark '{benchmark}', skipping...")
            continue

        success = run_benchmark(
            model_name=model_name,
            benchmark_name=benchmark,
            output_dir=output_dir,
            batch_size=batch_size,
            limit=limit
        )
        results[benchmark] = "success" if success else "failed"

    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()

    # Save summary
    summary = {
        "model": model_name,
        "timestamp": start_time.isoformat(),
        "duration_seconds": duration,
        "benchmarks": results,
        "output_dir": str(output_dir)
    }

    summary_path = output_dir / "benchmark_summary.json"
    with summary_path.open("w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n{'='*80}")
    print("BENCHMARK SUMMARY")
    print(f"{'='*80}")
    print(f"Model: {model_name}")
    print(f"Duration: {duration/60:.1f} minutes")
    print(f"\nResults:")
    for bench, status in results.items():
        print(f"  {bench}: {status}")
    print(f"\nFull results saved to: {output_dir}")
    print(f"{'='*80}\n")


def main():
    parser = argparse.ArgumentParser(
        description="Run standard benchmarks on base models before CPT"
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="HuggingFace model ID (e.g., meta-llama/Llama-3.1-8B)"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory to save benchmark results"
    )
    parser.add_argument(
        "--benchmarks",
        nargs="+",
        choices=list(STANDARD_BENCHMARKS.keys()) + ["all"],
        default=["all"],
        help="Which benchmarks to run (default: all)"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Batch size for evaluation (default: 8)"
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of samples per task (for testing)"
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="auto",
        choices=["auto", "fp16", "bf16", "fp32"],
        help="Model dtype (default: auto)"
    )

    args = parser.parse_args()

    # Check and install lm-eval if needed
    if not check_lm_eval_installed():
        print("lm-evaluation-harness not found.")
        install_lm_eval()

    # Determine which benchmarks to run
    if "all" in args.benchmarks:
        benchmarks = list(STANDARD_BENCHMARKS.keys())
    else:
        benchmarks = args.benchmarks

    print(f"\nStarting benchmark evaluation for: {args.model}")
    print(f"Benchmarks: {', '.join(benchmarks)}")
    print(f"Output directory: {args.output_dir}\n")

    run_all_benchmarks(
        model_name=args.model,
        output_dir=args.output_dir,
        benchmarks=benchmarks,
        batch_size=args.batch_size,
        limit=args.limit
    )


if __name__ == "__main__":
    main()
