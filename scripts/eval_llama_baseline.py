"""
Quick runner for Llama 3.1-8B baseline evaluation before CPT.
This will run standard benchmarks and your custom dental eval.
"""

import subprocess
import sys
from pathlib import Path
from datetime import datetime

MODEL_NAME = "meta-llama/Llama-3.1-8B"
OUTPUT_BASE = Path("evals/results")

def run_standard_benchmarks():
    """Run standard benchmarks (MMLU, HellaSwag, ARC, etc.)"""
    print("\n" + "="*80)
    print("PHASE 1: Standard Benchmarks")
    print("="*80 + "\n")

    output_dir = OUTPUT_BASE / "llama_3_1_8b_standard_benchmarks"

    cmd = [
        sys.executable,
        "scripts/run_standard_benchmarks.py",
        "--model", MODEL_NAME,
        "--output-dir", str(output_dir),
        "--benchmarks", "all",
        "--batch-size", "8"
    ]

    print(f"Running: {' '.join(cmd)}\n")
    subprocess.run(cmd, check=True)

    return output_dir

def run_dental_eval():
    """Run your custom dental domain evaluation"""
    print("\n" + "="*80)
    print("PHASE 2: Dental Domain Evaluation")
    print("="*80 + "\n")

    output_dir = OUTPUT_BASE / "llama_3_1_8b_dental_eval"

    cmd = [
        sys.executable,
        "scripts/run_text_eval.py",
        "--eval-dir", "evals/text_only",
        "--output-dir", str(output_dir),
        "--model", MODEL_NAME,
        "--backend", "local-hf",
        "--dtype", "auto"
    ]

    print(f"Running: {' '.join(cmd)}\n")
    subprocess.run(cmd, check=True)

    return output_dir

def main():
    start_time = datetime.now()

    print("\n" + "="*80)
    print("Llama 3.1-8B Baseline Evaluation")
    print("="*80)
    print(f"Model: {MODEL_NAME}")
    print(f"Start time: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*80 + "\n")

    try:
        # Run standard benchmarks
        standard_dir = run_standard_benchmarks()

        # Run dental domain eval
        dental_dir = run_dental_eval()

        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds() / 60

        print("\n" + "="*80)
        print("EVALUATION COMPLETE")
        print("="*80)
        print(f"Duration: {duration:.1f} minutes")
        print(f"\nResults saved to:")
        print(f"  Standard benchmarks: {standard_dir}")
        print(f"  Dental evaluation: {dental_dir}")
        print("="*80 + "\n")

    except subprocess.CalledProcessError as e:
        print(f"\n[ERROR] Evaluation failed: {e}")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\n\n[INTERRUPTED] Evaluation cancelled by user")
        sys.exit(1)

if __name__ == "__main__":
    main()
