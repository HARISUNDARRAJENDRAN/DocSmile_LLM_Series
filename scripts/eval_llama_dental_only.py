"""
Run Llama 3.1-8B baseline evaluation on dental-specific benchmarks only.
This establishes the pre-CPT baseline for dental domain performance.
"""

import subprocess
import sys
from pathlib import Path
from datetime import datetime

MODEL_NAME = "meta-llama/Llama-3.1-8B"

def main():
    start_time = datetime.now()

    # Get project root (parent of scripts dir if running from scripts/)
    script_dir = Path(__file__).parent
    project_root = script_dir.parent if script_dir.name == "scripts" else Path.cwd()

    eval_dir = project_root / "evals" / "text_only"
    output_dir = project_root / "evals" / "results" / "llama_3_1_8b_dental_baseline"
    run_eval_script = project_root / "scripts" / "run_text_eval.py"

    print("\n" + "="*80)
    print("Llama 3.1-8B Dental Domain Baseline Evaluation")
    print("="*80)
    print(f"Model: {MODEL_NAME}")
    print(f"Evaluation suite: {eval_dir}")
    print(f"Output: {output_dir}")
    print(f"Start time: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*80 + "\n")

    print("Running dental domain evaluation...")
    print("This will evaluate on:")
    print("  - MedMCQA Dental MCQ (250 questions)")
    print("  - Oral Disease Open QA (250 questions)")
    print("  - Dental Forum Open QA (500 questions)")
    print()

    cmd = [
        sys.executable,
        str(run_eval_script),
        "--eval-dir", str(eval_dir),
        "--output-dir", str(output_dir),
        "--model", MODEL_NAME,
        "--backend", "local-hf",
        "--dtype", "auto"
    ]

    print(f"Command: {' '.join(cmd)}\n")

    try:
        subprocess.run(cmd, check=True)

        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds() / 60

        print("\n" + "="*80)
        print("EVALUATION COMPLETE")
        print("="*80)
        print(f"Duration: {duration:.1f} minutes")
        print(f"Results saved to: {output_dir}")
        print("\nNext steps:")
        print("  1. Review summary.json for MCQ accuracy")
        print("  2. Check prediction files for open QA quality")
        print("  3. Compare against Qwen 2.5-7B baseline")
        print("  4. Proceed with CPT if baseline is acceptable")
        print("="*80 + "\n")

    except subprocess.CalledProcessError as e:
        print(f"\n[ERROR] Evaluation failed: {e}")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\n\n[INTERRUPTED] Evaluation cancelled by user")
        sys.exit(1)

if __name__ == "__main__":
    main()
