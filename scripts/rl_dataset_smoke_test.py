import subprocess
import sys
from pathlib import Path


def main() -> None:
    root_dir = Path(__file__).resolve().parents[1]
    input_dir = root_dir / "rl"
    output_dir = root_dir / "rl_prepared_sample"
    script_path = root_dir / "scripts" / "build_rl_datasets_gemini.py"

    cmd = [
        sys.executable,
        str(script_path),
        "--input-dir",
        str(input_dir),
        "--output-dir",
        str(output_dir),
        "--max-files",
        "1",
        "--max-chunks",
        "1",
        "--sft-min",
        "1",
        "--sft-max",
        "2",
        "--dpo-min",
        "1",
        "--dpo-max",
        "1",
        "--continue-on-error",
    ]

    result = subprocess.run(cmd, check=False)
    raise SystemExit(result.returncode)


if __name__ == "__main__":
    main()
