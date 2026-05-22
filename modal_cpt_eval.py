from __future__ import annotations

import json
from pathlib import Path

import modal


APP_NAME = "docsmile-cpt-eval"
BASE_MODEL = "meta-llama/Llama-3.1-8B"
REMOTE_ROOT = "/root/docsmile"
REMOTE_RESULTS = "/root/docsmile_results"


image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "torch",
        "transformers>=4.49.0",
        "accelerate>=1.5.0",
        "peft>=0.14.0",
        "bitsandbytes>=0.45.0",
        "safetensors>=0.4.5",
        "sentencepiece>=0.2.0",
        "protobuf>=5.28.0",
        "tensorboard>=2.18.0",
        "huggingface_hub>=0.27.0",
    )
    .add_local_file(
        "scripts/run_text_eval.py",
        remote_path=f"{REMOTE_ROOT}/scripts/run_text_eval.py",
    )
    .add_local_file(
        "scripts/probe_cpt_model.py",
        remote_path=f"{REMOTE_ROOT}/scripts/probe_cpt_model.py",
    )
    .add_local_dir(
        "evals/text_only",
        remote_path=f"{REMOTE_ROOT}/evals/text_only",
    )
    .add_local_dir(
        "outputs/final",
        remote_path=f"{REMOTE_ROOT}/outputs/final",
    )
)

app = modal.App(APP_NAME)

def _copy_results_to_volume() -> None:
    # Kept simple: Modal will expose returned JSON, and local_entrypoint downloads
    # the result files through modal file IO by reading from function return payload.
    pass


@app.function(
    image=image,
    gpu="L40S",
    timeout=60 * 60 * 4,
    secrets=[modal.Secret.from_name("huggingface-secret")],
)
def run_eval(max_rows: int = 0) -> dict:
    import os
    import shutil
    import subprocess
    import sys

    remote_root = Path(REMOTE_ROOT)
    remote_results = Path(REMOTE_RESULTS)

    os.environ.setdefault("HF_HOME", "/root/hf_cache")
    os.environ.setdefault("HUGGINGFACE_HUB_CACHE", "/root/hf_cache/hub")
    os.environ.setdefault("TRANSFORMERS_CACHE", "/root/hf_cache/transformers")
    os.environ.setdefault("HF_TOKEN", os.environ.get("HUGGINGFACE_TOKEN", ""))

    if not os.environ.get("HF_TOKEN"):
        raise RuntimeError("Missing HF_TOKEN in Modal secret `huggingface-secret`.")

    remote_results.mkdir(parents=True, exist_ok=True)
    dental_out = remote_results / "llama31_8b_cpt_dental"
    probes_out = remote_results / "llama31_8b_cpt_hard_probes" / "probes.jsonl"

    dental_cmd = [
        sys.executable,
        str(remote_root / "scripts" / "run_text_eval.py"),
        "--eval-dir",
        str(remote_root / "evals" / "text_only"),
        "--output-dir",
        str(dental_out),
        "--model",
        BASE_MODEL,
        "--adapter-path",
        str(remote_root / "outputs" / "final"),
        "--backend",
        "local-hf",
        "--dtype",
        "bf16",
        "--load-in-4bit",
    ]
    if max_rows > 0:
        dental_cmd.extend(["--max-rows", str(max_rows)])

    subprocess.run(dental_cmd, check=True)

    probe_cmd = [
        sys.executable,
        str(remote_root / "scripts" / "probe_cpt_model.py"),
        "--model",
        BASE_MODEL,
        "--adapter-path",
        str(remote_root / "outputs" / "final"),
        "--output-jsonl",
        str(probes_out),
        "--load-in-4bit",
        "--dtype",
        "bf16",
    ]
    subprocess.run(probe_cmd, check=True)

    files = {}
    for path in sorted(remote_results.rglob("*")):
        if path.is_file():
            rel = path.relative_to(remote_results).as_posix()
            files[rel] = path.read_text(encoding="utf-8", errors="ignore")

    return {
        "status": "completed",
        "result_files": sorted(files),
        "files": files,
    }


@app.local_entrypoint()
def main(max_rows: int = 0, output_dir: str = "evals/results/modal_llama31_8b_cpt") -> None:
    result = run_eval.remote(max_rows=max_rows)
    out_root = Path(output_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    for rel, text in result["files"].items():
        path = out_root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")

    summary_path = out_root / "modal_eval_summary.json"
    summary_path.write_text(
        json.dumps({"status": result["status"], "result_files": result["result_files"]}, indent=2),
        encoding="utf-8",
    )
    print(f"Wrote results to {out_root}")
