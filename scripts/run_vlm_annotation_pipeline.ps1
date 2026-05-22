param(
  [int]$MaxImages = 0,
  [int]$JudgeSampleSize = 100,
  [string]$Model = "gemini-3.1-flash-lite-preview"
)

$ErrorActionPreference = "Stop"
Set-Location (Split-Path -Parent $PSScriptRoot)
New-Item -ItemType Directory -Force -Path "vlm_prepared" | Out-Null

$annotateArgs = @(
  "scripts\build_textbook_vlm_dataset_gemini.py",
  "--image-root", "IMAGES\textbooks",
  "--book-dirs", "core_cpt", "rl",
  "--audit-jsonl", "vlm_prepared\textbook_vlm_audit.jsonl",
  "--sft-jsonl", "vlm_prepared\textbook_vlm_sft.jsonl",
  "--errors-jsonl", "vlm_prepared\textbook_vlm_errors.jsonl",
  "--progress-json", "vlm_prepared\textbook_vlm_progress.json",
  "--model", $Model,
  "--continue-on-error"
)
if ($MaxImages -gt 0) {
  $annotateArgs += @("--max-images", "$MaxImages")
}

python @annotateArgs *>&1 | Tee-Object -FilePath "vlm_prepared\annotation_run.log"

python scripts\validate_vlm_sft_dataset.py `
  --input-jsonl vlm_prepared\textbook_vlm_sft.jsonl `
  --report-json vlm_prepared\textbook_vlm_sft_report.json `
  *>&1 | Tee-Object -FilePath "vlm_prepared\validation_run.log"

python scripts\judge_vlm_annotations_gemini.py `
  --input-jsonl vlm_prepared\textbook_vlm_audit.jsonl `
  --output-jsonl vlm_prepared\textbook_vlm_judgments.jsonl `
  --report-json vlm_prepared\textbook_vlm_judge_report.json `
  --model $Model `
  --sample-size $JudgeSampleSize `
  *>&1 | Tee-Object -FilePath "vlm_prepared\judge_run.log"
