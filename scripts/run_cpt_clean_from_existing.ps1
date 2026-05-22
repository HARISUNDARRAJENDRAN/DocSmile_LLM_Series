$ErrorActionPreference = "Continue"
if (Get-Variable PSNativeCommandUseErrorActionPreference -ErrorAction SilentlyContinue) {
  $PSNativeCommandUseErrorActionPreference = $false
}
Set-Location "C:\Users\HARI\Desktop\DocSmile"

$out = "cpt_prepared\core_cpt_text_gemini_clean_v3"
New-Item -ItemType Directory -Force $out | Out-Null

python scripts\clean_cpt_production_gemini.py `
  --input-dir cpt_prepared\core_cpt_text_cleaned `
  --output-dir $out `
  --parallel-files 4 `
  --concurrent-per-key 1 `
  --per-key-rpm 12 `
  --request-spacing-sec 0 `
  --model gemini-3.1-flash-lite `
  *>&1 | Tee-Object -FilePath "$out\clean_run.log"
