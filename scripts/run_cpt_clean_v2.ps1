$ErrorActionPreference = "Stop"
Set-Location "C:\Users\HARI\Desktop\DocSmile"

python scripts\prepare_cpt_corpus_v2.py `
  --input-dirs core_cpt selective_cpt `
  --output-dir cpt_prepared\cpt_raw_text_v2 `
  --manifest-json cpt_prepared\cpt_raw_text_v2\manifest.json `
  --min-chars 2000

python scripts\clean_cpt_production_gemini.py `
  --input-dir cpt_prepared\cpt_raw_text_v2 `
  --output-dir cpt_prepared\cpt_clean_text_v2 `
  --parallel-files 4 `
  --concurrent-per-key 1 `
  --per-key-rpm 15 `
  --request-spacing-sec 0 `
  --model gemini-3.1-flash-lite `
  *>&1 | Tee-Object -FilePath cpt_prepared\cpt_clean_text_v2\clean_run.log
