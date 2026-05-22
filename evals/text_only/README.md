# DocSmile Text-Only Eval Pack

Frozen pre-CPT dental evaluation set. Do not include these rows in CPT, SFT, or DPO training.

## Files

- `medmcqa_dental_mcq.jsonl`: 250 dental multiple-choice questions with automatic accuracy.
- `oral_disease_open_qa.jsonl`: 250 oral-disease open QA cases for manual or judge-model grading.
- `dental_forum_open_qa.jsonl`: 500 patient-style dental forum QA cases for open-response review.
- `manifest.json`: counts and build notes.

## Baseline Commands

For the 7B backbone we plan to CPT:

```powershell
python scripts/run_text_eval.py --eval-dir evals/text_only --output-dir evals/results/qwen2_5_7b_base --model Qwen/Qwen2.5-7B --backend local-hf
```

Optional comparison against Qwen3's 8B base model:

```powershell
python scripts/run_text_eval.py --eval-dir evals/text_only --output-dir evals/results/qwen3_8b_base --model Qwen/Qwen3-8B-Base --backend local-hf
```

For a hosted OpenAI-compatible endpoint:

```powershell
python scripts/run_text_eval.py --eval-dir evals/text_only --output-dir evals/results/qwen2_5_7b_base --model Qwen/Qwen2.5-7B --backend openai-compatible --base-url https://YOUR_ENDPOINT/v1 --api-key YOUR_KEY
```

## Notes

This local Windows machine currently has no CUDA GPU visible to PyTorch, so 7B/8B local evaluation should be run on a GPU machine or hosted inference endpoint.
