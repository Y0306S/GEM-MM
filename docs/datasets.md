# Datasets

## MM-RLHF (main)

- Source: multimodal preference pairs with images  
- Our split convention: first **3000** unique short-prompt IDs for train, next
  **1000** held-out for eval (IDs 3001–4000)  
- Place prepared JSONL under `data/` (not shipped in git):

```text
data/mm_rlhf_train_gem.jsonl
data/mm_rlhf_eval_pref.jsonl
data/mm_rlhf_eval_gen.jsonl
```

Set `image_root` in the YAML config to your image directory.

## RLAIF-V (transfer)

- Hugging Face: `openbmb/RLAIF-V-Dataset`  
- Used as cross-dataset generalization evidence (same recipe family)

## Benchmarks (eval only)

| Suite | Scoring |
|-------|---------|
| MMHal-Bench | GPT judge (optional) / generations |
| HallusionBench | rule-based yes/no |
| MME | rule-based |
| POPE / AMBER / safety | API-free HAL suite |

Download instructions for public benches are environment-specific; keep paths
out of the repo and pass them via env vars.
