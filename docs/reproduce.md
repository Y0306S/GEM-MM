# Reproducing experiments

## Hardware

- 1× 40GB GPU is enough for Qwen3-VL-4B full FT with gradient checkpointing  
- Set `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` if you hit fragmentation  

## Fair protocol

1. Fix train IDs = first 3000 short prompts; eval = next 1000  
2. Same base model for all methods  
3. Same eval prompt style (`repo_limit128`)  
4. Report preference accuracy, A/B choice accuracy, near-chosen, beat-rejected  
5. Add public benches (MMHal / Hallusion / MME) for the core bake-off  

## Baselines we compare

SFT, DPO, MM-DPO, mDPO, SIMA, and GEM-MM (full FT SEGA).

## Checklist before claiming numbers

- [ ] Checkpoint path recorded  
- [ ] Config YAML committed or hashed  
- [ ] Eval JSON files archived  
- [ ] No LoRA in reported GEM-MM rows  
- [ ] No unfair mid-budget (`*1500*`) as headline  

See also [`docs/conference_repo_checklist.md`](conference_repo_checklist.md).
