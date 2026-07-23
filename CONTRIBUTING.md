# Contributing

Thanks for your interest in GEM-MM.

## Ground rules

1. Open an issue before large API changes.  
2. Keep the public tree free of checkpoints, raw data, and secrets.  
3. Match existing code style (`ruff` if available).  
4. Prefer small, reviewable PRs.

## Dev setup

```bash
conda env create -f environment.yml
conda activate gem-mm
pip install -e ".[dev]"
python scripts/train_gem_mm.py --config configs/default_gem_mm.yaml --dry-run
```

## Pull requests

- Describe the motivation and how you tested  
- Update docs if behavior changes  
- Do not commit `.env`, API keys, or large binaries  
