# Quickstart

## 1. Environment

```bash
conda env create -f environment.yml
conda activate gem-mm
pip install -e .
```

## 2. Dry-run config resolution

```bash
python scripts/train_gem_mm.py --config configs/default_gem_mm.yaml --dry-run
python scripts/eval_preference.py --config configs/default_gem_mm.yaml --dry-run
```

## 3. Data

Prepare MM-RLHF JSONL splits (see [`docs/datasets.md`](../docs/datasets.md)) and
set `image_root` in the YAML.

## 4. Train / eval

Hook your local Qwen3-VL SEGA trainer to `gem_mm.entropy` + `GemMMConfig`, or
extend `scripts/train_gem_mm.py`. Keep **full fine-tune** for paper-comparable
runs (`use_lora: false`).
