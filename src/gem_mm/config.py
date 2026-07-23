"""Configuration for GEM-MM training and evaluation."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class GemMMConfig:
    """Defaults match our fair @3000 full-FT SEGA recipe."""

    model_name: str = "Qwen/Qwen3-VL-4B-Instruct"
    output_dir: str = "outputs/gem_mm"

    # Generation / SEGA
    k_candidates: int = 5
    temperature: float = 0.9
    sega_max_new_tokens: int = 192
    sega_lr: float = 1e-5
    lambda_weight: float = 2.0
    top_m_ratio: float = 0.1
    entropy_split_style: str = "repo"  # repo | mm
    prompt_style: str = "repo_limit128"

    # Vision
    min_pixels: int = 200704
    max_pixels: int = 401408

    # Optimization
    full_finetune: bool = True
    use_lora: bool = False
    gradient_checkpointing: bool = True
    max_steps: int = 3000
    save_every_n_steps: int = 500
    seed: int = 42

    # Data
    train_file: str = "data/mm_rlhf_train_gem.jsonl"
    eval_pref_file: str = "data/mm_rlhf_eval_pref.jsonl"
    image_root: str = ""

    extra: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_yaml(cls, path: str | Path) -> "GemMMConfig":
        raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
        known = {f.name for f in cls.__dataclass_fields__.values()}  # type: ignore[attr-defined]
        kwargs = {k: v for k, v in raw.items() if k in known}
        extra = {k: v for k, v in raw.items() if k not in known}
        cfg = cls(**kwargs)
        cfg.extra = extra
        return cfg

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
