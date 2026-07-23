#!/usr/bin/env python3
"""Train GEM-MM (SEGA) from a YAML config.

This entrypoint loads config + documents the full-FT SEGA loop. Wire your
local Qwen3-VL stack in ``build_model`` for production runs.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from gem_mm.config import GemMMConfig
from gem_mm.prompts import REPO_LIMIT128_SYSTEM_PROMPT

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("train_gem_mm")


def parse_args():
    p = argparse.ArgumentParser(description="Train GEM-MM with SEGA")
    p.add_argument("--config", type=str, default="configs/default_gem_mm.yaml")
    p.add_argument("--dry-run", action="store_true", help="Print resolved config and exit")
    return p.parse_args()


def main():
    args = parse_args()
    cfg = GemMMConfig.from_yaml(args.config)
    out = Path(cfg.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "resolved_config.json").write_text(json.dumps(cfg.to_dict(), indent=2), encoding="utf-8")
    logger.info("GEM-MM train | full_ft=%s lora=%s λ=%.2f top_m=%.2f k=%d",
                cfg.full_finetune, cfg.use_lora, cfg.lambda_weight, cfg.top_m_ratio, cfg.k_candidates)
    logger.info("system_prompt=%s", REPO_LIMIT128_SYSTEM_PROMPT[:80] + "...")
    if args.dry_run:
        logger.info("dry-run OK → %s", out / "resolved_config.json")
        return
    logger.warning(
        "Full trainer hook not bundled in the public skeleton yet. "
        "Use your local SEGA loop with GemMMConfig + gem_mm.entropy / sega helpers, "
        "or extend scripts/train_gem_mm.py with model loading."
    )


if __name__ == "__main__":
    main()
