#!/usr/bin/env python3
"""Evaluate preference accuracy on held-out multimodal DPO pairs."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from gem_mm.config import GemMMConfig

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("eval_preference")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/default_gem_mm.yaml")
    p.add_argument("--checkpoint", required=False, default="")
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    cfg = GemMMConfig.from_yaml(args.config)
    pref = Path(cfg.eval_pref_file)
    logger.info("eval pref file: %s exists=%s", pref, pref.is_file())
    logger.info("checkpoint: %s", args.checkpoint or "(none)")
    if args.dry_run:
        print(json.dumps({"config": cfg.to_dict(), "pref_exists": pref.is_file()}, indent=2))
        return
    logger.warning(
        "Preference scorer is environment-specific (VLM forward on chosen/rejected). "
        "Plug in your local eval harness; keep prompt_style=%s for fair comparison.",
        cfg.prompt_style,
    )


if __name__ == "__main__":
    main()
