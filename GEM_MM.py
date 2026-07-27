"""GEM-MM: Entropy-Guided Preference Alignment for Vision-Language Models.

Public entrypoint mirroring the parent GEM repository style:
  python GEM_MM.py --data_path data/preference_data.jsonl --image_root data/images
"""

from __future__ import annotations

import argparse

import torch

from src.config import Config
from src.dataset import MultimodalPreferenceDataset
from src.gem_trainer import GEMTrainer
from src.model_utils import load_model_and_processor
from src.sft_trainer import run_sft


def main():
    parser = argparse.ArgumentParser(description="Run GEM-MM (SFT → SEGA)")
    parser.add_argument("--data_path", type=str, default="data/preference_data.jsonl")
    parser.add_argument("--image_root", type=str, default="data/images")
    parser.add_argument("--output_dir", type=str, default="output/gem_mm_final")
    parser.add_argument("--skip_sft", action="store_true")
    parser.add_argument("--skip_sega", action="store_true")
    parser.add_argument(
        "--model_name",
        type=str,
        default=None,
        help="Override Config.model_name (e.g. Qwen/Qwen3-VL-4B-Instruct)",
    )
    args = parser.parse_args()

    config = Config()
    if args.model_name:
        config.model_name = args.model_name

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Loading {config.model_name}...")
    model, processor = load_model_and_processor(config, device)

    if not args.skip_sft:
        sft_dataset = MultimodalPreferenceDataset(
            args.data_path, args.image_root, processor, config, mode="sft"
        )
        model = run_sft(model, sft_dataset, config, device)

    if not args.skip_sega:
        gem_dataset = MultimodalPreferenceDataset(
            args.data_path, args.image_root, processor, config, mode="gen"
        )
        trainer = GEMTrainer(model, processor, config, device)
        trainer.train_sega(gem_dataset)

    print(f"Saving GEM-MM model to {args.output_dir} ...")
    model.save_pretrained(args.output_dir, safe_serialization=True)
    processor.save_pretrained(args.output_dir)
    print("Done.")


if __name__ == "__main__":
    main()
