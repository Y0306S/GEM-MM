"""Supervised warm-start on chosen responses (multimodal)."""

from __future__ import annotations

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm


def run_sft(model, dataset, config, device):
    """Minimal SFT loop. Production runs should use the paper's full recipe
    (pixel bounds, chat template, gradient checkpointing, full FT).
    """
    model.train()
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=config.sft_lr,
    )
    loader = DataLoader(dataset, batch_size=config.sft_batch_size, shuffle=True)

    print(f"[SFT] epochs={config.sft_epochs} lr={config.sft_lr} n={len(dataset)}")
    for epoch in range(config.sft_epochs):
        for batch in tqdm(loader, desc=f"sft-epoch-{epoch+1}"):
            # Placeholder: wire processor.apply_chat_template + labels here.
            # Kept intentionally light so the public entrypoint mirrors GEM.py.
            _ = batch
            optimizer.zero_grad(set_to_none=True)
    print("[SFT] done (wire full multimodal forward for real training)")
    return model
