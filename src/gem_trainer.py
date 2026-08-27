"""GEM-MM SEGA trainer (on-policy, group-normalized advantages)."""

from __future__ import annotations

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.entropy_scorer import EntropyScorer


class GEMTrainer:
    def __init__(self, model, processor, config, device):
        self.model = model
        self.processor = processor
        self.config = config
        self.device = device
        self.scorer = EntropyScorer(config, processor)
        self.optimizer = torch.optim.AdamW(
            [p for p in model.parameters() if p.requires_grad],
            lr=config.sega_lr,
        )

    def train_sega(self, dataset):
        self.model.train()
        loader = DataLoader(dataset, batch_size=1, shuffle=True)
        print(
            f"[SEGA] k={self.config.k_candidates} λ={self.config.lambda_weight} "
            f"ρ={self.config.top_m_ratio} lr={self.config.sega_lr}"
        )
        for epoch in range(self.config.sega_epochs):
            for batch in tqdm(loader, desc=f"sega-epoch-{epoch+1}"):
                # Production recipe (see paper Algorithm 1):
                # 1) sample k CoT completions for the multimodal prompt
                # 2) score each with EntropyScorer.get_gem_score
                # 3) group-normalize advantages within the k-set
                # 4) advantage-weighted NLL update (full fine-tune)
                _ = batch
                self.optimizer.zero_grad(set_to_none=True)
        print("[SEGA] done (wire generate + entropy score + update for real training)")
