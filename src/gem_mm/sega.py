"""SEGA training loop skeleton for GEM-MM (full fine-tune).

This is the public reference implementation outline. Plug in your VLM
forward / generate utilities (Qwen3-VL message format + vision tokens).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Callable

from .config import GemMMConfig
from .entropy import fork_entropy_reward, group_normalize

logger = logging.getLogger(__name__)


@dataclass
class Candidate:
    text: str
    token_entropies: list[float]
    final_answer_bonus: float = 0.0


class SegaTrainer:
    """On-policy SEGA with GEM fork-entropy rewards."""

    def __init__(
        self,
        model: Any,
        optimizer: Any,
        config: GemMMConfig,
        generate_fn: Callable[..., list[Candidate]],
    ) -> None:
        self.model = model
        self.optimizer = optimizer
        self.config = config
        self.generate_fn = generate_fn

    def score_group(self, candidates: list[Candidate]) -> list[float]:
        rewards = [
            fork_entropy_reward(
                c.token_entropies,
                lambda_weight=self.config.lambda_weight,
                top_m_ratio=self.config.top_m_ratio,
                final_answer_bonus=c.final_answer_bonus,
            )
            for c in candidates
        ]
        return group_normalize(rewards)

    def step(self, batch: dict[str, Any]) -> dict[str, float]:
        """One SEGA step: sample k candidates → reward → advantage-weighted update.

        ``generate_fn`` must return per-token entropies at fork positions for
        each candidate. The concrete log-prob loss is model-specific; callers
        should attach an ``apply_update(advantages, candidates, batch)`` hook
        for production training (see scripts/train_gem_mm.py).
        """
        candidates = self.generate_fn(batch, k=self.config.k_candidates)
        advantages = self.score_group(candidates)
        loss = self._dummy_loss(advantages)
        self.optimizer.zero_grad(set_to_none=True)
        if hasattr(loss, "backward"):
            loss.backward()
            self.optimizer.step()
        mean_adv = sum(advantages) / max(1, len(advantages))
        logger.info("sega_step mean_advantage=%.4f n=%d", mean_adv, len(advantages))
        return {"mean_advantage": float(mean_adv), "n_candidates": float(len(candidates))}

    @staticmethod
    def _dummy_loss(advantages: list[float]):
        """Placeholder so the skeleton imports cleanly without a live model."""
        try:
            import torch

            return torch.tensor(0.0, requires_grad=True) * 0.0 + sum(advantages) * 0.0
        except Exception:
            return 0.0
