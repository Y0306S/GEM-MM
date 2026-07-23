"""Fork-token entropy reward helpers for GEM-MM.

This module exposes the scoring interface used by SEGA. The public release
keeps the math explicit; wire it to your model's next-token distributions
in ``sega.py``.
"""

from __future__ import annotations

import math
from typing import Sequence


def token_entropy(probs: Sequence[float]) -> float:
    """Shannon entropy H = -sum p log p (nats)."""
    h = 0.0
    for p in probs:
        if p > 0.0:
            h -= p * math.log(p)
    return h


def select_top_m_forks(entropies: Sequence[float], top_m_ratio: float) -> list[int]:
    """Return indices of the highest-entropy positions (top-m fraction)."""
    if not entropies:
        return []
    m = max(1, int(math.ceil(len(entropies) * top_m_ratio)))
    order = sorted(range(len(entropies)), key=lambda i: entropies[i], reverse=True)
    return order[:m]


def fork_entropy_reward(
    token_entropies: Sequence[float],
    *,
    lambda_weight: float = 2.0,
    top_m_ratio: float = 0.1,
    final_answer_bonus: float = 0.0,
) -> float:
    """GEM-style reward from fork entropies + optional final-answer term.

    r = λ * mean(H(top-m forks)) + final_answer_bonus
    """
    idxs = select_top_m_forks(token_entropies, top_m_ratio)
    if not idxs:
        fork = 0.0
    else:
        fork = sum(token_entropies[i] for i in idxs) / len(idxs)
    return float(lambda_weight * fork + final_answer_bonus)


def group_normalize(rewards: Sequence[float], eps: float = 1e-6) -> list[float]:
    """Per-prompt group normalization → advantages."""
    if not rewards:
        return []
    mean = sum(rewards) / len(rewards)
    var = sum((r - mean) ** 2 for r in rewards) / len(rewards)
    std = math.sqrt(var + eps)
    return [(r - mean) / std for r in rewards]
