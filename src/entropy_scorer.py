"""Entropy-guided token scoring for GEM-MM.

Reward (paper):
  r(y) = - mean(H_final) + λ * mean(top-ρ fork entropies on the CoT span)
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


class EntropyScorer:
    def __init__(self, config, tokenizer_or_processor):
        self.config = config
        # AutoProcessor or AutoTokenizer both expose .tokenizer / decode
        self.processor = tokenizer_or_processor
        self.tokenizer = getattr(tokenizer_or_processor, "tokenizer", tokenizer_or_processor)

    def compute_entropy(self, logits: torch.Tensor) -> torch.Tensor:
        probs = F.softmax(logits, dim=-1)
        log_probs = F.log_softmax(logits, dim=-1)
        return -torch.sum(probs * log_probs, dim=-1)

    def get_gem_score(self, input_ids, logits, prompt_len: int) -> torch.Tensor:
        response_ids = input_ids[prompt_len:]
        response_logits = logits[prompt_len - 1 : -1]
        if len(response_ids) == 0:
            return torch.tensor(0.0, device=logits.device)

        token_entropies = self.compute_entropy(response_logits)
        full_text = self.tokenizer.decode(response_ids, skip_special_tokens=True)

        # Prefer an explicit Final Answer marker; else use a fractional split
        # matching the paper's fork / final partition.
        if "Final Answer" in full_text:
            split_idx = int(len(token_entropies) * 0.75)
        else:
            split_idx = int(len(token_entropies) * 0.8)
        split_idx = max(1, min(split_idx, len(token_entropies) - 1))

        h_cot = token_entropies[:split_idx]
        h_final = token_entropies[split_idx:]

        score_final = (
            torch.mean(h_final)
            if len(h_final) > 0
            else torch.tensor(5.0, device=logits.device)
        )

        score_cot = torch.tensor(0.0, device=logits.device)
        if len(h_cot) > 0:
            m = max(1, int(len(h_cot) * self.config.top_m_ratio))
            top_m_vals, _ = torch.topk(h_cot, min(m, len(h_cot)))
            score_cot = torch.mean(top_m_vals)

        return -score_final + (self.config.lambda_weight * score_cot)
