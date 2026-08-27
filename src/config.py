from dataclasses import dataclass


@dataclass
class Config:
    # Default is a small VL checkpoint for smoke tests. For paper runs use
    # Qwen/Qwen3-VL-4B-Instruct (or a local path) with full fine-tuning.
    model_name: str = "Qwen/Qwen2-VL-2B-Instruct"
    max_length: int = 512
    max_new_tokens: int = 128
    min_pixels: int = 200_704
    max_pixels: int = 401_408

    sft_lr: float = 2e-5
    sft_epochs: int = 1
    sft_batch_size: int = 1

    k_candidates: int = 3
    temperature: float = 0.9
    sega_lr: float = 1e-5
    sega_epochs: int = 1
    gradient_accumulation_steps: int = 1

    # Paper primary: λ=2.0, ρ=0.05
    lambda_weight: float = 2.0
    top_m_ratio: float = 0.05

    system_prompt: str = (
        "You are a helpful vision assistant. "
        "Think step-by-step and then provide your Final Answer. "
        "Keep your step-by-step reasoning within 128 tokens before stating "
        "the final answer."
    )
