"""Model / processor loading helpers for Qwen2/3-VL."""

from __future__ import annotations

import torch


def load_model_and_processor(config, device):
    """Load a vision-language model. Tries Qwen3-VL first, then Qwen2-VL."""
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    model_name = config.model_name

    try:
        from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

        model = Qwen3VLForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype=dtype,
            device_map="auto" if torch.cuda.is_available() else None,
            trust_remote_code=True,
        )
        processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
    except Exception:
        from transformers import AutoProcessor, Qwen2VLForConditionalGeneration

        model = Qwen2VLForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype=dtype,
            device_map="auto" if torch.cuda.is_available() else None,
            trust_remote_code=True,
        )
        processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)

    if not torch.cuda.is_available():
        model.to(device)
    model.gradient_checkpointing_enable()
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()
    model.config.use_cache = False
    return model, processor
