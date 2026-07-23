# Method overview

GEM-MM adapts **GEM** (entropy-guided preference learning with SEGA) from text-only
LLMs to **vision-language models**.

## Reward

For each on-policy candidate completion, compute next-token entropy at *fork*
positions (high-uncertainty tokens under the model). Keep the top-$m$ fraction
and form:

$$
r = \lambda \cdot \frac{1}{|F|}\sum_{t \in F} H(\pi(\cdot \mid x, y_{<t})) + b_{\text{final}}
$$

where $F$ is the top-$m$ fork set and $b_{\text{final}}$ rewards well-formed
final answers (optional).

## Advantage + SEGA

Within a prompt group of $k$ candidates, normalize rewards to advantages and
apply an on-policy SEGA update (full fine-tune of the VLM with gradient
checkpointing in our reported runs).

## VLM-specific choices

| Choice | Setting |
|--------|---------|
| Backbone | Qwen3-VL-4B-Instruct |
| Prompt | `repo_limit128` system prompt + image/question |
| Tokens | `sega_max_new_tokens=192` (default recipe) |
| Pixels | `min_pixels` / `max_pixels` bounds for vision tokens |
| Tuning | **Full FT** (no LoRA in reported paper runs) |

## Ablations

Configs under `configs/` cover removing fork entropy, fork-only variants, and
hparam sweeps used in the paper appendix.
