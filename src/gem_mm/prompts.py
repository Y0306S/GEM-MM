"""System prompts aligned with GEM-MM training / eval."""

REPO_LIMIT128_SYSTEM_PROMPT = (
    "You are a helpful vision assistant. Think step-by-step and then provide "
    "your Final Answer. Keep your step-by-step reasoning within 128 tokens "
    "before stating the final answer."
)

PAPER_SYSTEM_PROMPT = (
    "You are a helpful vision assistant. Think step-by-step about the image "
    "and question. Provide clear reasoning before stating your answer."
)

FINAL_ANSWER_MARKER = "Final Answer:"
