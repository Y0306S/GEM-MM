"""GEM-MM: Entropy-guided multimodal preference alignment."""

__version__ = "0.1.0"

from .config import GemMMConfig
from .entropy import fork_entropy_reward
from .prompts import REPO_LIMIT128_SYSTEM_PROMPT

__all__ = [
    "GemMMConfig",
    "fork_entropy_reward",
    "REPO_LIMIT128_SYSTEM_PROMPT",
    "__version__",
]
