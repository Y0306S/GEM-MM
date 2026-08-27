"""Multimodal preference dataset for GEM-MM.

Expected JSONL fields (one object per line):
  {
    "id": ...,
    "prompt": "...",          # text question / instruction
    "image": "rel/path.jpg",  # path under image_root (optional for text-only smoke)
    "chosen": "...",
    "rejected": "..."
  }
"""

from __future__ import annotations

import json
import os
from typing import Any

from torch.utils.data import Dataset


class MultimodalPreferenceDataset(Dataset):
    def __init__(
        self,
        data_path: str,
        image_root: str,
        processor: Any,
        config: Any,
        mode: str = "sft",
    ) -> None:
        self.image_root = image_root
        self.processor = processor
        self.config = config
        self.mode = mode
        self.rows: list[dict] = []
        with open(data_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    self.rows.append(json.loads(line))

    def __len__(self) -> int:
        return len(self.rows)

    def _image_path(self, row: dict) -> str | None:
        rel = row.get("image") or row.get("image_path")
        if not rel:
            return None
        if os.path.isabs(rel):
            return rel
        return os.path.join(self.image_root, rel)

    def __getitem__(self, idx: int) -> dict:
        row = self.rows[idx]
        prompt = row["prompt"]
        # SFT uses chosen; SEGA generation uses the prompt (+ image) only.
        target = row.get("chosen", "") if self.mode == "sft" else ""
        return {
            "id": row.get("id", idx),
            "prompt": prompt,
            "image_path": self._image_path(row),
            "chosen": row.get("chosen", ""),
            "rejected": row.get("rejected", ""),
            "target": target,
        }
