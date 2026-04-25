from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


class SpeechTextDataset(Dataset):
    """
    Manifest jsonl lines:
    {"id": "...", "audio_path": "...", "feature_path": "...", "text": "...",
     "duration": 1.0, "hard_neg_ids": []}
    """

    def __init__(
        self,
        manifest_path: str | Path,
        tokenizer,
        max_audio_frames: int = 750,
        max_text_tokens: int = 128,
        load_hard_negs: bool = False,
        hard_neg_lookup: dict | None = None,
    ):
        self.tokenizer = tokenizer
        self.max_audio_frames = max_audio_frames
        self.max_text_tokens = max_text_tokens
        self.load_hard_negs = load_hard_negs
        self.hard_neg_lookup = hard_neg_lookup or {}

        self.samples: list[dict] = []
        with open(manifest_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                self.samples.append(json.loads(line))
        self.id_to_idx = {s["id"]: i for i, s in enumerate(self.samples)}

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        sample = self.samples[idx]
        feat = np.load(sample["feature_path"]).astype(np.float32)
        feat = torch.from_numpy(feat[: self.max_audio_frames])
        enc = self.tokenizer(
            sample["text"],
            max_length=self.max_text_tokens,
            truncation=True,
            return_tensors="pt",
        )
        text_ids = enc.input_ids[0]

        item: dict = {
            "audio_feat": feat,
            "audio_len": feat.size(0),
            "text_ids": text_ids,
            "text_len": text_ids.size(0),
            "id": sample["id"],
        }

        if self.load_hard_negs and self.hard_neg_lookup:
            neg_ids = self.hard_neg_lookup.get(sample["id"], [])[:4]
            neg_text_ids = []
            for nid in neg_ids:
                neg_sample = self.samples[self.id_to_idx[nid]]
                nenc = self.tokenizer(
                    neg_sample["text"],
                    max_length=self.max_text_tokens,
                    truncation=True,
                    return_tensors="pt",
                )
                neg_text_ids.append(nenc.input_ids[0])
            item["hard_neg_text_ids"] = neg_text_ids

        return item
