from __future__ import annotations

import torch
from torch.nn.utils.rnn import pad_sequence


def collate_stage1(batch: list[dict]) -> dict:
    audio_feats = pad_sequence(
        [b["audio_feat"] for b in batch], batch_first=True, padding_value=0.0
    )
    audio_lens = torch.tensor([b["audio_len"] for b in batch])
    audio_mask = torch.arange(audio_feats.size(1))[None, :] >= audio_lens[:, None]

    text_ids = pad_sequence(
        [b["text_ids"] for b in batch], batch_first=True, padding_value=0
    )
    text_lens = torch.tensor([b["text_len"] for b in batch])
    text_mask = torch.arange(text_ids.size(1))[None, :] >= text_lens[:, None]

    out: dict = {
        "audio_feat": audio_feats,
        "audio_mask": audio_mask,
        "text_ids": text_ids,
        "text_mask": text_mask,
    }

    if batch and "hard_neg_text_ids" in batch[0]:
        # Minimal path: first hard negative only (extend as needed)
        hn_list = []
        hn_lens = []
        for b in batch:
            t0 = b["hard_neg_text_ids"][0]
            hn_list.append(t0)
            hn_lens.append(t0.size(0))
        hn = pad_sequence(hn_list, batch_first=True, padding_value=0)
        hn_lens_t = torch.tensor(hn_lens)
        hn_mask = (torch.arange(hn.size(1))[None, :] >= hn_lens_t[:, None]).unsqueeze(1)
        out["hard_neg_text_ids"] = hn.unsqueeze(1)
        out["hard_neg_mask"] = hn_mask

    return out
