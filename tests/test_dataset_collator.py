from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from cogen_align.data.collator import collate_stage1
from cogen_align.data.dataset import SpeechTextDataset
from cogen_align.data.effective_frames import effective_audio_frames_from_duration


class _FakeTokenizer:
    """避免单测依赖 HuggingFace：固定长度 token id。"""

    def __call__(self, text: str, max_length: int, truncation: bool, return_tensors: str):
        del truncation, return_tensors
        n = min(len(text), max_length)
        ids = [1 + (ord(c) % 50) for c in text[:n]]
        if not ids:
            ids = [1]
        t = torch.tensor([ids], dtype=torch.long)
        return type("TokOut", (), {"input_ids": t})()


def test_dataset_and_collator_roundtrip(tmp_path: Path):
    feat_dir = tmp_path / "feats"
    feat_dir.mkdir()
    np.save(feat_dir / "a.npy", np.random.randn(20, 1280).astype(np.float32))
    np.save(feat_dir / "b.npy", np.random.randn(15, 1280).astype(np.float32))

    manifest = tmp_path / "m.jsonl"
    lines = [
        {
            "id": "a",
            "audio_path": "/dev/null/a.flac",
            "feature_path": str(feat_dir / "a.npy"),
            "text": "hello",
            "duration": 1.0,
        },
        {
            "id": "b",
            "audio_path": "/dev/null/b.flac",
            "feature_path": str(feat_dir / "b.npy"),
            "text": "world",
            "duration": 1.0,
        },
    ]
    manifest.write_text("\n".join(json.dumps(x) for x in lines) + "\n", encoding="utf-8")

    tok = _FakeTokenizer()
    ds = SpeechTextDataset(manifest, tok, max_audio_frames=100, max_text_tokens=16)
    assert len(ds) == 2
    batch = collate_stage1([ds[0], ds[1]])
    assert batch["audio_feat"].dim() == 3
    assert batch["audio_feat"].shape[0] == 2
    assert batch["audio_mask"].dtype == torch.bool
    assert batch["text_ids"].shape[0] == 2


def test_collator_hard_negatives(tmp_path: Path):
    feat_dir = tmp_path / "feats"
    feat_dir.mkdir()
    for name in ("x.npy", "y.npy"):
        np.save(feat_dir / name, np.random.randn(10, 1280).astype(np.float32))

    manifest = tmp_path / "m.jsonl"
    rows = [
        {
            "id": "x",
            "audio_path": "/x.flac",
            "feature_path": str(feat_dir / "x.npy"),
            "text": "aa",
            "duration": 1.0,
        },
        {
            "id": "y",
            "audio_path": "/y.flac",
            "feature_path": str(feat_dir / "y.npy"),
            "text": "bb",
            "duration": 1.0,
        },
    ]
    manifest.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    lookup = {"x": ["y"]}
    tok = _FakeTokenizer()
    ds = SpeechTextDataset(
        manifest,
        tok,
        max_audio_frames=50,
        max_text_tokens=8,
        load_hard_negs=True,
        hard_neg_lookup=lookup,
    )
    batch = collate_stage1([ds[0]])
    assert "hard_neg_text_ids" in batch
    assert batch["hard_neg_text_ids"].shape[0] == 1


def test_dataset_uses_pretrimmed_npy_length(tmp_path: Path):
    """预计算已截断的 .npy：Dataset 整段使用（仅受 max_audio_frames 限制）。"""
    feat_dir = tmp_path / "feats"
    feat_dir.mkdir()
    np.save(feat_dir / "u.npy", np.random.randn(88, 1280).astype(np.float32))
    manifest = tmp_path / "m.jsonl"
    manifest.write_text(
        json.dumps(
            {
                "id": "u",
                "audio_path": "/u.flac",
                "feature_path": str(feat_dir / "u.npy"),
                "text": "hi",
                "duration": 15.0,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    ds = SpeechTextDataset(
        manifest,
        _FakeTokenizer(),
        max_audio_frames=2000,
    )
    item = ds[0]
    assert item["audio_feat"].shape[0] == 88
    assert item["audio_len"] == 88


def test_dataset_caps_max_audio_frames(tmp_path: Path):
    feat_dir = tmp_path / "feats"
    feat_dir.mkdir()
    np.save(feat_dir / "u.npy", np.random.randn(500, 1280).astype(np.float32))
    manifest = tmp_path / "m.jsonl"
    manifest.write_text(
        json.dumps(
            {
                "id": "u",
                "audio_path": "/u.flac",
                "feature_path": str(feat_dir / "u.npy"),
                "text": "hi",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    ds = SpeechTextDataset(manifest, _FakeTokenizer(), max_audio_frames=100)
    item = ds[0]
    assert item["audio_feat"].shape[0] == 100
    assert item["audio_len"] == 100


def test_effective_frames_helper():
    assert effective_audio_frames_from_duration(
        15.0, 1500, max_duration=30.0, max_audio_frames=2000
    ) == 750
    # 1/30*100 ≈ 3.33，round 得 3，ceil 得 4（多留一帧）
    assert effective_audio_frames_from_duration(
        1.0, 100, max_duration=30.0, max_audio_frames=200
    ) == 4
    assert effective_audio_frames_from_duration(
        None, 100, max_duration=30.0, max_audio_frames=50
    ) == 50
