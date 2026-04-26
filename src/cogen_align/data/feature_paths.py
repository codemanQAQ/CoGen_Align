"""音频路径与预计算特征 .npy 路径之间的映射（与 LibriSpeech 目录树一致）。"""

from __future__ import annotations

from pathlib import Path


def audio_path_to_feature_path(
    audio_path: Path,
    *,
    audio_root: Path,
    feature_root: Path,
) -> Path:
    """
    将一条音频的绝对路径映射为特征 .npy 的绝对路径。

    在 feature_root 下复刻 audio_root 的相对子目录，仅把扩展名改为 .npy。
    """
    audio_path = audio_path.resolve()
    audio_root = audio_root.resolve()
    feature_root = feature_root.resolve()
    rel = audio_path.relative_to(audio_root)
    return (feature_root / rel).with_suffix(".npy")
