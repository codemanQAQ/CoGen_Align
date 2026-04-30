"""Whisper 编码长度与真实时长对齐（帧数换算），供预计算落盘截断使用。"""

from __future__ import annotations

import math


def effective_audio_frames_from_duration(
    duration_sec: float | None,
    feat_len: int,
    *,
    max_duration: float,
    max_audio_frames: int,
) -> int:
    """
    将「秒」映射到 encoder 时间维应保留的帧数（**向上取整**）。

    假设 ``feat_len`` 对应满窗 ``max_duration`` 的整段 Whisper 输出；真实有效时长为
    ``min(duration_sec, max_duration)`` 时，按比例 ``ceil`` 得到截断长度，略多留帧以少裁语音尾。

    ``duration_sec`` 为 ``None`` 或 ``<=0`` 时，退回 ``min(feat_len, max_audio_frames)``。
    """
    if feat_len <= 0:
        return 0
    if duration_sec is None or duration_sec <= 0:
        return min(feat_len, max_audio_frames)
    effective_sec = min(float(duration_sec), max_duration)
    n_raw = effective_sec / max_duration * feat_len
    n = max(1, min(math.ceil(n_raw), feat_len))
    return min(n, max_audio_frames)
