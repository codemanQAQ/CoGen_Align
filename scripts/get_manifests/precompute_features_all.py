#!/usr/bin/env python3
"""
全量扫描 audio_root 下所有 .flac/.wav，按目录树写入 feature_root 下的 .npy（不读 manifest）。

第二步请用 attach_feature_paths.py 给 manifest 插入 feature_path。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from tqdm import tqdm

# 本文件在 scripts/get_manifests/，仓库根为 parents[2]
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "src"))

from cogen_align.data.feature_paths import audio_path_to_feature_path  # noqa: E402
from cogen_align.utils.config import load_config  # noqa: E402


def iter_audio_files(
    audio_root: Path,
    *,
    extensions: tuple[str, ...] = (".flac", ".wav"),
) -> list[Path]:
    """
    功能：在 ``audio_root`` 下递归查找符合扩展名的音频文件，供全量预计算遍历。

    Args:
        audio_root: 音频根目录（如 LibriSpeech 根）。
        extensions: 要匹配的扩展名元组，例如 ``(".flac", ".wav")``。

    Returns:
        去重后按路径排序的 ``Path`` 列表；顺序稳定，便于重复运行与断点续跑。
    """
    audio_root = audio_root.resolve()
    found: list[Path] = []
    for ext in extensions:
        found.extend(audio_root.rglob(f"*{ext}"))
    # 过滤目录误匹配、只保留文件
    found = sorted(p.resolve() for p in found if p.is_file())
    return found


def load_whisper_encoder_bundle(
    model_id_or_path: str,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[torch.nn.Module, object]:
    """
    功能：从 Hub 或本地路径加载 Whisper 的 Processor 与 **Encoder**（不含 decoder）。

    Args:
        model_id_or_path: 模型 id（如 ``openai/whisper-large-v3``）或本地目录。
        device: 推理设备。
        dtype: 模型权重的浮点类型（如 ``bfloat16`` / ``float32``）。

    Returns:
        ``(encoder, processor)``，encoder 已 ``eval()`` 并置于 ``device``。
    """
    # 延迟 import，避免仅 --help 时也拉 transformers
    from transformers import AutoProcessor, WhisperModel

    # 只加载 encoder：Stage1 仅需 last_hidden_state，省显存与算力
    processor = AutoProcessor.from_pretrained(model_id_or_path)
    model = WhisperModel.from_pretrained(model_id_or_path, torch_dtype=dtype)
    model = model.to(device)
    model.eval()
    return model.encoder, processor


def read_audio_mono(path: Path, *, max_samples: int | None) -> tuple[np.ndarray, int]:
    """
    功能：从磁盘读取单条音频为 float32 单声道波形，并可按样本数上限截断。

    Args:
        path: 音频文件路径。
        max_samples: 最大样本数；``None`` 表示不截断。用于与训练 ``max_duration`` 对齐。

    Returns:
        ``(wav, sr)``，``wav`` 为一维 ``float32``，``sr`` 为采样率。
    """
    # soundfile 解码，不依赖 torchcodec；多声道取均值变单声道
    wav, sr = sf.read(str(path), dtype="float32", always_2d=False)
    if isinstance(wav, np.ndarray) and wav.ndim > 1:
        wav = wav.mean(axis=1)
    wav = np.asarray(wav, dtype=np.float32)
    # 与训练侧 max_duration 一致：过长波形在入 Whisper 前截断，避免 OOM
    if max_samples is not None and wav.shape[0] > max_samples:
        wav = wav[:max_samples]
    return wav, int(sr)


def waveform_to_encoder_features(
    encoder: torch.nn.Module,
    processor: object,
    waveform: np.ndarray,
    sample_rate: int,
    *,
    device: torch.device,
    dtype: torch.dtype,
    max_samples: int | None,
) -> np.ndarray:
    """
    功能：将波形经 Whisper Processor 转成 log-mel 特征，再经 Encoder 得到帧级 hidden，
    作为一条 utterance 的预计算特征（numpy float32，形状 ``[T, D]``）。

    Args:
        encoder: Whisper 的 encoder 子模块。
        processor: ``AutoProcessor`` 实例，负责特征提取与张量打包。
        waveform: 单声道 float32 波形。
        sample_rate: 波形采样率（需与配置一致，如 16000）。
        device: 前向计算设备。
        dtype: ``input_features`` 在 GPU 上的 dtype（如 bfloat16）。
        max_samples: 非 ``None`` 时在函数内再次按样本数截断波形；与 ``read_audio_mono`` 二选一使用时可传 ``None``。

    Returns:
        Encoder ``last_hidden_state`` 去掉 batch 维后的 ``numpy.ndarray``，dtype float32。
    """
    if waveform.ndim > 1:
        waveform = waveform.mean(axis=1)
    w = np.asarray(waveform, dtype=np.float32)
    if max_samples is not None and w.shape[0] > max_samples:
        w = w[:max_samples]
    # processor 负责 mel / 长度等；输出 shape 约 [1, T', D]
    inputs = processor(w, sampling_rate=sample_rate, return_tensors="pt")
    input_features = inputs["input_features"].to(device=device, dtype=dtype)
    with torch.no_grad():
        out = encoder(input_features)
        hidden = out.last_hidden_state
    # 去掉 batch 维，落盘 float32 以省空间、与 Dataset 读取一致
    return hidden.squeeze(0).float().cpu().numpy().astype(np.float32)


def main() -> None:
    """
    功能：解析命令行与 YAML 配置，枚举 ``audio_root`` 下全部音频，按 ``audio_path_to_feature_path``
    映射在 ``feature_root`` 下写出对应 ``.npy``；已存在文件默认跳过，``--force`` 时覆盖。

    不读取 manifest；生成清单后请用 ``attach_feature_paths.py`` 写入 ``feature_path``。
    """
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=str, default=str(_REPO_ROOT / "configs/base.yaml"))
    p.add_argument("--audio-root", type=str, default=None)
    p.add_argument("--feature-root", type=str, default=None)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument(
        "--force",
        action="store_true",
        help="已存在同名 .npy 时仍重算并覆盖",
    )
    p.add_argument(
        "--extensions",
        type=str,
        default=".flac,.wav",
        help="逗号分隔扩展名，如 .flac,.wav",
    )
    args = p.parse_args()

    # 支持传入绝对路径，或相对仓库根的 configs/xxx.yaml
    cfg_path = Path(args.config)
    if not cfg_path.is_file():
        cfg_path = _REPO_ROOT / args.config
    cfg = load_config(cfg_path)
    data_cfg = cfg.get("data", {})

    audio_root = Path(
        args.audio_root or data_cfg.get("audio_root", "")
    ).expanduser().resolve()
    feature_root = Path(
        args.feature_root or data_cfg.get("feature_root", "")
    ).expanduser().resolve()
    whisper_name = str(data_cfg.get("whisper_model", "openai/whisper-large-v3"))
    sample_rate = int(data_cfg.get("sample_rate", 16000))
    max_duration = float(data_cfg.get("max_duration", 30.0))
    # 与 manifest 生成、训练 collator 使用同一套时长上限
    max_samples = int(max_duration * sample_rate)

    exts = tuple(x.strip() for x in args.extensions.split(",") if x.strip())
    files = iter_audio_files(audio_root, extensions=exts)
    if not files:
        raise SystemExit(f"No audio files under {audio_root} with extensions {exts}")

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    encoder, processor = load_whisper_encoder_bundle(whisper_name, device=device, dtype=dtype)

    skipped = 0
    done = 0
    for ap in tqdm(files, desc="precompute all", ncols=100):
        # 与 attach_feature_paths 使用同一规则，保证 manifest 里的路径可解析
        feat = audio_path_to_feature_path(ap, audio_root=audio_root, feature_root=feature_root)
        if feat.exists() and not args.force:
            skipped += 1
            continue
        feat.parent.mkdir(parents=True, exist_ok=True)
        wav, sr = read_audio_mono(ap, max_samples=max_samples)
        # LibriSpeech 为 16k；若混了其它语料需先重采样或放宽此处检查
        if sr != sample_rate:
            raise RuntimeError(f"Bad sr {sr} for {ap}, expected {sample_rate}")
        # read 时已按 max_samples 截断，这里不再截，避免双截断不一致
        feats = waveform_to_encoder_features(
            encoder,
            processor,
            wav,
            sr,
            device=device,
            dtype=dtype,
            max_samples=None,
        )
        np.save(str(feat), feats)
        done += 1

    print(f"[OK] audio_root={audio_root} feature_root={feature_root}")
    print(f"[OK] total_files={len(files)} written={done} skipped_existing={skipped}")


if __name__ == "__main__":
    main()
