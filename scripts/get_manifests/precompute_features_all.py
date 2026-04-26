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

_REPO_ROOT = Path(__file__).resolve().parents[1]
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
    枚举 audio_root 下所有音频文件（递归），用于全量预计算。

    返回排序后的路径列表，便于多次运行顺序一致、断点续跑可预期。
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
    from transformers import AutoProcessor, WhisperModel

    processor = AutoProcessor.from_pretrained(model_id_or_path)
    model = WhisperModel.from_pretrained(model_id_or_path, torch_dtype=dtype)
    model = model.to(device)
    model.eval()
    return model.encoder, processor


def read_audio_mono(path: Path, *, max_samples: int | None) -> tuple[np.ndarray, int]:
    wav, sr = sf.read(str(path), dtype="float32", always_2d=False)
    if isinstance(wav, np.ndarray) and wav.ndim > 1:
        wav = wav.mean(axis=1)
    wav = np.asarray(wav, dtype=np.float32)
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
    if waveform.ndim > 1:
        waveform = waveform.mean(axis=1)
    w = np.asarray(waveform, dtype=np.float32)
    if max_samples is not None and w.shape[0] > max_samples:
        w = w[:max_samples]
    inputs = processor(w, sampling_rate=sample_rate, return_tensors="pt")
    input_features = inputs["input_features"].to(device=device, dtype=dtype)
    with torch.no_grad():
        out = encoder(input_features)
        hidden = out.last_hidden_state
    return hidden.squeeze(0).float().cpu().numpy().astype(np.float32)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=str, default=str(_REPO_ROOT / "configs/base.yaml"))
    p.add_argument("--audio-root", type=str, default=None)
    p.add_argument("--feature-root", type=str, default=None)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--force", action="store_true")
    p.add_argument(
        "--extensions",
        type=str,
        default=".flac,.wav",
        help="逗号分隔扩展名，如 .flac,.wav",
    )
    args = p.parse_args()

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
        feat = audio_path_to_feature_path(ap, audio_root=audio_root, feature_root=feature_root)
        if feat.exists() and not args.force:
            skipped += 1
            continue
        feat.parent.mkdir(parents=True, exist_ok=True)
        wav, sr = read_audio_mono(ap, max_samples=max_samples)
        if sr != sample_rate:
            raise RuntimeError(f"Bad sr {sr} for {ap}, expected {sample_rate}")
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
