#!/usr/bin/env python3
"""
Prepare jsonl manifests from a standard LibriSpeech directory.

Input layout (standard):
  <librispeech_root>/
    train-clean-100/
    train-clean-360/
    train-other-500/
    dev-clean/
    dev-other/
    test-clean/
    test-other/

Each chapter directory contains:
  <utt_id>.flac
  <speaker>-<chapter>.trans.txt  (lines: "<utt_id> <text>")

Output:
  <out_dir>/*.jsonl

Manifest format (one json per line):
  {
    "id": "103-1240-0000",
    "audio_path": "/abs/.../103-1240-0000.flac",
    "feature_path": "/abs/.../features/103-1240-0000.npy",   # optional
    "text": "HELLO WORLD",
    "duration": 1.23
  }
"""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass
from pathlib import Path

import soundfile as sf
from tqdm import tqdm


@dataclass(frozen=True)
class Utterance:
    utt_id: str
    audio_path: Path
    text: str
    duration: float


def _read_trans_file(trans_path: Path) -> dict[str, str]:
    mapping: dict[str, str] = {}
    with open(trans_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(maxsplit=1)
            if len(parts) != 2:
                continue
            utt_id, text = parts
            mapping[utt_id] = text
    return mapping


def _iter_split_utts(split_dir: Path) -> list[Utterance]:
    if not split_dir.exists():
        raise FileNotFoundError(f"Split directory not found: {split_dir}")

    utts: list[Utterance] = []
    trans_files = list(split_dir.rglob("*.trans.txt"))
    if not trans_files:
        # 允许某些 split 尚未下载完成或为空：直接跳过，避免中断整体生成流程
        print(f"[Skip] No *.trans.txt found under {split_dir}")
        return []

    for trans_path in tqdm(
        trans_files,
        desc=f"scan trans.txt ({split_dir.name})",
        ncols=100,
    ):
        trans = _read_trans_file(trans_path)
        chapter_dir = trans_path.parent
        for utt_id, text in trans.items():
            audio_path = chapter_dir / f"{utt_id}.flac"
            if not audio_path.exists():
                # Some corpora may use .wav; keep flac as default but be tolerant.
                alt = chapter_dir / f"{utt_id}.wav"
                if alt.exists():
                    audio_path = alt
                else:
                    continue
            try:
                info = sf.info(str(audio_path))
                duration = float(info.frames) / float(info.samplerate)
            except Exception:
                continue
            utts.append(
                Utterance(
                    utt_id=utt_id,
                    audio_path=audio_path.resolve(),
                    text=text,
                    duration=duration,
                )
            )
    return utts


def _write_jsonl(
    items: list[Utterance],
    out_path: Path,
    *,
    feature_root: Path | None,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for u in items:
            obj = {
                "id": u.utt_id,
                "audio_path": str(u.audio_path),
                "text": u.text,
                "duration": round(float(u.duration), 6),
            }
            if feature_root is not None:
                obj["feature_path"] = str((feature_root / f"{u.utt_id}.npy").resolve())
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def _take_by_hours(items: list[Utterance], hours: float) -> tuple[list[Utterance], list[Utterance]]:
    if hours <= 0:
        return [], items
    target = hours * 3600.0
    picked: list[Utterance] = []
    acc = 0.0
    for u in items:
        if acc >= target:
            break
        picked.append(u)
        acc += u.duration
    rest = items[len(picked) :]
    return picked, rest


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--librispeech_root",
        type=str,
        required=True,
        help="Path to LibriSpeech root (contains train-clean-100/, dev-clean/, etc.)",
    )
    p.add_argument(
        "--out_dir",
        type=str,
        default="data/manifests/all",
        help="Output directory for jsonl manifests (default: data/manifests)",
    )
    p.add_argument(
        "--feature_root",
        type=str,
        default=None,
        help="If set, write feature_path=<feature_root>/<utt_id>.npy into manifests",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Shuffle seed for train subset sampling",
    )
    p.add_argument(
        "--train_hours",
        type=float,
        default=0.0,
        help="If >0, sample a training subset of total hours from the full training pool "
        "and write train_<X>h.jsonl. If 0, write full train_all.jsonl. "
        "Dev/test manifests are always written if available.",
    )
    p.add_argument(
        "--train_splits",
        nargs="+",
        default=["train-clean-100", "train-clean-360", "train-other-500"],
        help="Which splits to use as training pool (default: all three train splits)",
    )
    p.add_argument(
        "--write_all_in_one",
        action="store_true",
        help="Also write all.jsonl that merges train+dev+test utterances",
    )
    args = p.parse_args()

    root = Path(args.librispeech_root).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()
    feature_root = Path(args.feature_root).expanduser().resolve() if args.feature_root else None

    # Eval manifests (fixed)
    eval_items: list[Utterance] = []
    eval_splits = ["dev-clean", "dev-other", "test-clean", "test-other"]
    for s in tqdm(eval_splits, desc="build dev/test manifests", ncols=100):
        split_dir = root / s
        if split_dir.exists():
            utts = _iter_split_utts(split_dir)
            _write_jsonl(utts, out_dir / f"{s.replace('-', '_')}.jsonl", feature_root=feature_root)
            eval_items.extend(utts)

    # Training pool
    pool: list[Utterance] = []
    for s in tqdm(args.train_splits, desc="collect train pool", ncols=100):
        split_dir = root / s
        if not split_dir.exists():
            continue
        pool.extend(_iter_split_utts(split_dir))

    if not pool:
        raise RuntimeError(
            f"No training utterances found. root={root}, train_splits={args.train_splits}"
        )

    # Deterministic shuffle
    rng = random.Random(args.seed)
    rng.shuffle(pool)

    # Rule:
    # - if --train_hours not specified (==0): write full train_all.jsonl
    # - else: sample X hours from train pool and write train_Xh.jsonl
    # Dev/test manifests are already written above.
    #
    if (args.train_hours or 0.0) > 0.0:
        train_items, _ = _take_by_hours(pool, args.train_hours)
        train_name = f"train_{int(args.train_hours)}h.jsonl"
    else:
        train_items = pool
        train_name = "train_all.jsonl"

    _write_jsonl(train_items, out_dir / train_name, feature_root=feature_root)

    if args.write_all_in_one:
        merged = train_items + eval_items
        _write_jsonl(merged, out_dir / "all.jsonl", feature_root=feature_root)

    # Print quick stats
    def hours(xs: list[Utterance]) -> float:
        return sum(u.duration for u in xs) / 3600.0

    print(f"[OK] root={root}")
    print(f"[OK] out_dir={out_dir}")
    print(f"[OK] train_pool_utts={len(pool)} hours≈{hours(pool):.2f}")
    print(f"[OK] train_manifest={train_name} utts={len(train_items)} hours≈{hours(train_items):.2f}")
    if args.write_all_in_one:
        print(f"[OK] all.jsonl utts={len(merged)} hours≈{hours(merged):.2f}")


if __name__ == "__main__":
    main()

