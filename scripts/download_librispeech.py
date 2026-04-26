# scripts/download_librispeech_modelscope.py
"""
从 ModelScope 下载 openslr/librispeech_asr，并解码成标准 LibriSpeech 目录结构。

仓库格式（HF 风格）:
  - configs: clean, other
  - splits: train.100, train.360, train.500, validation, test
  - 数据格式: parquet (audio 字段是 16kHz int16/float32 numpy array)

输出（标准 LibriSpeech 格式）:
  /data/librispeech/LibriSpeech/
    ├── train-clean-100/
    │   └── <speaker>/<chapter>/
    │       ├── <utt_id>.flac
    │       └── <speaker>-<chapter>.trans.txt
    ├── train-clean-360/
    ├── train-other-500/
    ├── dev-clean/
    ├── dev-other/
    ├── test-clean/
    └── test-other/

Usage:
    # 先小后大，第一次只下 100h + dev/test 试水
    python scripts/download_librispeech_modelscope.py \
        --output_dir /data/librispeech \
        --splits clean:train.100 clean:validation clean:test
    
    # 后续下完整
    python scripts/download_librispeech_modelscope.py \
        --output_dir /data/librispeech \
        --splits clean:train.360 other:train.500 other:validation other:test
"""

import os
import argparse
from pathlib import Path
import soundfile as sf
import numpy as np
from tqdm import tqdm
import tempfile
from typing import Iterable
import io


# (config, split) → 本地标准目录名
SPLIT_MAP = {
    ("clean", "train.100"):  "train-clean-100",
    ("clean", "train.360"):  "train-clean-360",
    ("other", "train.500"):  "train-other-500",
    ("clean", "validation"): "dev-clean",
    ("other", "validation"): "dev-other",
    ("clean", "test"):       "test-clean",
    ("other", "test"):       "test-other",
}


def _maybe_set_env(var: str, value: str | None):
    if value:
        os.environ[var] = value


def _configure_cache_dirs(cache_dir: Path | None, tmp_dir: Path | None):
    """
    避免 HF datasets / pyarrow 默认写到容器盘（常见会触发 ENOSPC）。
    这些变量对 modelscope -> datasets 路径也有效。
    """
    if cache_dir is not None:
        cache_dir.mkdir(parents=True, exist_ok=True)
        _maybe_set_env("HF_HOME", str(cache_dir / "hf_home"))
        _maybe_set_env("HF_DATASETS_CACHE", str(cache_dir / "hf_datasets"))
        _maybe_set_env("TRANSFORMERS_CACHE", str(cache_dir / "hf_models"))
        _maybe_set_env("MODELSCOPE_CACHE", str(cache_dir / "modelscope"))
    if tmp_dir is not None:
        tmp_dir.mkdir(parents=True, exist_ok=True)
        _maybe_set_env("TMPDIR", str(tmp_dir))
        tempfile.tempdir = str(tmp_dir)


def _resolve_audio_path(p: str, *, search_roots: Iterable[Path]) -> Path:
    """
    datasets Audio(decode=False) 有时会返回相对路径/仅文件名（例如 '6930-...flac'），
    需要在缓存目录里把真实文件找出来。
    """
    cand = Path(p)
    if cand.is_absolute() and cand.exists():
        return cand

    # 相对路径：先按 cwd 试一次
    if not cand.is_absolute():
        wd = Path.cwd() / cand
        if wd.exists():
            return wd

    fname = cand.name
    for root in search_roots:
        root = Path(root)
        if not root.exists():
            continue
        # 先按“相对路径”拼一次
        direct = root / cand
        if direct.exists():
            return direct
        # 兜底：只按文件名查找一次
        try:
            hit = next(root.rglob(fname))
            if hit.exists():
                return hit
        except StopIteration:
            pass

    raise FileNotFoundError(
        f"Could not resolve audio path: {p!r}. "
        f"Tried cwd and roots: {[str(r) for r in search_roots]}"
    )


def download_one_split(
    config: str,
    split: str,
    output_root: Path,
    *,
    streaming: bool,
    cache_dir: Path | None,
    tmp_dir: Path | None,
    decode_audio: bool,
):
    """下载一个 (config, split) 并解码成标准目录结构。"""
    key = (config, split)
    if key not in SPLIT_MAP:
        print(f"  [Skip] Unknown ({config}, {split})")
        return
    
    local_dir_name = SPLIT_MAP[key]
    local_dir = output_root / "LibriSpeech" / local_dir_name
    
    # 已经有就跳过（但如果缺 trans.txt，则允许“修复模式”补写转写文件）
    if local_dir.exists() and any(local_dir.rglob("*.flac")):
        existing_flac = len(list(local_dir.rglob("*.flac")))
        existing_trans = len(list(local_dir.rglob("*.trans.txt")))
        if existing_trans > 0:
            print(
                f"  [Skip] {local_dir_name} already has {existing_flac} flac files "
                f"and {existing_trans} trans.txt files"
            )
            return
        print(
            f"  [Repair] {local_dir_name} has {existing_flac} flac files but 0 trans.txt; "
            "will regenerate transcripts (audio will not be re-written)."
        )
    
    local_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n=== Loading {config}/{split} → {local_dir_name} ===")
    
    # 用 ModelScope SDK 加载 dataset
    from modelscope.msdatasets import MsDataset

    _configure_cache_dirs(cache_dir=cache_dir, tmp_dir=tmp_dir)

    # 关键点：优先使用 streaming，避免触发 datasets 的 download_and_prepare，
    # 否则可能会“顺带”prepare 同 config 下其它 split（例如 train.360），导致磁盘爆炸。
    load_kwargs = dict(
        subset_name=config,
        split=split,
    )
    ds = None
    if streaming:
        try:
            ds = MsDataset.load("openslr/librispeech_asr", streaming=True, **load_kwargs)
            print("  Using streaming=True (avoid Arrow cache generation)")
        except TypeError:
            ds = None
    if ds is None:
        ds = MsDataset.load("openslr/librispeech_asr", **load_kwargs)
        print("  Using non-streaming load (may generate Arrow cache)")

    # datasets 的 Audio feature 可能会走 torchcodec（依赖 ffmpeg/动态库）。
    # 默认关闭解码，直接拿到缓存文件路径后用 soundfile 读取，最稳。
    if not decode_audio:
        try:
            from datasets import Audio  # type: ignore

            if hasattr(ds, "cast_column"):
                try:
                    ds = ds.cast_column("audio", Audio(decode=False))
                    print("  Audio decode: OFF (read audio['path'] via soundfile)")
                except Exception:
                    pass
        except Exception:
            pass
    else:
        print("  Audio decode: ON (will rely on datasets' audio decoder)")

    try:
        n = len(ds)  # may fail for streaming
        print(f"  Loaded {n} samples")
        total = n
    except Exception:
        print("  Loaded streaming dataset (unknown length)")
        total = None
    
    # 累积每个 chapter 的 trans.txt 内容
    chapter_lines = {}  # {(speaker, chapter): [(utt_id, text), ...]}
    
    for sample in tqdm(ds, desc=f"  Decoding {local_dir_name}", ncols=80, total=total):
        utt_id = sample["id"]                  # e.g. "103-1240-0000"
        speaker_id = str(sample["speaker_id"])
        chapter_id = str(sample["chapter_id"])
        text = sample["text"]
        audio = sample["audio"]                # decode=False: {"path": "...", ...} ; decode=True: {"array": ..., "sampling_rate": ...}
        
        chapter_dir = local_dir / speaker_id / chapter_id
        chapter_dir.mkdir(parents=True, exist_ok=True)
        
        # 写 .flac
        flac_path = chapter_dir / f"{utt_id}.flac"
        if not flac_path.exists():
            audio_array = None
            sr = None
            if isinstance(audio, dict):
                # decode=True path
                if "array" in audio and audio.get("array") is not None:
                    audio_array = audio["array"]
                    sr = audio.get("sampling_rate", 16000)
                # decode=False path
                elif "path" in audio and audio.get("path"):
                    # datasets Audio(decode=False) 可能提供 bytes（无需依赖真实文件落盘）
                    if audio.get("bytes"):
                        bio = io.BytesIO(audio["bytes"])
                        audio_array, sr = sf.read(bio, dtype="float32")
                    else:
                        raw_p = str(audio["path"])
                        roots = [
                            Path(os.environ.get("HF_DATASETS_CACHE", "")),
                            Path(os.environ.get("MODELSCOPE_CACHE", "")),
                            cache_dir or Path(""),
                            tmp_dir or Path(""),
                        ]
                        ap = _resolve_audio_path(raw_p, search_roots=roots)
                        audio_array, sr = sf.read(str(ap), dtype="float32")
            if audio_array is None:
                raise RuntimeError(
                    "audio decoding failed: expected audio['array'] or audio['path']"
                )
            if sr is None:
                sr = 16000
            # soundfile 对 float32 / int16 更稳妥
            if getattr(audio_array, "dtype", None) not in (np.int16, np.float32, np.float64):
                audio_array = np.asarray(audio_array, dtype=np.float32)
            sf.write(str(flac_path), audio_array, sr, format="FLAC")
        
        # 累积 trans.txt
        chapter_lines.setdefault((speaker_id, chapter_id), []).append((utt_id, text))
    
    # 写 trans.txt（每个 chapter 一个）
    print(f"  Writing trans.txt files...")
    for (speaker_id, chapter_id), lines in chapter_lines.items():
        chapter_dir = local_dir / speaker_id / chapter_id
        trans_path = chapter_dir / f"{speaker_id}-{chapter_id}.trans.txt"
        # 按 utt_id 排序，和官方 LibriSpeech 一致
        lines.sort(key=lambda x: x[0])
        with open(trans_path, "w") as f:
            for utt_id, text in lines:
                f.write(f"{utt_id} {text}\n")
    
    print(f"  ✓ Done: {local_dir_name}")


def parse_split_arg(s: str):
    """解析 'config:split' 格式参数。"""
    if ":" not in s:
        raise ValueError(f"Invalid split format: {s}, expected 'config:split'")
    config, split = s.split(":", 1)
    return (config, split)


def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument(
        "--output_dir", type=str, default="/data/speech/tts/intern/mqb/wavs",
        help="LibriSpeech 输出根目录"
    )
    parser.add_argument(
        "--no_streaming",
        action="store_true",
        help="禁用 streaming（不推荐；可能触发 Arrow cache 生成并写爆磁盘）",
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default=None,
        help="把 HF/MsDataset 缓存写到该目录（强烈建议设到 /data/... 大盘）",
    )
    parser.add_argument(
        "--tmp_dir",
        type=str,
        default=None,
        help="把临时文件写到该目录（强烈建议设到 /data/... 大盘）",
    )
    parser.add_argument(
        "--decode_audio",
        action="store_true",
        help="让 datasets 解码 audio（可能依赖 torchcodec/ffmpeg）；默认关闭更稳",
    )
    parser.add_argument(
        "--splits", nargs="+", required=True,
        help="要下载的 (config, split) 列表，格式 'config:split'，例如:\n"
             "  clean:train.100 clean:train.360 clean:validation clean:test\n"
             "  other:train.500 other:validation other:test"
    )
    args = parser.parse_args()
    
    output_root = Path(args.output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    cache_dir = Path(args.cache_dir) if args.cache_dir else None
    tmp_dir = Path(args.tmp_dir) if args.tmp_dir else None
    
    print(f"=== Output directory: {output_root} ===")
    print(f"Will download: {args.splits}\n")
    if not args.no_streaming:
        print("Streaming: ON (recommended)\n")
    else:
        print("Streaming: OFF\n")
    
    for s in args.splits:
        config, split = parse_split_arg(s)
        try:
            download_one_split(
                config,
                split,
                output_root,
                streaming=not args.no_streaming,
                cache_dir=cache_dir,
                tmp_dir=tmp_dir,
                decode_audio=args.decode_audio,
            )
        except Exception as e:
            print(f"  [ERROR] Failed to download {config}/{split}: {e}")
            import traceback; traceback.print_exc()
    
    print(f"\n=== All done. LibriSpeech is at {output_root / 'LibriSpeech'} ===")


if __name__ == "__main__":
    main()