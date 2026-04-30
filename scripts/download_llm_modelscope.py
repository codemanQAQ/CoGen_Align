#!/usr/bin/env python3
"""
从 ModelScope 下载 Qwen2.5-7B-Instruct（或同系列）到本地目录，供 transformers 离线加载。

依赖：
  pip install "modelscope>=1.9"

下载完成后，将 ``configs/base.yaml`` 中 ``model.llm_name`` 改为打印出的本地路径即可。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--model_id",
        type=str,
        default="qwen/Qwen2.5-7B-Instruct",
        help="ModelScope 模型 id（与仓库默认 Qwen2.5-7B-Instruct 对应，一般为小写 qwen/ 前缀）",
    )
    p.add_argument(
        "--local_dir",
        type=str,
        required=True,
        help="模型保存目录（将直接写入该路径，请用大磁盘绝对路径）",
    )
    p.add_argument(
        "--revision",
        type=str,
        default="master",
        help="分支或版本，默认 master",
    )
    args = p.parse_args()

    out = Path(args.local_dir).expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)

    try:
        from modelscope import snapshot_download
    except ImportError as e:
        raise SystemExit(
            "请先安装: pip install 'modelscope>=1.9'\n"
            f"原始错误: {e}"
        ) from e

    # 新版支持 local_dir，文件直接落在指定目录；旧版仅 cache_dir（模型在 cache 子目录树中）
    try:
        model_dir = snapshot_download(
            args.model_id,
            revision=args.revision,
            local_dir=str(out),
        )
    except TypeError:
        model_dir = snapshot_download(
            args.model_id,
            revision=args.revision,
            cache_dir=str(out),
        )

    resolved = Path(model_dir).resolve()
    print(f"[OK] model_id={args.model_id}")
    print(f"[OK] path={resolved}")
    print()
    print("请在 configs/base.yaml 中设置：")
    print(f"  model:")
    print(f"    llm_name: {resolved}")


if __name__ == "__main__":
    main()
