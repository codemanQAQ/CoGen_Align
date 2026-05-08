#!/usr/bin/env python3
"""多 seed 下按数据量 × 条件聚合 WER，输出均值 ± 标准差（论文表）。

支持两种目录布局（``--layout auto`` 默认自动识别）：

- **nested**：``outputs/<seed>/stage2/baseline_50h/``（推荐，与 ``--seed-output-root`` 一致）
- **flat**：``outputs/stage2/baseline_50h_seed42/``（旧版；可用 ``--multiseed-only`` 只统计带 ``_seed`` 的目录）
"""
from __future__ import annotations

import argparse
import json
import math
import re
import sys
from collections import defaultdict
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]


def _rows(p: Path) -> list[dict]:
    if not p.is_file():
        return []
    out: list[dict] = []
    with open(p, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    out.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    return out


def _wer_for_split(rows: list[dict], split: str) -> float | None:
    last: float | None = None
    for r in rows:
        if r.get("split") == split and "wer" in r:
            last = float(r["wer"])
    return last


_RE_DIR = re.compile(r"^(baseline|cogen)_(\d+)h(?:_seed(\d+))?$")


def _parse_stage2_dirname(
    name: str, *, multiseed_only: bool
) -> tuple[str, int, str | None] | None:
    m = _RE_DIR.match(name)
    if not m:
        return None
    cond, h_s, seed = m.group(1), int(m.group(2)), m.group(3)
    if multiseed_only and seed is None:
        return None
    return cond, h_s, seed


def _mean_std(xs: list[float]) -> tuple[float, float]:
    n = len(xs)
    if n == 0:
        return float("nan"), float("nan")
    mean = sum(xs) / n
    if n == 1:
        return mean, 0.0
    var = sum((x - mean) ** 2 for x in xs) / (n - 1)
    return mean, math.sqrt(var)


def _discover_nested_runs(outputs: Path) -> list[tuple[str, Path]]:
    """(seed 文件夹名, run 目录)。仅当存在 ``outputs/<x>/stage2`` 且 x != stage2 时视为 nested。"""
    runs: list[tuple[str, Path]] = []
    for child in sorted(outputs.iterdir()):
        if not child.is_dir() or child.name == "stage2":
            continue
        s2 = child / "stage2"
        if not s2.is_dir():
            continue
        for run in sorted(s2.iterdir()):
            if run.is_dir() and (run / "metrics.jsonl").is_file():
                runs.append((child.name, run))
    return runs


def _discover_flat_runs(outputs: Path) -> list[tuple[str | None, Path]]:
    s2 = outputs / "stage2"
    if not s2.is_dir():
        return []
    out: list[tuple[str | None, Path]] = []
    for run in sorted(s2.iterdir()):
        if run.is_dir() and (run / "metrics.jsonl").is_file():
            out.append((None, run))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--outputs-root",
        type=Path,
        default=_ROOT / "outputs",
        help="outputs 根目录（默认仓库 outputs/）。",
    )
    ap.add_argument(
        "--layout",
        choices=("auto", "nested", "flat"),
        default="auto",
        help="auto：若存在 outputs/*/stage2 则用 nested，否则 flat。",
    )
    ap.add_argument(
        "--split",
        type=str,
        default="val_final",
        help="metrics.jsonl 中的 split 字段。",
    )
    ap.add_argument(
        "--out",
        type=Path,
        default=_ROOT / "outputs" / "multiseed_mean_summary.md",
        help="均值汇总 Markdown（默认 outputs/multiseed_mean_summary.md）。",
    )
    ap.add_argument("--hours", type=str, default="50,100,200,300,400,500", help="表格行（逗号分隔）。")
    ap.add_argument(
        "--multiseed-only",
        action="store_true",
        help="仅 flat 布局有效：只统计目录名含 _seed<数字> 的运行。",
    )
    args = ap.parse_args()

    outputs = args.outputs_root.resolve()
    if not outputs.is_dir():
        raise SystemExit(f"目录不存在: {outputs}")

    nested = _discover_nested_runs(outputs)
    flat = _discover_flat_runs(outputs)

    if args.layout == "nested":
        runs = nested
        layout_note = "nested（outputs/<seed>/stage2/…）"
    elif args.layout == "flat":
        runs = flat
        layout_note = "flat（outputs/stage2/…）"
    else:
        if nested:
            runs = nested
            layout_note = "auto → nested"
        else:
            runs = flat
            layout_note = "auto → flat"

    if args.layout == "auto" and nested and flat:
        print(
            "[aggregate_seed_metrics] 警告: 同时存在 outputs/<seed>/stage2/ 与 outputs/stage2/；"
            "auto 已只采用 nested，flat 下旧结果未纳入表内。仅看旧实验请加 --layout flat。",
            file=sys.stderr,
        )

    effective_nested = args.layout == "nested" or (args.layout == "auto" and bool(nested))
    use_multiseed_suffix_filter = args.multiseed_only and not effective_nested

    grouped: dict[tuple[str, int], list[float]] = defaultdict(list)

    for seed_key, run_dir in runs:
        parsed = _parse_stage2_dirname(run_dir.name, multiseed_only=use_multiseed_suffix_filter)
        if parsed is None:
            continue
        cond, h, _name_seed = parsed
        rows = _rows(run_dir / "metrics.jsonl")
        wer = _wer_for_split(rows, args.split)
        if wer is not None:
            grouped[(cond, h)].append(wer)

    def _seed_sort_key(s: str) -> tuple[int, str]:
        return (int(s), s) if s.isdigit() else (10**9, s)

    seed_keys = sorted({sk for sk, _ in runs if sk is not None}, key=_seed_sort_key)

    hours = [int(x.strip()) for x in args.hours.split(",") if x.strip()]
    lines = [
        "# Table1 多 seed 聚合（均值 ± 标准差）",
        "",
        f"- split: `{args.split}`",
        f"- 布局: {layout_note}",
        f"- 扫描根: `{outputs}`",
        f"- 检出 seed 子目录: {', '.join(seed_keys) if seed_keys else '—（flat 或无子目录名）'}",
        "",
        "| 数据量 (h) | Baseline WER (mean±std) | CoGen WER (mean±std) | n (runs) |",
        "| ---: | --- | --- | ---: |",
    ]

    def fmt_cell(vals: list[float]) -> str:
        if not vals:
            return "—"
        m, s = _mean_std(vals)
        if math.isnan(m):
            return "—"
        if len(vals) == 1:
            return f"{m:.4f}"
        return f"{m:.4f} ± {s:.4f}"

    for h in hours:
        b = grouped.get(("baseline", h), [])
        c = grouped.get(("cogen", h), [])
        if len(b) != len(c) and (b or c):
            n_note = f"{len(b)} baseline / {len(c)} cogen"
        else:
            n_note = str(len(b)) if b else "0"

        lines.append(
            f"| {h} | {fmt_cell(b)} | {fmt_cell(c)} | {n_note} |",
        )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"written -> {args.out}")


if __name__ == "__main__":
    main()
