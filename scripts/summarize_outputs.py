#!/usr/bin/env python3
"""扫描 outputs/*/metrics.jsonl，重写 outputs/experiment_summary.md。"""
from __future__ import annotations

import json
import re
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
_OUT = _ROOT / "outputs"


def _rows(p: Path) -> list[dict]:
    if not p.is_file():
        return []
    out = []
    with open(p, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    out.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    return out


def _s1(rows: list[dict]) -> dict:
    meta, last_tr = {}, {}
    for r in rows:
        if r.get("split") == "meta":
            meta = r
        if r.get("split") == "train":
            last_tr = r
    return {
        "steps": meta.get("max_steps"),
        "n_train": meta.get("n_train"),
        "loss": last_tr.get("loss"),
        "top1": last_tr.get("top1_acc"),
    }


def _s2(rows: list[dict]) -> dict:
    best_wer = best_step = None
    last_val_wer = last_val_step = None
    val_final = test_clean = test_other = None
    last_tr_step = last_tr_loss = None
    meta_ms = None
    for r in rows:
        sp = r.get("split")
        if sp == "meta":
            meta_ms = r.get("max_steps")
        if sp == "train" and "loss" in r:
            last_tr_step = r.get("step")
            last_tr_loss = r.get("loss")
        if sp == "val" and "wer" in r:
            last_val_wer = r["wer"]
            last_val_step = r.get("step")
            best_wer = r.get("best_wer")
            best_step = r.get("best_step")
        if sp == "val_final" and "wer" in r:
            val_final = r["wer"]
        if sp == "final_test_clean" and "wer" in r:
            test_clean = r["wer"]
        if sp == "final_test_other" and "wer" in r:
            test_other = r["wer"]
    return {
        "meta_ms": meta_ms,
        "best_wer": best_wer,
        "best_step": best_step,
        "last_val_wer": last_val_wer,
        "last_val_step": last_val_step,
        "val_final": val_final,
        "test_clean": test_clean,
        "test_other": test_other,
        "last_tr_step": last_tr_step,
        "last_tr_loss": last_tr_loss,
    }


def _fmt(x, nd=4):
    if x is None:
        return "—"
    if isinstance(x, float):
        return f"{x:.{nd}f}"
    return str(x)


def _delta_c_minus_b(cogen_v: float | None, base_v: float | None, nd: int = 4) -> str:
    """cogen − baseline；WER 越小越好，故负值表示 cogen 更优。"""
    if cogen_v is None or base_v is None:
        return "—"
    return f"{float(cogen_v) - float(base_v):+.{nd}f}"


def _stage1_hours_from_rel(rel: str) -> str | None:
    m = re.search(r"table1_(\d+h)_e1", rel)
    if m:
        return m.group(1)
    m = re.search(r"main_(\d+h)_e1", rel)
    if m:
        return m.group(1)
    return None


def main() -> None:
    s1_rows: list[tuple[str, dict]] = []
    s1_by_h: dict[str, dict] = {}
    s2_by_h: dict[str, dict[str, dict]] = {}

    for m in sorted(_OUT.rglob("metrics.jsonl")):
        rel = m.parent.relative_to(_OUT)
        parts = rel.parts
        if len(parts) < 2:
            continue
        rows = _rows(m)
        if not rows:
            continue
        # 统一按路径识别，避免 Stage1 首行无 stage 字段、或多 seed 下 parts[0] 为 seed 名
        if len(parts) >= 2 and parts[-2] == "stage2":
            # 仅合并 flat：outputs/stage2/<run>（多 seed 为 outputs/<seed>/stage2/<run>，此处跳过）
            if len(parts) != 2 or parts[0] != "stage2":
                continue
            name = parts[-1]
            mdir = re.match(r"^(baseline|cogen)_(\d+h)(?:_seed\d+)?$", name)
            if mdir:
                if "_seed" in name:
                    continue
                if not any(r.get("stage") == 2 for r in rows[:3]):
                    continue
                h = mdir.group(2)
                cond = "baseline" if mdir.group(1) == "baseline" else "cogen"
                s2_by_h.setdefault(h, {})[cond] = _s2(rows)
        elif len(parts) >= 2 and parts[-2] == "stage1":
            rel_s = str(rel)
            s1_rows.append((rel_s, _s1(rows)))
            hk = _stage1_hours_from_rel(rel_s)
            if hk:
                s1_by_h[hk] = _s1(rows)

    lines = [
        "# `outputs/` 实验结果汇总",
        "",
        "由 `scripts/summarize_outputs.py` 根据 `metrics.jsonl` 生成；**WER 为 0–1 小数**。",
        "",
        "多 seed：``outputs/<seed>/stage2/`` 布局不在此表合并；均值见 ``outputs/multiseed_mean_summary.md``（`python scripts/aggregate_seed_metrics.py`）。",
        "",
        "## Stage1",
        "",
        "| 目录 | train 步数 | n_train | 末步 loss | 末步 top1_acc |",
        "|------|------------|---------|-----------|---------------|",
    ]
    for rel, d in sorted(s1_rows):
        lines.append(
            f"| `{rel}` | {_fmt(d['steps'], 0)} | {_fmt(d['n_train'], 0)} | "
            f"{_fmt(d['loss'])} | {_fmt(d['top1'])} |"
        )

    hours_order = ["50h", "100h", "200h", "300h", "400h", "500h"]
    lines += [
        "",
        "## Stage2",
        "",
        "| 数据量 | 条件 | best WER | best@step | val_final | test_clean | test_other | 末次 train step |",
        "|--------|------|------------|-------------|-----------|--------------|--------------|-----------------|",
    ]
    for h in hours_order:
        if h not in s2_by_h:
            continue
        for cond in ("baseline", "cogen"):
            d = s2_by_h[h].get(cond)
            if not d:
                continue
            lines.append(
                f"| {h} | {cond} | {_fmt(d['best_wer'])} | {_fmt(d['best_step'], 0)} | "
                f"{_fmt(d['val_final'])} | {_fmt(d['test_clean'])} | {_fmt(d['test_other'])} | "
                f"{_fmt(d['last_tr_step'], 0)} |"
            )

    lines += [
        "",
        "## Stage2：相同数据量 baseline vs cogen",
        "",
        "Δ 列为 **cogen − baseline**（WER 越小越好，**负值表示 cogen 更优**）。",
        "",
        "| 数据量 | S1 步 | S1 loss | S1 top1 | BL best | CG best | Δbest | BL val_f | CG val_f | Δval_f | BL test_c | CG test_c | Δtc | BL test_o | CG test_o | Δto |",
        "|--------|-------|---------|---------|----------|---------|-------|----------|----------|-------|------------|-----------|-----|-----------|-----------|-----|",
    ]
    for h in hours_order:
        if h not in s2_by_h:
            continue
        b = s2_by_h[h].get("baseline") or {}
        c = s2_by_h[h].get("cogen") or {}
        s1 = s1_by_h.get(h) or {}
        lines.append(
            f"| {h} | {_fmt(s1.get('steps'), 0)} | {_fmt(s1.get('loss'))} | {_fmt(s1.get('top1'))} | "
            f"{_fmt(b.get('best_wer'))} | {_fmt(c.get('best_wer'))} | "
            f"{_delta_c_minus_b(c.get('best_wer'), b.get('best_wer'))} | "
            f"{_fmt(b.get('val_final'))} | {_fmt(c.get('val_final'))} | "
            f"{_delta_c_minus_b(c.get('val_final'), b.get('val_final'))} | "
            f"{_fmt(b.get('test_clean'))} | {_fmt(c.get('test_clean'))} | "
            f"{_delta_c_minus_b(c.get('test_clean'), b.get('test_clean'))} | "
            f"{_fmt(b.get('test_other'))} | {_fmt(c.get('test_other'))} | "
            f"{_delta_c_minus_b(c.get('test_other'), b.get('test_other'))} |"
        )

    out_path = _OUT / "experiment_summary.md"
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
