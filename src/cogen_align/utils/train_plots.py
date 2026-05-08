"""训练过程自动生成曲线 PNG（matplotlib Agg，无显示器服务器可用）。"""

from __future__ import annotations

from pathlib import Path
from typing import Any


def _plt_agg():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def save_stage1_curves(
    out_dir: str | Path,
    *,
    step: int,
    train_rows: list[dict[str, Any]],
    val_rows: list[dict[str, Any]],
) -> Path:
    plt = _plt_agg()
    out = Path(out_dir) / "plots"
    out.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(11, 9), dpi=120)
    ax00, ax01, ax10, ax11 = axes[0, 0], axes[0, 1], axes[1, 0], axes[1, 1]

    if train_rows:
        xs = [int(r["step"]) for r in train_rows]
        ax00.plot(xs, [float(r["loss"]) for r in train_rows], color="C0", lw=1.2)
        ax00.set_title("Train loss")
        ax00.set_xlabel("step")
        ax00.grid(True, alpha=0.3)
        if any("top1_acc" in r for r in train_rows):
            ax01.plot(
                xs,
                [float(r.get("top1_acc", float("nan"))) for r in train_rows],
                color="C1",
                lw=1.2,
            )
            ax01.set_title("Train top1_acc")
            ax01.set_xlabel("step")
            ax01.grid(True, alpha=0.3)
        else:
            ax01.set_visible(False)
        if any("lr" in r for r in train_rows):
            ax10.plot(xs, [float(r["lr"]) for r in train_rows], color="C2", lw=1.2)
            ax10.set_title("Learning rate")
            ax10.set_xlabel("step")
            ax10.grid(True, alpha=0.3)
        else:
            ax10.set_visible(False)
    else:
        ax00.set_visible(False)
        ax01.set_visible(False)
        ax10.set_visible(False)

    if val_rows and any("top1_a2t" in r for r in val_rows):
        vs = [int(r["step"]) for r in val_rows if "top1_a2t" in r]
        vy = [float(r["top1_a2t"]) for r in val_rows if "top1_a2t" in r]
        ax11.plot(vs, vy, marker="o", ms=4, color="C3")
        ax11.set_title("Val top1_a2t")
        ax11.set_xlabel("step")
        ax11.grid(True, alpha=0.3)
    else:
        ax11.set_visible(False)

    fig.suptitle(f"Stage1 curves (saved at train step {step})")
    fig.tight_layout()
    stamped = out / f"curves_step_{step:08d}.png"
    latest = out / "curves_latest.png"
    fig.savefig(stamped)
    fig.savefig(latest)
    plt.close(fig)
    return stamped


def save_stage2_curves(
    out_dir: str | Path,
    *,
    step: int,
    train_rows: list[dict[str, Any]],
    val_rows: list[dict[str, Any]],
) -> Path:
    plt = _plt_agg()
    out = Path(out_dir) / "plots"
    out.mkdir(parents=True, exist_ok=True)

    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(11, 4.5), dpi=120)

    if train_rows:
        xs = [int(r["step"]) for r in train_rows]
        ax0.plot(xs, [float(r["loss"]) for r in train_rows], color="C0", lw=1.2)
        ax0.set_title("Train loss (CE)")
        ax0.set_xlabel("step")
        ax0.grid(True, alpha=0.3)
    else:
        ax0.set_visible(False)

    wer_rows = [r for r in val_rows if "wer" in r and r["wer"] is not None]
    if wer_rows:
        vs = [int(r["step"]) for r in wer_rows]
        ax1.plot(vs, [float(r["wer"]) for r in wer_rows], marker="o", ms=4, color="C1")
        ax1.set_title("Val WER")
        ax1.set_xlabel("step")
        ax1.grid(True, alpha=0.3)
    else:
        ax1.text(0.5, 0.5, "no val/wer\n(install jiwer)", ha="center", va="center", transform=ax1.transAxes)
        ax1.set_axis_off()

    fig.suptitle(f"Stage2 curves (saved at train step {step})")
    fig.tight_layout()
    stamped = out / f"curves_step_{step:08d}.png"
    latest = out / "curves_latest.png"
    fig.savefig(stamped)
    fig.savefig(latest)
    plt.close(fig)
    return stamped
