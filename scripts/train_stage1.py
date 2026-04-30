#!/usr/bin/env python3
"""Stage 1 contrastive warmup — 支持 torchrun 多卡 DDP 与实验流程记录。"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from datetime import datetime, timezone
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
_SRC = _ROOT / "src"
if _SRC.is_dir() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from transformers import AutoModelForCausalLM, AutoTokenizer

from cogen_align.data.collator import collate_stage1
from cogen_align.data.dataset import SpeechTextDataset
from cogen_align.models.losses import SymmetricInfoNCE
from cogen_align.models.projector import Projector, SpeechTextEncoder
from cogen_align.utils.config import load_config
from cogen_align.utils.experiment_record import write_experiment_record


def _loss_out_scalars(out: dict) -> dict[str, float]:
    """InfoNCE 返回的张量标量 → float，便于写 jsonl / wandb。"""
    scalars: dict[str, float] = {}
    for k, v in out.items():
        if isinstance(v, torch.Tensor) and v.numel() == 1:
            scalars[k] = float(v.detach().cpu())
        elif isinstance(v, (float, int)) and not isinstance(v, bool):
            scalars[k] = float(v)
    return scalars


def _append_metrics_jsonl(out_dir: str, record: dict) -> None:
    """rank0 追加一行 JSONL，供离线绑图（与 W&B 无关）。"""
    row = {
        **record,
        "ts_utc": datetime.now(timezone.utc).isoformat(),
    }
    path = Path(out_dir) / "metrics.jsonl"
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")
        f.flush()


def _dist_setup() -> tuple[int, int, int]:
    """返回 (rank, world_size, local_rank)。world_size>1 时初始化 NCCL。"""
    ws = int(os.environ.get("WORLD_SIZE", "1"))
    lr = int(os.environ.get("LOCAL_RANK", "0"))
    rk = int(os.environ.get("RANK", "0"))
    if ws > 1:
        if not torch.cuda.is_available():
            raise RuntimeError("DDP 需要 CUDA")
        torch.cuda.set_device(lr)
        dist.init_process_group(backend="nccl")
    return rk, ws, lr


def _dist_teardown() -> None:
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def _unwrap(model: torch.nn.Module | DDP) -> torch.nn.Module:
    return model.module if isinstance(model, DDP) else model


def evaluate_retrieval(model: SpeechTextEncoder, val_loader: DataLoader, device: torch.device) -> dict:
    model.eval()
    all_z_a, all_z_t = [], []
    with torch.no_grad():
        for batch in val_loader:
            audio_feat = batch["audio_feat"].to(device, dtype=torch.bfloat16)
            audio_mask = batch["audio_mask"].to(device)
            text_ids = batch["text_ids"].to(device)
            text_mask = batch["text_mask"].to(device)
            z_a = model.encode_audio(audio_feat, audio_mask)
            z_t = model.encode_text(text_ids, text_mask)
            all_z_a.append(z_a.float())
            all_z_t.append(z_t.float())
    z_a = torch.cat(all_z_a)
    z_t = torch.cat(all_z_t)
    z_a = torch.nn.functional.normalize(z_a, dim=-1)
    z_t = torch.nn.functional.normalize(z_t, dim=-1)
    sim = z_a @ z_t.T
    n = sim.size(0)
    labels = torch.arange(n, device=sim.device)
    top10_a2t = sim.topk(10, dim=-1).indices
    top1_a2t = (top10_a2t[:, 0] == labels).float().mean().item()
    top10_a2t_acc = (top10_a2t == labels.unsqueeze(-1)).any(dim=-1).float().mean().item()
    top10_t2a = sim.T.topk(10, dim=-1).indices
    top1_t2a = (top10_t2a[:, 0] == labels).float().mean().item()
    top10_t2a_acc = (top10_t2a == labels.unsqueeze(-1)).any(dim=-1).float().mean().item()
    idx1 = torch.randint(0, n, (min(5000, n * n),), device=sim.device)
    idx2 = torch.randint(0, n, (min(5000, n * n),), device=sim.device)
    aniso = (z_a[idx1] * z_a[idx2]).sum(-1).mean().item()
    model.train()
    return {
        "top1_a2t": top1_a2t,
        "top10_a2t": top10_a2t_acc,
        "top1_t2a": top1_t2a,
        "top10_t2a": top10_t2a_acc,
        "anisotropy": aniso,
    }


def synthetic_loader(batch_size: int, device: torch.device, steps: int):
    for _ in range(steps):
        t = 50
        d = 1280
        audio = torch.randn(batch_size, t, d, device=device, dtype=torch.bfloat16)
        audio_mask = torch.zeros(batch_size, t, dtype=torch.bool, device=device)
        text_ids = torch.randint(4, 1000, (batch_size, 32), device=device)
        text_mask = torch.zeros(batch_size, 32, dtype=torch.bool, device=device)
        yield {
            "audio_feat": audio,
            "audio_mask": audio_mask,
            "text_ids": text_ids,
            "text_mask": text_mask,
        }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="忽略 manifest；单卡随机张量冒烟（多卡时请 nproc_per_node=1）。",
    )
    parser.add_argument("--smoke-steps", type=int, default=5)
    parser.add_argument(
        "--smoke-embed-dim",
        type=int,
        default=64,
        help="--smoke 时文本嵌入维度（不下载 HuggingFace）。",
    )
    args = parser.parse_args()

    cfg_path = Path(args.config)
    if not cfg_path.is_file():
        cfg_path = _ROOT / args.config
    cfg = load_config(cfg_path)

    world_size_pre = int(os.environ.get("WORLD_SIZE", "1"))
    if args.smoke and world_size_pre > 1:
        raise SystemExit("--smoke 与多进程冲突：请使用 torchrun --nproc_per_node=1")

    seed = int(cfg.get("project", {}).get("seed", 42))
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    rank, world_size, local_rank = _dist_setup()
    is_master = rank == 0

    if torch.cuda.is_available():
        device = torch.device(f"cuda:{local_rank}")
    else:
        device = torch.device("cpu")
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32

    if args.smoke:
        hidden = args.smoke_embed_dim
        llm_embed = torch.nn.Embedding(4096, hidden, device=device, dtype=dtype)
        for p in llm_embed.parameters():
            p.requires_grad = False
    else:
        if is_master:
            print(f"[stage1] DDP world_size={world_size} rank={rank} local_rank={local_rank}")
        llm_name = cfg["model"]["llm_name"]
        tokenizer = AutoTokenizer.from_pretrained(llm_name)
        llm = AutoModelForCausalLM.from_pretrained(llm_name, torch_dtype=dtype)
        llm_embed = llm.get_input_embeddings()
        for p in llm_embed.parameters():
            p.requires_grad = False
        hidden = int(getattr(llm.config, "hidden_size", cfg["model"]["llm_hidden_size"]))
        del llm

    projector = Projector(
        in_dim=cfg["model"]["whisper_feature_dim"],
        out_dim=hidden,
        hidden_dim=min(cfg["model"]["projector"]["hidden_dim"], hidden * 2),
        subsample=cfg["model"]["projector"]["subsample"],
    )
    encoder = SpeechTextEncoder(
        projector,
        llm_embed,
        out_dim=hidden,
    ).to(device, dtype=dtype)

    if world_size > 1:
        encoder = DDP(
            encoder,
            device_ids=[local_rank] if device.type == "cuda" else None,
            output_device=local_rank if device.type == "cuda" else None,
            find_unused_parameters=False,
        )

    loss_fn = SymmetricInfoNCE(temperature=cfg["training"].get("temperature", 0.07)).to(
        device
    )

    wb = None
    wandb_mod = None
    if is_master and os.environ.get("WANDB_DISABLED", "") != "1":
        try:
            import wandb as wandb_mod  # type: ignore
        except Exception:
            wandb_mod = None
        if wandb_mod is not None and cfg.get("wandb", {}).get("project"):
            try:
                wb = wandb_mod.init(
                    project=cfg["wandb"]["project"],
                    entity=cfg["wandb"].get("entity") or None,
                    name=cfg.get("run_name", "stage1"),
                    config=cfg,
                    tags=["stage1", cfg.get("experiment_type", "main"), f"ddp{world_size}"],
                    notes=cfg.get("wandb", {}).get("notes"),
                )
            except Exception:
                wb = None

    core = _unwrap(encoder)
    trainable = [p for p in core.parameters() if p.requires_grad]
    opt = AdamW(
        trainable,
        lr=cfg["training"].get("lr", 1e-4),
        weight_decay=cfg["training"].get("weight_decay", 0.01),
    )

    out_dir = cfg.get("output_dir", "outputs/stage1_default")
    os.makedirs(out_dir, exist_ok=True)

    if is_master and not args.smoke:
        write_experiment_record(
            out_dir,
            cfg=cfg,
            config_path=cfg_path,
            extra={"stage": "1", "script": "train_stage1.py"},
            repo_root=_ROOT,
        )
        print(f"[stage1] experiment_record -> {Path(out_dir) / 'experiment_record.yaml'}")

    if args.smoke:
        sched = CosineAnnealingLR(opt, T_max=max(1, args.smoke_steps))
        encoder.train()
        step = 0
        for batch in synthetic_loader(
            min(4, cfg["training"].get("batch_size_per_gpu", 4)),
            device,
            args.smoke_steps,
        ):
            z_a = encoder.encode_audio(batch["audio_feat"], batch["audio_mask"])
            z_t = encoder.encode_text(batch["text_ids"], batch["text_mask"])
            out = loss_fn(z_a, z_t)
            loss = out["loss"]
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                trainable, cfg["training"].get("gradient_clip", 1.0)
            )
            opt.step()
            sched.step()
            print(f"smoke step {step} loss={loss.item():.4f}")
            step += 1
        if wb is not None:
            wb.finish()
        _dist_teardown()
        return

    train_ds = SpeechTextDataset(
        _ROOT / cfg["data"]["train_manifest"],
        tokenizer,
        load_hard_negs=cfg["training"].get("use_hard_neg", False),
    )
    val_ds = SpeechTextDataset(_ROOT / cfg["data"]["val_manifest"], tokenizer)
    bs = cfg["training"].get("batch_size_per_gpu", 4)

    train_sampler: DistributedSampler | None = None
    if world_size > 1:
        train_sampler = DistributedSampler(
            train_ds,
            num_replicas=world_size,
            rank=rank,
            seed=seed,
            shuffle=True,
            drop_last=False,
        )

    train_loader = DataLoader(
        train_ds,
        batch_size=bs,
        shuffle=train_sampler is None,
        sampler=train_sampler,
        collate_fn=collate_stage1,
        num_workers=2,
        pin_memory=device.type == "cuda",
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=64,
        shuffle=False,
        collate_fn=collate_stage1,
        num_workers=2,
        pin_memory=device.type == "cuda",
    )

    steps_per_epoch = max(1, len(train_loader))
    if cfg["training"].get("max_epochs") is not None:
        max_steps = int(steps_per_epoch * int(cfg["training"]["max_epochs"]))
    else:
        max_steps = int(cfg["training"].get("max_steps", 1000))
    sched = CosineAnnealingLR(opt, T_max=max_steps)
    if is_master:
        print(
            f"[train] n_train={len(train_ds)} batch_size={bs} world_size={world_size} "
            f"steps_per_epoch={steps_per_epoch} max_steps={max_steps}"
        )
        _append_metrics_jsonl(
            out_dir,
            {
                "split": "meta",
                "n_train": len(train_ds),
                "n_val": len(val_ds),
                "batch_size_per_gpu": bs,
                "world_size": world_size,
                "steps_per_epoch": steps_per_epoch,
                "max_steps": max_steps,
            },
        )

    encoder.train()
    global_step = 0
    epoch_idx = 0
    while global_step < max_steps:
        if train_sampler is not None:
            train_sampler.set_epoch(epoch_idx)
        for batch in train_loader:
            audio_feat = batch["audio_feat"].to(device, dtype=dtype)
            audio_mask = batch["audio_mask"].to(device)
            text_ids = batch["text_ids"].to(device)
            text_mask = batch["text_mask"].to(device)

            z_a = encoder.encode_audio(audio_feat, audio_mask)
            z_t = encoder.encode_text(text_ids, text_mask)
            out = loss_fn(z_a, z_t)
            loss = out["loss"]

            opt.zero_grad()
            loss.backward()
            gn = torch.nn.utils.clip_grad_norm_(
                trainable, cfg["training"].get("gradient_clip", 1.0)
            )
            opt.step()
            sched.step()

            if is_master and global_step % cfg["training"].get("log_every", 50) == 0:
                loss_scalars = _loss_out_scalars(out)
                msg = (
                    f"step {global_step} loss={loss.item():.4f} "
                    f"top1={out['top1_acc'].item():.3f} grad_norm={gn:.3f}"
                )
                print(msg)
                train_row: dict = {
                    "split": "train",
                    "step": global_step,
                    "epoch": epoch_idx,
                    "grad_norm": float(gn),
                    "lr": float(sched.get_last_lr()[0]),
                }
                train_row.update(loss_scalars)
                _append_metrics_jsonl(out_dir, train_row)
                if wb is not None and wandb_mod is not None:
                    wb_train = {
                        "train/grad_norm": float(gn),
                        "train/lr": float(sched.get_last_lr()[0]),
                        "train/epoch": float(epoch_idx),
                    }
                    for k, v in loss_scalars.items():
                        wb_train[f"train/{k}"] = v
                    wandb_mod.log(wb_train, step=global_step)

            ev = cfg["training"].get("eval_every", 2000)
            if ev and global_step > 0 and global_step % ev == 0:
                if world_size > 1:
                    dist.barrier()
                if is_master:
                    metrics = evaluate_retrieval(_unwrap(encoder), val_loader, device)
                    print("val", metrics)
                    row = {"split": "val", "step": global_step, "epoch": epoch_idx}
                    row.update({k: float(v) for k, v in metrics.items()})
                    _append_metrics_jsonl(out_dir, row)
                    if wb is not None and wandb_mod is not None:
                        wb_val = {f"val/{k}": v for k, v in metrics.items()}
                        wb_val["val/epoch"] = float(epoch_idx)
                        wandb_mod.log(wb_val, step=global_step)
                if world_size > 1:
                    dist.barrier()

            sv = cfg["training"].get("save_every", 5000)
            if sv and global_step > 0 and global_step % sv == 0:
                if world_size > 1:
                    dist.barrier()
                if is_master:
                    ckpt_path = os.path.join(out_dir, f"ckpt_{global_step}.pt")
                    core_m = _unwrap(encoder)
                    torch.save(
                        {
                            "projector": core_m.projector.state_dict(),
                            "audio_pool": core_m.audio_pool.state_dict(),
                            "text_pool": core_m.text_pool.state_dict(),
                            "step": global_step,
                            "config": cfg,
                        },
                        ckpt_path,
                    )
                    print("saved", ckpt_path)
                if world_size > 1:
                    dist.barrier()

            global_step += 1
            if global_step >= max_steps:
                break
        epoch_idx += 1
        if global_step >= max_steps:
            break

    if world_size > 1:
        dist.barrier()
    if is_master:
        last_path = os.path.join(out_dir, "ckpt_last.pt")
        core_m = _unwrap(encoder)
        torch.save(
            {
                "projector": core_m.projector.state_dict(),
                "audio_pool": core_m.audio_pool.state_dict(),
                "text_pool": core_m.text_pool.state_dict(),
                "step": global_step,
                "config": cfg,
            },
            last_path,
        )
        print("saved", last_path)

    if wb is not None:
        wb.finish()
    _dist_teardown()


if __name__ == "__main__":
    main()
