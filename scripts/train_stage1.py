#!/usr/bin/env python3
"""Stage 1 contrastive warmup (single-process scaffold; extend to DDP per handbook)."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
_SRC = _ROOT / "src"
if _SRC.is_dir() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer

from cogen_align.data.collator import collate_stage1
from cogen_align.data.dataset import SpeechTextDataset
from cogen_align.models.losses import SymmetricInfoNCE
from cogen_align.models.projector import Projector, SpeechTextEncoder
from cogen_align.utils.config import load_config


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
    """Yield random batches mimicking Whisper features + token ids."""
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
        help="Ignore manifests; run a few steps on random tensors (shape sanity).",
    )
    parser.add_argument("--smoke-steps", type=int, default=5)
    parser.add_argument(
        "--smoke-embed-dim",
        type=int,
        default=64,
        help="Text embedding dim for --smoke (no Hugging Face download).",
    )
    args = parser.parse_args()

    cfg_path = Path(args.config)
    if not cfg_path.is_file():
        cfg_path = _ROOT / args.config
    cfg = load_config(cfg_path)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32

    if args.smoke:
        hidden = args.smoke_embed_dim
        llm_embed = torch.nn.Embedding(4096, hidden, device=device, dtype=dtype)
        for p in llm_embed.parameters():
            p.requires_grad = False
    else:
        llm_name = cfg["model"]["llm_name"]
        tokenizer = AutoTokenizer.from_pretrained(llm_name)
        llm = AutoModelForCausalLM.from_pretrained(llm_name, torch_dtype=dtype)
        llm_embed = llm.get_input_embeddings()
        for p in llm_embed.parameters():
            p.requires_grad = False
        hidden = int(getattr(llm.config, "hidden_size", cfg["model"]["llm_hidden_size"]))
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

    loss_fn = SymmetricInfoNCE(
        temperature=cfg["training"].get("temperature", 0.07)
    ).to(device)

    wb = None
    wandb_mod = None
    if os.environ.get("WANDB_DISABLED", "") != "1":
        try:
            import wandb as wandb_mod  # type: ignore
        except Exception:
            wandb_mod = None
        if wandb_mod is not None and cfg.get("wandb", {}).get("project"):
            try:
                wb = wandb_mod.init(
                    project=cfg["wandb"]["project"],
                    entity=cfg["wandb"].get("entity"),
                    name=cfg.get("run_name", "stage1"),
                    config=cfg,
                    tags=["stage1", cfg.get("experiment_type", "main")],
                )
            except Exception:
                wb = None

    trainable = [p for p in encoder.parameters() if p.requires_grad]
    opt = AdamW(trainable, lr=cfg["training"].get("lr", 1e-4), weight_decay=0.01)
    max_steps = cfg["training"].get("max_steps", 1000)
    sched = CosineAnnealingLR(opt, T_max=max_steps)

    os.makedirs(cfg.get("output_dir", "outputs/stage1_default"), exist_ok=True)

    if args.smoke:
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
        return

    train_ds = SpeechTextDataset(
        _ROOT / cfg["data"]["train_manifest"],
        tokenizer,
        load_hard_negs=cfg["training"].get("use_hard_neg", False),
    )
    val_ds = SpeechTextDataset(_ROOT / cfg["data"]["val_manifest"], tokenizer)
    bs = cfg["training"].get("batch_size_per_gpu", 4)
    train_loader = DataLoader(
        train_ds,
        batch_size=bs,
        shuffle=True,
        collate_fn=collate_stage1,
        num_workers=2,
        pin_memory=device.type == "cuda",
    )
    val_loader = DataLoader(
        val_ds, batch_size=64, shuffle=False, collate_fn=collate_stage1, num_workers=2
    )

    encoder.train()
    global_step = 0
    while global_step < max_steps:
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

            if global_step % cfg["training"].get("log_every", 50) == 0:
                msg = (
                    f"step {global_step} loss={loss.item():.4f} "
                    f"top1={out['top1_acc'].item():.3f} grad_norm={gn:.3f}"
                )
                print(msg)
                if wb is not None and wandb_mod is not None:
                    wandb_mod.log(
                        {
                            "train/loss": loss.item(),
                            "train/top1_acc": out["top1_acc"].item(),
                            "train/grad_norm": float(gn),
                            "train/lr": sched.get_last_lr()[0],
                        },
                        step=global_step,
                    )

            ev = cfg["training"].get("eval_every", 2000)
            if ev and global_step > 0 and global_step % ev == 0:
                metrics = evaluate_retrieval(encoder, val_loader, device)
                print("val", metrics)
                if wb is not None and wandb_mod is not None:
                    wandb_mod.log(
                        {f"val/{k}": v for k, v in metrics.items()}, step=global_step
                    )

            sv = cfg["training"].get("save_every", 5000)
            if sv and global_step > 0 and global_step % sv == 0:
                out_dir = cfg.get("output_dir", "outputs/stage1_default")
                ckpt_path = os.path.join(out_dir, f"ckpt_{global_step}.pt")
                torch.save(
                    {
                        "projector": encoder.projector.state_dict(),
                        "audio_pool": encoder.audio_pool.state_dict(),
                        "text_pool": encoder.text_pool.state_dict(),
                        "step": global_step,
                        "config": cfg,
                    },
                    ckpt_path,
                )
                print("saved", ckpt_path)

            global_step += 1
            if global_step >= max_steps:
                break
        if global_step >= max_steps:
            break

    if wb:
        wb.finish()


if __name__ == "__main__":
    main()
