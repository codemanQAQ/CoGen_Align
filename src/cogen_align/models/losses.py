from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class SymmetricInfoNCE(nn.Module):
    """Symmetric InfoNCE over batch (optional hard negatives on text side)."""

    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature

    def forward(
        self,
        z_a: torch.Tensor,
        z_t: torch.Tensor,
        hard_neg_z_t: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        z_a = F.normalize(z_a, dim=-1)
        z_t = F.normalize(z_t, dim=-1)
        b = z_a.size(0)

        if hard_neg_z_t is not None:
            hard_neg_z_t = F.normalize(hard_neg_z_t, dim=-1)
            k = hard_neg_z_t.size(1)
            extra = hard_neg_z_t.reshape(b * k, -1)
            all_z_t = torch.cat([z_t, extra], dim=0)
            logits_a2t = z_a @ all_z_t.T / self.temperature
            logits_t2a = z_t @ z_a.T / self.temperature
        else:
            logits_a2t = z_a @ z_t.T / self.temperature
            logits_t2a = z_t @ z_a.T / self.temperature

        labels = torch.arange(b, device=z_a.device)
        loss_a2t = F.cross_entropy(logits_a2t, labels)
        loss_t2a = F.cross_entropy(logits_t2a, labels)
        loss = (loss_a2t + loss_t2a) / 2

        with torch.no_grad():
            pos_sim = (z_a * z_t).sum(dim=-1).mean()
            mask = ~torch.eye(b, dtype=torch.bool, device=z_a.device)
            neg_sim = (z_a @ z_t.T)[mask].mean()
            top1_acc = (logits_a2t.argmax(dim=-1) == labels).float().mean()

        return {
            "loss": loss,
            "loss_a2t": loss_a2t,
            "loss_t2a": loss_t2a,
            "pos_sim": pos_sim,
            "neg_sim": neg_sim,
            "top1_acc": top1_acc,
        }
