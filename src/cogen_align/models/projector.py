from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class AttentionPooling(nn.Module):
    """Single-query attention pooling over time."""

    def __init__(self, dim: int, num_heads: int = 8):
        super().__init__()
        self.query = nn.Parameter(torch.randn(1, 1, dim) * 0.02)
        self.attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        # x: [B, T, D], mask: [B, T] True = padding (ignored by MHA as key_padding_mask)
        b = x.size(0)
        q = self.query.expand(b, -1, -1)
        out, _ = self.attn(q, x, x, key_padding_mask=mask)
        return self.norm(out.squeeze(1))


class Projector(nn.Module):
    """Whisper frames -> LLM hidden space with frame subsampling."""

    def __init__(
        self,
        in_dim: int = 1280,
        out_dim: int = 3584,
        hidden_dim: int = 4096,
        subsample: int = 5,
    ):
        super().__init__()
        self.subsample = subsample
        self.fc1 = nn.Linear(in_dim * subsample, hidden_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, out_dim)
        self.norm = nn.LayerNorm(out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, in_dim]
        b, t, d = x.shape
        pad = (self.subsample - t % self.subsample) % self.subsample
        if pad:
            x = F.pad(x, (0, 0, 0, pad))
        t_new = x.size(1) // self.subsample
        x = x.reshape(b, t_new, d * self.subsample)
        x = self.act(self.fc1(x))
        x = self.fc2(x)
        return self.norm(x)


class SpeechTextEncoder(nn.Module):
    """Stage 1 dual tower: audio (projector + pool) and text (embed + pool)."""

    def __init__(
        self,
        projector: Projector,
        llm_embed: nn.Embedding,
        out_dim: int = 3584,
    ):
        super().__init__()
        self.projector = projector
        self.llm_embed = llm_embed
        self.audio_pool = AttentionPooling(out_dim)
        self.text_pool = AttentionPooling(out_dim)

    def encode_audio(
        self, audio_feat: torch.Tensor, audio_mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        x = self.projector(audio_feat)
        if audio_mask is not None:
            m = audio_mask[:, :: self.projector.subsample][:, : x.size(1)]
        else:
            m = None
        return self.audio_pool(x, mask=m)

    def encode_text(
        self, text_ids: torch.Tensor, text_mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        x = self.llm_embed(text_ids)
        return self.text_pool(x, mask=text_mask)
