from __future__ import annotations

import torch
import torch.nn as nn
from peft import LoraConfig, get_peft_model
from transformers import PreTrainedTokenizerBase

from cogen_align.models.projector import Projector


class SpeechLLM(nn.Module):
    """Whisper features -> projector -> soft prompt + LoRA causal LM."""

    def __init__(
        self,
        llm,
        projector: Projector,
        tokenizer: PreTrainedTokenizerBase,
        audio_start_token: str = "<|audio_start|>",
        audio_end_token: str = "<|audio_end|>",
        lora_rank: int = 64,
        lora_alpha: int = 128,
    ):
        super().__init__()
        special = {"additional_special_tokens": [audio_start_token, audio_end_token]}
        n_added = tokenizer.add_special_tokens(special)
        if n_added > 0:
            llm.resize_token_embeddings(len(tokenizer))

        self.audio_start_id = tokenizer.convert_tokens_to_ids(audio_start_token)
        self.audio_end_id = tokenizer.convert_tokens_to_ids(audio_end_token)
        self.tokenizer = tokenizer

        lora_config = LoraConfig(
            r=lora_rank,
            lora_alpha=lora_alpha,
            target_modules=[
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
                "gate_proj",
                "up_proj",
                "down_proj",
            ],
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
        )
        self.llm = get_peft_model(llm, lora_config)
        self.projector = projector

    def load_projector_from_stage1(self, ckpt_path: str) -> None:
        try:
            ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        except TypeError:
            ckpt = torch.load(ckpt_path, map_location="cpu")
        self.projector.load_state_dict(ckpt["projector"])
        step = ckpt.get("step", "?")
        print(f"Loaded projector from {ckpt_path} (step {step})")

    def build_inputs(
        self,
        audio_feat: torch.Tensor,
        audio_mask: torch.Tensor | None,
        text_ids: torch.Tensor,
        text_mask: torch.Tensor | None,
    ):
        b = audio_feat.size(0)
        device = audio_feat.device

        audio_emb = self.projector(audio_feat)
        t_a = audio_emb.size(1)
        if audio_mask is not None:
            new_mask = audio_mask[:, :: self.projector.subsample][:, :t_a]
        else:
            new_mask = torch.zeros(b, t_a, dtype=torch.bool, device=device)

        embed_layer = self.llm.get_input_embeddings()
        text_emb = embed_layer(text_ids)

        audio_start_emb = embed_layer(torch.tensor([self.audio_start_id], device=device))
        audio_end_emb = embed_layer(torch.tensor([self.audio_end_id], device=device))
        audio_start_emb = audio_start_emb.unsqueeze(0).expand(b, 1, -1)
        audio_end_emb = audio_end_emb.unsqueeze(0).expand(b, 1, -1)

        inputs_embeds = torch.cat(
            [audio_start_emb, audio_emb, audio_end_emb, text_emb], dim=1
        )

        audio_attn = (~new_mask).long()
        ones = torch.ones(b, 1, dtype=torch.long, device=device)
        text_attn = (~text_mask).long() if text_mask is not None else torch.ones_like(
            text_ids
        )
        attention_mask = torch.cat([ones, audio_attn, ones, text_attn], dim=1)

        audio_labels = torch.full(
            (b, 1 + t_a + 1), -100, dtype=torch.long, device=device
        )
        text_labels = text_ids.clone()
        if text_mask is not None:
            text_labels[text_mask] = -100
        labels = torch.cat([audio_labels, text_labels], dim=1)

        return inputs_embeds, attention_mask, labels

    def forward(
        self,
        audio_feat: torch.Tensor,
        audio_mask: torch.Tensor | None,
        text_ids: torch.Tensor,
        text_mask: torch.Tensor | None,
    ):
        inputs_embeds, attention_mask, labels = self.build_inputs(
            audio_feat, audio_mask, text_ids, text_mask
        )
        return self.llm(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            labels=labels,
        )

    @torch.no_grad()
    def generate(
        self, audio_feat: torch.Tensor, audio_mask: torch.Tensor | None, max_new_tokens: int = 128
    ):
        b = audio_feat.size(0)
        device = audio_feat.device
        audio_emb = self.projector(audio_feat)
        t_a = audio_emb.size(1)
        if audio_mask is not None:
            new_mask = audio_mask[:, :: self.projector.subsample][:, :t_a]
        else:
            new_mask = torch.zeros(b, t_a, dtype=torch.bool, device=device)

        embed_layer = self.llm.get_input_embeddings()
        audio_start_emb = embed_layer(
            torch.tensor([self.audio_start_id], device=device)
        ).unsqueeze(0).expand(b, 1, -1)
        audio_end_emb = embed_layer(
            torch.tensor([self.audio_end_id], device=device)
        ).unsqueeze(0).expand(b, 1, -1)
        inputs_embeds = torch.cat([audio_start_emb, audio_emb, audio_end_emb], dim=1)

        audio_attn = (~new_mask).long()
        ones = torch.ones(b, 1, dtype=torch.long, device=device)
        attention_mask = torch.cat([ones, audio_attn, ones], dim=1)

        pad = self.tokenizer.pad_token_id or self.tokenizer.eos_token_id
        return self.llm.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=pad,
        )
