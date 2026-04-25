# CoGen-Align

Contrastive warmup for data-efficient speech–text alignment in Speech LMs.  
设计说明与文献见 `doc/`。

## 布局

- `configs/`：实验 YAML（`inherit` 指向 `configs/base.yaml`，由 `cogen_align.utils.config` 合并）
  - `configs/stage1/default.yaml`：Stage1 默认
  - `configs/stage2/baseline_*.yaml` / `cogen_*.yaml`：主实验矩阵
  - `configs/stage2/default.yaml`：本地 smoke（短 `max_steps`，继承 `baseline_10h`）
- `configs/sweeps/`：Wandb sweep 示例
- `src/cogen_align/`：可安装包（模型、数据、工具）
- `scripts/`：训练与预处理入口；`scripts/analysis/`：机制分析脚本
- `tests/`：`projector`、`losses`、`dataset+collator` 单测
- `data/manifests/`：jsonl 清单（字段见 `data/manifests/README.md`）
- `data/features/`：预计算 Whisper 特征（仅 `.gitkeep` 入库；`.npy` 在 `.gitignore`）
- `outputs/`：checkpoint / log（仅 `.gitkeep` 入库；其余见 `.gitignore`）

## 快速开始

```bash
cd /data/speech/tts/intern/mqb/personal/cogen-align
python -m venv .venv && source .venv/bin/activate
pip install -e ".[train,dev]"
```

单测（不拉模型、不读真实 manifest）：

```bash
pytest tests/ -q
```

离线验证算子与 loss（不访问 Hugging Face、不读 manifest）：

```bash
WANDB_DISABLED=1 python scripts/train_stage1.py --config configs/stage1/default.yaml --smoke
```

正式 Stage1 需本机已缓存或联网拉取 `Qwen/Qwen2.5-7B-Instruct`，并准备好 `data/manifests/*.jsonl` 与 `feature_path` 的 `.npy`：

```bash
torchrun --nproc_per_node=1 scripts/train_stage1.py --config configs/stage1/default.yaml
```

Stage2 脚本为占位实现，需按 `doc/CoGen_Align_Engineering_Handbook.docx` 接满 DDP/WER 等逻辑。
