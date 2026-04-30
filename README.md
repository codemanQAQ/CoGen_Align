# CoGen-Align

Contrastive warmup for data-efficient speech–text alignment in Speech LMs.  
设计说明与文献见 `doc/`。

## 仓库布局

| 路径 | 说明 |
|------|------|
| `configs/` | 实验 YAML（`inherit` 合并 `configs/base.yaml`，见 `cogen_align.utils.config`） |
| `src/cogen_align/` | 可安装包：模型、数据、loss、配置工具等 |
| `scripts/` | 训练、下载、评测、分析入口 |
| `scripts/get_manifests/` | **数据清单与特征路径**：生成 manifest、全量预计算 Whisper 特征、给 manifest 插入 `feature_path` |
| `data/manifests/` | jsonl 清单（字段约定见 `data/manifests/README.md`） |
| `data/features/` | 预计算 Whisper 特征 `.npy`（大文件不入库，见 `.gitignore`） |
| `outputs/` | checkpoint / log（占位见 `.gitignore`） |
| `tests/` | 单测：`projector`、`losses`、`dataset+collator` |
| `doc/` | 项目全览、调研等文档 |

`configs/base.yaml` 中与数据管线相关的字段：

- `data.audio_root`：LibriSpeech 根目录（含 `train-clean-100/` 等）
- `data.feature_root`：特征根目录（下面会**镜像** `audio_root` 的相对路径，扩展名为 `.npy`）
- `data.whisper_model`：Whisper 权重（Hub id 或本地目录）
- `data.sample_rate` / `data.max_duration`：预计算读波形与 **落盘前按时长截断特征** 的上限；须与训练用 ``max_audio_frames`` 等配置自洽

路径映射实现：`src/cogen_align/data/feature_paths.py` 中的 `audio_path_to_feature_path`。

---

## 环境

```bash
cd /data/speech/tts/intern/mqb/personal/cogen-align
python -m venv .venv && source .venv/bin/activate
pip install -e ".[train,dev]"
```

---

## 推荐数据流水线（manifest → 全量特征 → 补全 feature_path）

以下命令均在仓库根目录执行；路径按你本机修改。

### 1）下载 LibriSpeech（可选）

```bash
python scripts/download_librispeech.py \
  --output_dir /data/speech/tts/intern/mqb/wavs \
  --cache_dir  /data/speech/tts/intern/mqb/cache/datasets \
  --tmp_dir    /data/speech/tts/intern/mqb/cache/tmp \
  --splits clean:train.100 clean:train.360 clean:validation clean:test other:train.500 other:validation other:test
```

### 2）生成 manifest（不含或含 `feature_path` 均可）

```bash
python scripts/get_manifests/get_manifests.py \
  --librispeech_root /data/speech/tts/intern/mqb/wavs/LibriSpeech \
  --out_dir data/manifests/train \
  --train_hours 50
```

- 不传 `--train_hours`：写出全量 `train_all.jsonl`，并写出 `dev_*.jsonl` / `test_*.jsonl`（若目录存在）。
- 传 `--train_hours`：从训练池抽约 N 小时为 `train_Nh.jsonl`。
- 生成 manifest 时**可不写** `feature_path`；下一步全量算完 `.npy` 后用 `attach_feature_paths.py` 统一插入。

### 3）全量预计算 Whisper 特征（只扫音频目录，不读 manifest）

```bash
python scripts/get_manifests/precompute_features_all.py \
  --config configs/base.yaml
```

- 递归 `audio_root` 下所有 `.flac` / `.wav`，在 `feature_root` 下写**同目录树**的 `.npy`（**变长时间维**：按实际波形时长 ceil 去掉 Whisper padding 尾）。
- `--force`：覆盖已存在的 `.npy`（旧全长特征需重跑才能与当前 Dataset 对齐）。
- Whisper 首次会从 Hub 拉取到缓存；离线请先把模型拉到本地，再把 `data.whisper_model` 改为本地路径。

### 4）只读 manifest，插入 `feature_path`

```bash
python scripts/get_manifests/attach_feature_paths.py \
  --config configs/base.yaml \
  --manifest-in data/manifests/train/train_50h.jsonl \
  --manifest-out data/manifests/train/train_50h_with_features.jsonl \
  --check-npy
```

`--check-npy`：检查对应 `.npy` 是否存在，缺失会告警（仍写出 manifest，便于发现漏算）。

对 `data/manifests` 下全部 `.jsonl` **原地**写入 `feature_path`（不写新文件名）：

```bash
bash scripts/get_manifests/attach_all_manifests_features.sh
```

---

## Linux 正式训练：数值 + W&B 折线图 + checkpoint

三者并行、互不替代：

| 产物 | 位置 | 说明 |
|------|------|------|
| **Checkpoint** | `output_dir` 下 `ckpt_<step>.pt`、`ckpt_last.pt` | 权重文件，Stage2 用 `ckpt_last.pt` |
| **本地标量表** | `output_dir/metrics.jsonl` | 每 `log_every` 一行 train（含 `loss`、`loss_a2t`、`pos_sim` 等）+ `eval_every` 的 val 检索；**不设 W&B 也会写** |
| **W&B 折线图** | 浏览器打开 run → **Charts** | 对 `wandb.log` 的标量自动生成曲线；需联网与账号 |

**要用 W&B 画图**：不要设置 `WANDB_DISABLED`；在服务器上执行一次 `wandb login`（`pip install -e ".[train]"` 已含 wandb）。`configs/base.yaml` 的 `wandb.project` / `wandb.entity` / `wandb.notes` 按需填写。

多卡示例（4 卡、开 W&B）：

```bash
cd /data/speech/tts/intern/mqb/personal/cogen-align
NPROC=4 bash scripts/run_stage1_train.sh configs/stage1/table1_s1_50h.yaml
```

**一阶段按数据量从小到大连跑**（50h→…→500h 共 6 次，仅 `scripts/run_stage1_full_train.sh`，无命令行参数；多卡设 `NPROC`）：

```bash
NPROC=8 bash scripts/run_stage1_full_train.sh
```

**各数据量 Baseline（仅 Stage2）+ CoGen（Stage1→Stage2）顺序批跑**（无参数；`SKIP_STAGE2=1` 可在 Stage2 未实现时只跑 Stage1）：

```bash
SKIP_STAGE2=1 NPROC=8 bash scripts/run_full_experiment_batch.sh
```

---

## 训练前自检

```bash
pytest tests/ -q
```

Stage1 冒烟（不拉 Qwen、不读真实 manifest；**不写 W&B、不写 metrics.jsonl**）：

```bash
WANDB_DISABLED=1 python scripts/train_stage1.py --config configs/stage1/default.yaml --smoke
```

正式 Stage1：manifest 已含 `feature_path` 后，按 `doc/CoGen_Align_实验设计终稿.md` 选用配置（`configs/stage1/default.yaml` 或 `table1_s1_*h.yaml`），并确保本机可加载 Qwen：

```bash
torchrun --nproc_per_node=1 scripts/train_stage1.py --config configs/stage1/default.yaml
```

训练步数由 `training.max_epochs` 或 `max_steps` 控制。`output_dir` 另有 `experiment_record*.yaml/jsonl` 记录命令与 Git。

Stage2 脚本仍需按工程手册补全 DDP / WER 等逻辑（见 `doc/`）。

---

## 维护说明

- **推荐**只保留上述 `scripts/get_manifests/` 下的数据脚本；若你本地还有旧的「单文件 manifest 边算边写」脚本，可自行删除，避免与全量流程重复。
- 若移动 `get_manifests/` 下脚本的位置，注意修正文件内 `_REPO_ROOT = Path(__file__).resolve().parents[2]`（须指向本仓库根目录，以便找到 `src/` 与 `configs/`）。
