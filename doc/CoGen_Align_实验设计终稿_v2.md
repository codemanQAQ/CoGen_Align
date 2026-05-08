# CoGen-Align 实验设计终稿 v2

> 更新时间：2026年4月30日
> 基于路线 B（自监督 Encoder）重构，覆盖主实验、消融、泛化性验证、数据切分、Sanity Check
> 
> **v2 核心变更**：语音编码器从 Whisper-large-v3 → **HuBERT-large**（自监督），以获得真正有挑战性的对齐场景

---

## 〇、为什么换 Encoder（v1 → v2 的核心决策）

### 问题

v1 使用 Whisper-large-v3 作为语音编码器。实验发现：50h 数据 + 随机初始化 Projector，仅 500 步训练即可在 LibriSpeech 上实现近乎完美转写。**原因：Whisper 本身是用 68 万小时标注数据训出的 ASR 模型，其 encoder 输出天然就是 ASR-ready 的表征，Projector 只需学习简单的线性变换。** 在这个设置下，对比学习 warmup 几乎没有改进空间。

### 解决方案

将编码器换为 **HuBERT-large**（自监督预训练，无 ASR 标注数据）。HuBERT 的表征是通用语音表征（语义、声学、说话人信息混合），不是专为 ASR 优化的。将其对齐到 LLM 空间需要 Projector 做更复杂的变换——**这才是"对齐"真正有挑战性的场景，也是对比学习 warmup 有价值的场景。**

### 论文叙事

> "工业方案（如 Qwen2-Audio）使用监督式 Encoder（Whisper），对齐相对容易但 Encoder 本身依赖大量标注数据。自监督 Encoder（HuBERT, WavLM）不需要标注数据即可预训练，是低资源语言和新领域的首选——但将其表征对齐到 LLM 的难度显著更高。CoGen-Align 正是针对这一更具挑战性的场景提出的。"

### 补充实验（Appendix）

保留一组 Whisper encoder 的对比实验放 Appendix，证明"用 Whisper 时 CoGen 优势变小"——这反过来支撑了论文的核心论点：**CoGen 在对齐困难的场景下价值最大。**

---

## 一、核心 Claim

> 在自监督语音编码器（HuBERT）与 LLM 的对齐场景中，相同数据量下，CoGen-Align（对比学习 warmup + 生成式对齐）的 WER 全程低于纯生成式 Baseline。
> 由此可量化：CoGen 用 X 小时达到 Baseline 需要 Y 小时才能达到的效果（数据效率提升）。

---

## 二、架构（固定配置）

| 组件 | 选型 | 参数量 | 输出维度 | 是否冻结 |
|------|------|--------|---------|---------|
| 语音编码器 | **HuBERT-large** (`facebook/hubert-large-ll60k`) | 316M | **1024** | 冻结 |
| Projector | 2-layer MLP，hidden 4096，subsample 5x | ~25M | 3584 | 训练 |
| LLM | Qwen2.5-7B-Instruct | 7B | 3584 | Stage 1 冻结，Stage 2 LoRA rank=64 |

### 与 v1 的差异

| | v1 (Whisper) | v2 (HuBERT) |
|---|---|---|
| Encoder | Whisper-large-v3 | **HuBERT-large** |
| 预训练方式 | 有监督 ASR（68万小时标注） | **自监督**（6万小时无标注） |
| Encoder 输出维度 | 1280 | **1024** |
| Projector in_dim | 1280 | **1024** |
| 帧率 | 50 Hz | **50 Hz**（相同，subsample 不变） |
| 对齐难度 | 低（encoder 已是 ASR-ready） | **高（通用表征需要更多对齐）** |

### 代码改动（仅 config 和特征预计算）

```yaml
# configs/base.yaml 改动
model:
  encoder_name: "facebook/hubert-large-ll60k"    # 新增
  whisper_feature_dim: 1024                       # 从 1280 改为 1024
  # 其余不变
```

```python
# 特征预计算改动
# 之前: WhisperModel → encoder → last_hidden_state [T, 1280]
# 现在: HubertModel → last_hidden_state [T, 1024]

from transformers import HubertModel, Wav2Vec2Processor
model = HubertModel.from_pretrained("facebook/hubert-large-ll60k")
processor = Wav2Vec2Processor.from_pretrained("facebook/hubert-large-ll60k")

inputs = processor(audio, sampling_rate=16000, return_tensors="pt")
features = model(inputs.input_values).last_hidden_state  # [T, 1024]
```

---

## 三、训练流程

### Stage 1：Contrastive Warmup

- 输入：`<audio, text>` 配对数据（与 Stage 2 完全相同，数据复用）
- 前向：**HuBERT feat** → Projector → AttentionPooling → z_a；LLM Embed(text) → AttentionPooling → z_t
- 损失：对称 InfoNCE(z_a, z_t)，in-batch 负例
- 更新：仅 Projector + AttentionPooling，**HuBERT** 和 LLM 全部冻结
- **Table 1 主实验**：Stage1 **收敛式停止**（dev-clean 上 top-1 检索均值 > 0.95 且连续 3 次 eval 无提升；见 `configs/stage1/default.yaml`）。**消融 A.1**：固定 100h，**optimizer steps ∈ {0,500,1000,2000}**，与主实验停止方式解耦。第一版仅 1 epoch 时 warmup 严重不足（见 §九 Pilot 历史结论）。

### Stage 2：Generative Alignment

- 输入：与 Stage 1 **相同**的 `<audio, text>` 配对数据
- Projector 从 Stage 1 checkpoint 初始化
- 损失：Cross-entropy on text tokens（ASR 任务）
- 更新：Projector（继续训练）+ LLM LoRA (rank=64)
- 可选：L_total = L_gen + λ · L_contrast（由消融 A.3 确定最优 λ）

### 数据复用说明

Warmup 和生成式使用同一份数据，无额外数据成本。总数据用量与 Baseline 严格相同，比较公平。

---

## 四、主实验（Table 1）

固定架构：MLP + Qwen2.5-7B；**Stage 1 主实验为收敛式停止**；A.1 在固定 100h 上扫 warmup **步数**，与主实验解耦。

| 数据量 X | Baseline | CoGen |
|---------|---------|-------|
| 50h | Gen=50h | Warmup=50h → Gen=50h |
| 100h | Gen=100h | Warmup=100h → Gen=100h |
| 200h | Gen=200h | Warmup=200h → Gen=200h |
| 300h | Gen=300h | Warmup=300h → Gen=300h |
| 400h | Gen=400h | Warmup=400h → Gen=400h |
| 500h | Gen=500h | Warmup=500h → Gen=500h |

**对比逻辑：**

```
Baseline-100h：[────────── 生成式 100h ──────────]  → WER_B
CoGen-100h：   [── warmup 100h ──][── 生成式 100h ──]  → WER_C

若 WER_C < WER_B，说明相同数据量下 warmup 带来了提升
横向箭头：CoGen@Xh ≈ Baseline@Yh（Y>X），节省 (Y-X)/Y × 100% 的数据
```

### 预期性能（基于文献估算）

由于 HuBERT encoder 不是 ASR 专用，baseline 性能将远弱于 Whisper 设置：

| 数据量 | Baseline WER-clean 预估 | CoGen 改进空间 |
|--------|------------------------|---------------|
| 50h | 35-50% | 大 |
| 100h | 20-35% | 大 |
| 200h | 15-25% | 中 |
| 300h | 12-20% | 中 |
| 500h | 8-15% | 中-小 |

**与 Whisper 设置的对比（Appendix 数据）：**

| | Whisper baseline 100h | HuBERT baseline 100h |
|---|---|---|
| WER-clean 预估 | 5-8% | 20-35% |
| CoGen 改进空间 | 极小 | **大** |

---

## 五、消融实验（Table 2）

固定：MLP + Qwen2.5-7B + 100h。

| ID | 变量 | 取值 | 目的 |
|----|------|------|------|
| A.1 | Stage 1 **optimizer steps**（100h） | {0, 500, 1000, 2000} | 与主实验收敛停止解耦；回答 warmup 时长效应 |
| A.2 | Pooling 方式 | {mean, attention} | 验证 AttentionPooling 是否必要 |
| A.3 | λ (joint contrast) | {0, 0.1, 0.3, 0.5} | Stage 2 加对比正则是否有帮助 |

**说明：**
- A.1 中 steps=0 为「零步 warmup」ckpt；是否与 Baseline-100h 完全等价以实验设计为准
- A.3 中 λ=0 可复用主实验 CoGen-100h 的结果

---

## 六、泛化性验证（Table 3）

固定：最优 epoch + 最优 λ + 100h。MLP+7B 结果复用主实验。

| ID | 变量 | 取值 | 目的 |
|----|------|------|------|
| A.4 | Projector 架构 | {MLP, Q-Former, Subsampling} | 普适性（projector） |
| A.5 | LLM 尺寸 | {Qwen2.5-1B, 3B, 7B} | 普适性（LLM scale） |
| A.6 | Encoder 选择 | {HuBERT-large, WavLM-large, Whisper-large-v3} | **跨 encoder 验证** |

### A.6 的设计说明（新增）

| Encoder | 预训练方式 | 对齐难度 | 预期 CoGen 优势 |
|---------|----------|---------|----------------|
| HuBERT-large | 自监督 | 高 | 大（主实验验证） |
| WavLM-large | 自监督 | 高 | 大（泛化性验证） |
| Whisper-large-v3 | 有监督 ASR | 低 | 小（反向验证：解释为什么 CoGen 在对齐难时更有价值） |

A.6 的论文价值：
- WavLM 结果证明 CoGen 方法对不同自监督 encoder 普适
- Whisper 结果证明"对齐越难 → CoGen 越有用"这个 insight，**正反两面验证**

---

## 七、数据切分方案

### 数据来源

LibriSpeech 960h 训练集：
```
train-clean-100：100h，251 speakers
train-clean-360：360h，921 speakers
train-other-500：500h，1166 speakers
```

### 切分方式

960h 合并后**按 utterance 随机打乱**（seed=42），按时长截取各数据量点，保证严格子集关系：

```
50h ⊂ 100h ⊂ 200h ⊂ 300h ⊂ 400h ⊂ 500h
```

使用现有 `get_manifests.py` 生成：

```bash
for hours in 50 100 200 300 400 500; do
    python get_manifests.py \
        --librispeech_root /path/to/librispeech \
        --out_dir data/manifests \
        --train_hours $hours \
        --seed 42
done
```

### 验证集 / 开发集

**Stage 2**：**`dev-clean` + `dev-other`** 同时作为开发集（`data.val_manifests`）；训练中 WER 与 **早停** 用两集 WER **均值**；`metrics.jsonl` 中 **分别记录** 各集 WER、**`wer_dev_mean`/`wer`**。

**Stage 1**：仍仅用 **`dev-clean`**（`val_manifest`），与 Stage 2 WER 开发集 **独立**。

**测试集**：`test-clean` / `test-other`，仅 Stage 2 训练结束后一次性评测。

---

## 八、评测指标

| 指标 | 用途 | 来源 | 备注 |
|------|------|------|------|
| WER-clean | 主指标 | LibriSpeech test-clean | 现在确定 |
| WER-other | 主指标 | LibriSpeech test-other | 现在确定 |
| 达到目标 WER 所需步数 | 辅助指标，支撑收敛速度 claim | 训练 log | 待 Baseline 跑完后确定目标 WER 锚点 |

**收敛速度指标的确定流程：**
```
1. 先跑完所有 Baseline，得到各数据量点的最终 WER
2. 选取若干 WER 锚点（如 Baseline-500h 的最终 WER 作为目标）
3. 回头看 Baseline 和 CoGen 分别在第几步达到该锚点
4. 计算步数节省比例
```

**Killer Chart：** 横轴数据量（log scale），纵轴 WER，Baseline 和 CoGen 两条曲线对比。

---

## 九、前置实验与 Sanity Check

### 前置实验（Week 1-2 末）

在正式跑主实验之前，先跑 50h 快速验证：

```
Baseline-50h：HuBERT feat → random Projector → LLM，Gen=50h，训练到收敛
CoGen-50h：  Warmup=50h（**Stage1 收敛式**）→ Gen=50h，训练到收敛
```

目的：
1. 得到真实 WER 数字，用来定 Sanity Check 阈值
2. 验证 CoGen 在最小数据量点是否有优势（**这是最关键的验证——如果 HuBERT 设置下 CoGen 仍无优势，需要重新评估**）
3. 成本低，出问题早发现

### Pilot 结论（第一版：`max_epochs=1`，Whisper/与现脚本同设定下）

Stage1 仅 1 epoch 时步数极少：50h **29**、100h **58**、200h **115**、500h **286**；projector **warmup 不足**（如 200h 末 loss≈0.59），CoGen 收益不稳定。**现行主实验改为收敛式 Stage1**（见 `default.yaml` 与 `train_stage1.py`）。

### 预期结果（与 v1 Whisper 设置的对比）

| 设置 | Baseline-50h WER 预估 | CoGen 改进预期 |
|------|----------------------|---------------|
| v1 Whisper | ~8-15%（太容易） | 几乎无改进 |
| **v2 HuBERT** | ~35-50% | **显著改进（目标：降 5-15 个绝对点）** |

### Sanity Check 标准（基于前置实验结果填入）

| Check | 条件 | 阈值 |
|-------|------|------|
| Check 0 | HuBERT Baseline-50h WER 显著高于 Whisper Baseline-50h WER | HuBERT WER > 2× Whisper WER（**若不满足，说明 HuBERT 对齐也太容易，需再换方案**） |
| Check 1 | Stage 1 结束后，验证集 audio→text top-1 retrieval accuracy | > 20% |
| Check 2 | Baseline-50h 训练 2000 steps 后 WER | < 待定（参考前置实验） |
| Check 3 | CoGen-50h 同等 steps 下 WER < Baseline-50h WER | 必须成立 |

### 决策规则

```
Check 0 未过 → HuBERT 对齐也太容易，考虑换更弱的 encoder 或换任务
Check 1 未过 → debug Stage 1，不继续
Check 2 未过 → debug Stage 2 pipeline，不继续
Check 3 未过 → 停止，排查根本原因，不跑主实验
四条全过 → 放大规模跑主实验
```

---

## 十、机制分析

| ID | 内容 | 产出 | 是否保留 |
|----|------|------|---------|
| AN.1 | Representation cosine similarity heatmap | warmup 前/后对比热图 | 必做 |
| AN.3 | Loss landscape 2D slice | 等高线图 | 必做 |
| AN.4 | Gradient norm trajectory | 曲线图（从 wandb 直接拉，零额外成本） | 必做 |
| AN.5 | t-SNE of speech/text representations | 散点图 | 必做 |
| ~~AN.2~~ | ~~Anisotropy of speech embeddings~~ | ~~曲线 vs 训练步数~~ | 砍掉（结论与 AN.1 重叠，读者不直观） |

**Figure 安排：**
- Figure 3：AN.1 + AN.5，4 个 panel（warmup 前/后各一组 heatmap + t-SNE）
- Figure 4：AN.3 + AN.4，loss landscape 等高线 + gradient norm 曲线

---

## 十一、与相关工作的区分

### 与 Züfle & Niehues (arXiv:2412.15712) 的区分

| 差异点 | Züfle & Niehues | CoGen-Align |
|-------|-----------------|-------------|
| Stage 2 LLM | 全程冻结 | LoRA 微调（填补其 limitation） |
| 数据量点 | 仅 10% vs 100% 两个相对比例点 | 50h–500h 六个绝对小时量级点 |
| 机制分析 | 无 | AN.1 / AN.3 / AN.4 / AN.5 |
| **Encoder** | **Whisper（对齐容易）** | **HuBERT（对齐困难，更有挑战性）** |

### 与 Qwen2-Audio 的定位

| | Qwen2-Audio | CoGen-Align |
|---|---|---|
| Encoder | Whisper-large-v3（有监督） | **HuBERT-large（自监督）** |
| 训练数据 | 500K+ 小时多任务数据 | 50-500 小时 LibriSpeech |
| 对齐方法 | 多任务生成式预训练 | **对比学习 warmup + 生成式对齐** |
| 目标 | 多模态通用理解 | **数据高效对齐** |
| 关系 | 不直接对比，作为工业参考 | 解决更基础的对齐效率问题 |

### Related Work 定位句

> Concurrent to our work, Züfle & Niehues (2024) explore contrastive pretraining for SpeechLLMs, but freeze the LLM throughout fine-tuning, evaluate only two data points (10% vs. 100%), and provide no mechanistic analysis. CoGen-Align extends this line of work by incorporating LoRA-based LLM adaptation, providing fine-grained scaling curves across absolute data budgets (50h–500h), and offering mechanistic insights via representation geometry and loss landscape analysis. Notably, we study the more challenging setting of self-supervised encoders (HuBERT), where the alignment problem is non-trivial and data efficiency gains are most impactful.

### Introduction motivation 句

> Industrial SpeechLMs such as Qwen2-Audio (Chu et al., 2024) use supervised encoders (Whisper) pretrained on 680K hours of labeled ASR data, making alignment relatively easy. However, self-supervised encoders (HuBERT, WavLM) — which require no labeled data for pretraining and are thus the preferred choice for low-resource languages and novel domains — present a significantly harder alignment challenge. We focus on this more demanding setting and show that contrastive warmup substantially improves data efficiency.

---

## 十二、论文结构（ICASSP 2026，4 页 + references）

### 页面分配

| Section | 篇幅 | 内容 |
|---------|------|------|
| 1. Introduction | 1 页 | 背景（监督 vs 自监督 encoder 的对齐难度差异）→ 两种范式缺陷 → 洞察 → 贡献列表 → Related Work |
| 2. Method | 0.75 页 | Stage 1 / Stage 2 / Joint Regularization + Figure 1 Pipeline 图 |
| 3. Experiments | 2 页 | Setup / Figure 2 Killer Chart / Table 1 主实验 / Table 2 消融 / Table 3 泛化性（含 A.6 encoder 对比） / Figure 3+4 机制分析 |
| 4. Conclusion | 0.25 页 | 总结 + limitation + future work |

### Figure 安排

| Figure | 内容 |
|--------|------|
| Figure 1 | Pipeline 总览（**HuBERT** → Projector → Stage 1/Stage 2） |
| Figure 2 | Killer Chart（数据效率曲线，横轴 log scale） |
| Figure 3 | AN.1 + AN.5（cosine heatmap + t-SNE，4 panels） |
| Figure 4 | AN.3 + AN.4（loss landscape + gradient norm） |

### Table 安排

| Table | 内容 |
|-------|------|
| Table 1 | 主实验（6 个数据量点，Baseline vs CoGen，WER-clean / WER-other） |
| Table 2 | 消融（A.1 warmup steps / A.2 pooling / A.3 λ） |
| Table 3 | 泛化性（A.4 projector / A.5 LLM size / A.6 encoder，含 Whisper 反向验证） |

---

## 十三、Appendix 补充实验

### Whisper encoder 的反向验证

在 Appendix 中放一组 Whisper-large-v3 的实验结果（100h 数据点即可）：

| | Baseline WER | CoGen WER | 差值 |
|---|---|---|---|
| HuBERT-large (主实验) | 20-35% | 15-25% | **5-10 绝对点** |
| Whisper-large-v3 (appendix) | 5-8% | 5-7% | <1 绝对点 |

这个对比**正面支撑了论文的核心论点**：

> "The benefit of contrastive warmup is most pronounced when the encoder-LLM alignment is challenging (self-supervised encoders). When alignment is already easy (supervised encoders like Whisper), warmup provides diminishing returns — which is expected and consistent with our mechanistic analysis."

---

## 十四、待讨论事项

- [x] 评测指标：WER-clean / WER-other / 收敛步数
- [x] 与 Züfle (arXiv:2412.15712) 的区分写法
- [x] 论文结构
- [x] Encoder 选择：HuBERT-large（主实验）+ WavLM / Whisper（泛化性验证）
- [ ] 时间规划更新（需基于 HuBERT 特征预计算时间重新估算）
- [ ] HuBERT 特征预计算脚本编写与测试
- [ ] 50h Sanity Check（HuBERT baseline + CoGen）

---

## 十五、Stage 1 训练配置（代码实现）

与本文二、三、五、七、九节对齐的 YAML 位于 `configs/stage1/`：

| 文件 | 用途 |
|------|------|
| `default.yaml` | 默认主配置：`train_100h` + `dev_clean`，**`stop_mode: convergence`（Table 1）**，对称 InfoNCE |
| `table1_s1_{50,200,300,400,500}h.yaml` | Table 1 各数据量 Stage1 |
| `table1_s1_50h.yaml` | 前置实验 Warmup（50h + dev-clean）与 Table1@50h 共用 |
| `ablation_a1_steps_{0,500,1000,2000}.yaml` | A.1 固定步数消融（100h） |
| `ablation_temperature.yaml` / `ablation_hardneg.yaml` | 扩展消融 |

### 关键 config 变更（v1 → v2）

```yaml
# configs/base.yaml
model:
  encoder_type: "hubert"                              # 新增
  encoder_name: "facebook/hubert-large-ll60k"         # 新增（替代 whisper_model）
  whisper_feature_dim: 1024                           # 从 1280 → 1024
  llm_hidden_size: 3584                               # 不变
  projector:
    type: mlp
    hidden_dim: 4096                                  # 不变
    output_dim: 3584                                  # 不变
    subsample: 5                                      # 不变（HuBERT 帧率也是 50Hz）
    use_attention_pooling: true
```

训练脚本：`scripts/train_stage1.py` 不需改动（它读取预计算特征，不关心 encoder 是什么）。

**Stage 2（`configs/stage2/`）**

| 文件 | 用途 |
|------|------|
| `common.yaml` | Baseline / CoGen 共用：LoRA、优化超参；dev/test 清单均为 `data/manifests/train/` 下单 `.jsonl` |
| `baseline_{50,100,200,300,400,500}h.yaml` | Table1 纯生成式 |
| `cogen_{50,…,500}h.yaml` | 同数据量 + `warmup_init_path` |
| `default.yaml` | 短 `max_steps` smoke test |

Stage2 训练脚本同样不需改动，只需确保 `Projector(in_dim=1024)` 匹配 config。

---

## 十六、特征预计算脚本变更

### HuBERT 特征预计算（替代 Whisper 版本）

```python
# scripts/precompute_hubert_features.py
from transformers import HubertModel, Wav2Vec2Processor
import soundfile as sf
import torch
import numpy as np

model = HubertModel.from_pretrained("facebook/hubert-large-ll60k",
                                      torch_dtype=torch.bfloat16)
processor = Wav2Vec2Processor.from_pretrained("facebook/hubert-large-ll60k")
model.eval().cuda()

def encode_audio(audio_path):
    audio, sr = sf.read(audio_path)
    if sr != 16000:
        import librosa
        audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)
    inputs = processor(audio, sampling_rate=16000, return_tensors="pt")
    with torch.no_grad():
        outputs = model(inputs.input_values.to("cuda", dtype=torch.bfloat16))
    # last_hidden_state: [1, T, 1024]
    feat = outputs.last_hidden_state[0].to(torch.float16).cpu().numpy()
    return feat  # [T, 1024]
```

### 与 Whisper 版本的差异

| | Whisper | HuBERT |
|---|---|---|
| 输入 | log-mel spectrogram (80 dim) | 原始波形 (raw waveform) |
| 处理器 | `WhisperProcessor` | `Wav2Vec2Processor` |
| 模型 | `WhisperModel.encoder` | `HubertModel` |
| 输出维度 | 1280 | **1024** |
| 帧率 | ~50 Hz | ~50 Hz |
| 最大长度 | 固定 30s (1500 帧) | **可变长度**（按实际音频长度） |

注意：Whisper encoder 固定输出 1500 帧（30s），HuBERT 输出长度和音频时长成正比。Projector 的 subsample=5 仍然适用。

---

## 十七、立即行动清单

```
今天（Day 0）:
[ ] 1. 下载 HuBERT-large 模型
[ ] 2. 用 8h 数据预计算 HuBERT 特征（5-10 分钟）
[ ] 3. 修改 config: whisper_feature_dim 1280 → 1024
[ ] 4. 跑 Baseline-8h 500 步，看 WER（目标：> 50%，证明对齐确实难）

Day 1:
[ ] 5. 如果 Check 0 通过（HuBERT 对齐确实比 Whisper 难很多）→ 继续
[ ] 6. 预计算 50h 的 HuBERT 特征
[ ] 7. 跑前置实验（Baseline-50h + CoGen-50h）

Day 2-3:
[ ] 8. 完成 Sanity Check（Check 1-3）
[ ] 9. 如果全部通过 → 预计算完整 500h 特征 + 开始主实验

Day 4-10:
[ ] 10. 跑完整主实验 + 消融实验

Day 11-14:
[ ] 11. 机制分析 + 论文写作
```
