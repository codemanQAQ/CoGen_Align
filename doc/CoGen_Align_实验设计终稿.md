# CoGen-Align 实验设计终稿

> 更新时间：2026年4月26日
> 基于与 Claude 的讨论整理，覆盖主实验、消融、泛化性验证、数据切分、Sanity Check

---

## 一、核心 Claim

> 相同数据量下，CoGen-Align（对比学习 warmup + 生成式对齐）的 WER 全程低于纯生成式 Baseline。
> 由此可量化：CoGen 用 X 小时达到 Baseline 需要多少小时才能达到的效果（数据效率提升）。

---

## 二、架构（固定配置）

| 组件 | 选型 | 是否冻结 |
|------|------|---------|
| 语音编码器 | Whisper-large-v3 | 冻结 |
| Projector | 2-layer MLP，hidden 4096，subsample 5x | 训练 |
| LLM | Qwen2.5-7B-Instruct | Stage 1 冻结，Stage 2 LoRA rank=64 |

---

## 三、训练流程

### Stage 1：Contrastive Warmup

- 输入：`<audio, text>` 配对数据（与 Stage 2 完全相同，数据复用）
- 前向：Whisper → Projector → AttentionPooling → z_a；LLM Embed(text) → AttentionPooling → z_t
- 损失：对称 InfoNCE(z_a, z_t)，in-batch 负例
- 更新：仅 Projector，Whisper 和 LLM 全部冻结
- 训练量：固定 epochs（由消融 A.1 确定最优值）

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

固定架构：MLP + Qwen2.5-7B，固定最优 Stage 1 epoch（由消融 A.1 确定）。

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

---

## 五、消融实验（Table 2）

固定：MLP + Qwen2.5-7B + 100h。

| ID | 变量 | 取值 | 目的 |
|----|------|------|------|
| A.1 | Stage 1 epochs | {0, 1, 2, 3} | 找最优 warmup 轮数，0 即 Baseline |
| A.2 | Pooling 方式 | {mean, attention} | 验证 AttentionPooling 是否必要 |
| A.3 | λ (joint contrast) | {0, 0.1, 0.3, 0.5} | Stage 2 加对比正则是否有帮助 |

**说明：**
- A.1 中 epoch=0 即纯 Baseline，可复用主实验 Baseline-100h 的结果
- A.3 中 λ=0 可复用主实验 CoGen-100h 的结果

---

## 六、泛化性验证（Table 3）

固定：最优 epoch + 最优 λ + 100h。MLP+7B 结果复用主实验。

| ID | 变量 | 取值 | 目的 |
|----|------|------|------|
| A.4 | Projector 架构 | {MLP, Q-Former, Subsampling} | 普适性（projector） |
| A.5 | LLM 尺寸 | {Qwen2.5-1B, 3B, 7B} | 普适性（LLM scale） |

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

### 验证集

使用 LibriSpeech 标准验证集：
- `dev-clean`：验证集（训练过程监控）
- `test-clean` / `test-other`：最终评测（训练结束后一次性评测）

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
Baseline-50h：Gen=50h，训练到收敛
CoGen-50h：  Warmup=50h（1 epoch）→ Gen=50h，训练到收敛
```

目的：
1. 得到真实 WER 数字，用来定 Sanity Check 阈值
2. 验证 CoGen 在最小数据量点是否有优势
3. 成本低，出问题早发现

### Sanity Check 标准（基于前置实验结果填入）

| Check | 条件 | 阈值（待填） |
|-------|------|------------|
| Check 1 | Stage 1 结束后，验证集 audio→text top-1 retrieval accuracy | > 20% |
| Check 2 | Baseline-50h 训练 2000 steps 后 WER | < 待定（参考前置实验） |
| Check 3 | CoGen-50h 同等 steps 下 WER < Baseline-50h WER | 必须成立 |

### 决策规则

```
三条全过 → 放大规模跑主实验
Check 1 未过 → debug Stage 1，不继续
Check 2 未过 → debug Stage 2 pipeline，不继续
Check 3 未过 → 停止，排查根本原因，不跑主实验
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

## 十一、与 Züfle & Niehues (arXiv:2412.15712) 的区分写法

### 可作为贡献的核心差异

| 差异点 | Züfle & Niehues | CoGen-Align |
|-------|-----------------|-------------|
| Stage 2 LLM | 全程冻结 | LoRA 微调（填补其 limitation） |
| 数据量点 | 仅 10% vs 100% 两个相对比例点 | 50h–500h 六个绝对小时量级点 |
| 机制分析 | 无 | AN.1 / AN.3 / AN.4 / AN.5 |

### Related Work 定位句

> Concurrent to our work, Züfle & Niehues (2024) explore contrastive pretraining for SpeechLLMs, but freeze the LLM throughout fine-tuning, evaluate only two data points (10% vs. 100%), and provide no mechanistic analysis. CoGen-Align extends this line of work by incorporating LoRA-based LLM adaptation, providing fine-grained scaling curves across absolute data budgets (50h–500h), and offering mechanistic insights via representation geometry and loss landscape analysis.

### Introduction motivation 句

> While Züfle & Niehues (2024) demonstrate promise for contrastive warmup in SpeechLLMs, their study leaves open three questions: (i) does the benefit hold when the LLM is also adapted via LoRA? (ii) how does data efficiency scale across absolute data budgets rather than relative proportions? (iii) why does warmup help, mechanistically? We address all three.

**注意**：定位为"系统性延伸"（extend）而非竞争，必须引用，不能忽略。

---

## 十二、论文结构（ICASSP 2026，4 页 + references）

### 页面分配

| Section | 篇幅 | 内容 |
|---------|------|------|
| 1. Introduction | 1 页 | 背景 → 两种范式缺陷 → 洞察 → 贡献列表 → Related Work |
| 2. Method | 0.75 页 | Stage 1 / Stage 2 / Joint Regularization + Figure 1 Pipeline 图 |
| 3. Experiments | 2 页 | Setup / Figure 2 Killer Chart / Table 1 主实验 / Table 2 消融 / Figure 3 泛化性 / Figure 4+5 机制分析 |
| 4. Conclusion | 0.25 页 | 总结 + limitation + future work |

### Figure 安排

| Figure | 内容 |
|--------|------|
| Figure 1 | Pipeline 总览（Stage 1 → Stage 2） |
| Figure 2 | Killer Chart（数据效率曲线，横轴 log scale） |
| Figure 3 | 泛化性（Projector 架构 + LLM 尺寸，合并为一张图） |
| Figure 4 | AN.1 + AN.5（cosine heatmap + t-SNE，4 panels） |
| Figure 5 | AN.3 + AN.4（loss landscape + gradient norm） |

### Table 安排

| Table | 内容 |
|-------|------|
| Table 1 | 主实验（6 个数据量点，Baseline vs CoGen，WER-clean / WER-other） |
| Table 2 | 消融（A.1 epochs / A.2 pooling / A.3 λ） |

---

## 十三、待讨论事项

- [x] 评测指标：WER-clean / WER-other / 收敛步数（VoiceBench / URO-Bench 不保留）
- [x] 与 Züfle (arXiv:2412.15712) 的区分写法
- [x] 论文结构
- [ ] 时间规划更新
