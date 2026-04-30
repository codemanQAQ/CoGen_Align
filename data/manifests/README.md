# Manifests（jsonl）

每行一个 JSON 对象，**Stage1/2 共用**时建议字段稳定、可复现。

## 推荐字段

| 字段 | 说明 |
|------|------|
| `id` | 全局唯一字符串 |
| `audio_path` | 原始 wav/flac（预处理或调试可用） |
| `feature_path` | 预计算 Whisper 特征 `.npy`（训练主路径） |
| `text` | 转写文本 |
| `duration` | 秒（推荐）；清单过滤 / 统计用。Whisper 尾截断在 **预计算** ``precompute_features_all.py`` 完成，训练 Dataset **不再** 按 `duration` 裁特征 |
| `hard_neg_ids` | 同 manifest 内其它样本 `id` 列表（可选） |

示例（路径请按本机修改）：

```json
{"id":"ls_0","audio_path":"/data/.../0.flac","feature_path":"/data/.../features/0.npy","text":"hello world","duration":1.2}
```

将 `train/train_*h.jsonl`、`all/train_all.jsonl` 等按实验切分生成后放在本目录，与 `configs/stage1/*.yaml`、`configs/stage2/*.yaml` 中的 `train_manifest` / `val_manifest` 对应。
