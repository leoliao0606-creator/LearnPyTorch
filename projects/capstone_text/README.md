# 文本分类综合项目

这个小项目在一个合成电影评论数据集上比较几种文本分类方案。

## 任务

- 二分类情感分析：`positive` / `negative`
- 项目结构和 `../capstone_tabular/` 保持一致

## 比较的模型

| 模型 | 类型 | 说明 |
| --- | --- | --- |
| Unigram-LR | sklearn | bag-of-words 上的逻辑回归 |
| Bigram-LR | sklearn | bigram 特征上的逻辑回归 |
| LSTM-Small | PyTorch (`src.models`) | `embed_dim=16`, `hidden_dim=32` |
| LSTM-Medium | PyTorch (`src.models`) | `embed_dim=32`, `hidden_dim=64` |

## 运行方式

```bash
cd projects/capstone_text
python train.py
```

运行结果会写到 `artifacts/`：

- `config_used.json`：本次运行的完整配置
- `results.json`：按效果排序的结果表和 confusion matrices
- `histories.json`：LSTM 模型逐 epoch 的 train / val 指标

## 如何扩展实验

修改 `config.json`，增加新的 LSTM 变体或调整超参数，然后重新运行 `train.py` 即可。
