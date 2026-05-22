# LearnPyTorch

一个以 notebook 为主、以小项目收尾的 PyTorch 学习仓库。当前结构覆盖基础 Python/NumPy/Pandas/sklearn、PyTorch 核心训练流程、CNN、序列模型、工程化调试，以及两个可复现实验型 capstone。

## Structure

- `notebooks/`: 主学习材料，按阶段排列。
- `exercises/`: 从 notebook 抽出的独立练习和参考答案。
- `src/`: notebook 和项目共用的模型、数据处理、训练与工具函数。
- `projects/capstone_tabular/`: 乳腺癌 tabular 分类，对比 sklearn baseline 与 PyTorch MLP。
- `projects/capstone_text/`: 合成文本情感分类，对比 bag-of-words baseline 与 PyTorch LSTM。
- `notes/`: 错误日志、阶段复盘和速查笔记。

## Run

本仓库当前依赖在 `olinml` conda 环境里可用：

```bash
/home/cliao/miniconda3/envs/olinml/bin/python projects/capstone_tabular/train.py
/home/cliao/miniconda3/envs/olinml/bin/python projects/capstone_text/train.py
```

两个 capstone 都从各自的 `config.json` 读取 `seed`、`device` 和 batch/epoch 设置。`device` 可设为 `"auto"`、`"cpu"` 或可用的加速设备；`use_amp` 只会在 CUDA 上启用。

## Verify

核心可复用代码用标准库 `unittest` 覆盖，不需要额外安装 pytest：

```bash
/home/cliao/miniconda3/envs/olinml/bin/python -m unittest discover -s tests
```

这些测试重点检查数据划分与标准化、DataLoader shuffle 可复现性、文本编码、训练循环和空 loader 错误处理。
