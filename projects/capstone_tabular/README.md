# Capstone Tabular Project / 表格综合项目

这个目录把 `05_01_capstone_tabular_classification.ipynb` 中的核心流程整理成了一个可复现的小项目。  
This directory turns the core workflow from `05_01_capstone_tabular_classification.ipynb` into a reproducible mini-project.

## 目标 / Goal

- 跑一个表格二分类项目 / run a tabular binary classification project
- 比较经典 baseline 和 `MLP` / compare a classical baseline with an `MLP`
- 持久化实验结果 / persist experiment results

## 运行方式 / How to Run

在仓库根目录执行 / Run from the repository root:

```bash
/home/cliao/miniconda3/envs/olinml/bin/python projects/capstone_tabular/train.py
```

## 输出内容 / Outputs

运行后会生成：

- `projects/capstone_tabular/artifacts/config_used.json`
- `projects/capstone_tabular/artifacts/results.json`
- `projects/capstone_tabular/artifacts/histories.json`

## 当前范围 / Current Scope

- 数据集：`sklearn.datasets.load_breast_cancer`
- baseline：`LogisticRegression`
- 改进模型：两个 `TabularMLP` 变体
- 指标：验证集 / 测试集准确率、分类报告 / classification report、混淆矩阵 / confusion matrix

## 下一步 / Next Step

如果要把这个目录继续升级到更完整的项目，可以继续补：

- `train.py / evaluate.py / infer.py` 拆分
- 图表文件导出
- 多次实验汇总脚本
- 更明确的 `README` 实验结论区
