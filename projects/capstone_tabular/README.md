# 表格综合项目

这个目录把 `05_01_capstone_tabular_classification.ipynb` 里的核心流程整理成了一个可复现的小项目。

## 目标

- 跑一个表格二分类项目
- 比较经典 baseline 和 `MLP`
- 持久化实验结果

## 运行方式

在仓库根目录执行：

```bash
/home/cliao/miniconda3/envs/olinml/bin/python projects/capstone_tabular/train.py
```

## 输出内容

运行后会生成：

- `projects/capstone_tabular/artifacts/config_used.json`
- `projects/capstone_tabular/artifacts/results.json`
- `projects/capstone_tabular/artifacts/histories.json`

## 当前范围

- 数据集：`sklearn.datasets.load_breast_cancer`
- baseline：`LogisticRegression`
- 改进模型：两个 `TabularMLP` 变体
- 指标：验证集准确率、测试集准确率、classification report 和 confusion matrix

## 下一步

如果要把这个目录继续升级成更完整的项目，可以继续补：

- 更清晰的 `train.py` / `evaluate.py` / `infer.py` 拆分
- 图表文件导出
- 多次实验汇总脚本
- 更明确的 `README` 结论区
