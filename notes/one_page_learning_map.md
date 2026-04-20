# 一页学习地图

## PyTorch 核心数据流

`Tensor -> Dataset -> DataLoader -> Model -> Loss -> Optimizer -> Training Loop -> Metrics -> Analysis`

更具体地说：

1. `Tensor` 负责承载数值、shape、dtype 和 device。
2. `Dataset` 负责定义“如何取一个样本”。
3. `DataLoader` 负责 batching、shuffle 和迭代。
4. `Model` 负责把输入变成预测。
5. `Loss` 负责定义“错了多少”。
6. `Optimizer` 负责根据梯度更新参数。
7. `Training Loop` 负责按正确顺序把这些模块串起来。
8. `Metrics and Analysis` 负责判断模型是不是真的变好了。

## 三类常见模型的直觉

| 模型族 | 典型输入 | 核心结构 | 适合场景 |
| --- | --- | --- | --- |
| MLP | 表格特征 | 多层线性层堆叠 | 中小型表格任务 |
| CNN | 图像 | 卷积加 pooling | 视觉任务 |
| LSTM / Transformer | token 序列或时间序列 | 循环状态或 self-attention | 文本和序列建模 |

## 默认调试顺序

1. shape
2. dtype
3. device
4. loss 是否有限
5. 梯度是否正常
6. 是否存在数据泄漏

## 一个重要提醒

`Notebook 能跑通，不等于知识已经内化。`

真正的掌握通常至少要经过三步：

1. 跟着做一遍
2. 不看提示重写一遍
3. 自己改动并解释结果
