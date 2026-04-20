# One-Page Learning Map / 一页纸学习地图

## 1. PyTorch 最核心的数据流 / The Core PyTorch Data Flow

`Tensor -> Dataset -> DataLoader -> Model -> Loss -> Optimizer -> Training Loop -> Metrics -> Analysis`

更具体一点：

1. `Tensor / 张量`  
   负责承载数值和 shape。
2. `Dataset / 数据集对象`  
   负责定义“如何取一个样本”。
3. `DataLoader / 数据加载器`  
   负责 batching、shuffle、迭代。
4. `Model / 模型`  
   负责把输入映射到输出 logits / predictions。
5. `Loss / 损失函数`  
   负责定义“错了多少”。
6. `Optimizer / 优化器`  
   负责根据梯度更新参数。
7. `Training Loop / 训练循环`  
   负责把上面这些东西按正确顺序连起来。
8. `Metrics and Analysis / 指标与分析`  
   负责判断模型是否真的变好了。

## 2. 三类模型的直觉比较 / Intuition for Three Model Families

| 模型 / Model | 最擅长什么 / Good At | 最常见输入 / Typical Input | 关键结构 / Core Structure |
|---|---|---|---|
| `CNN` | 局部空间模式 / local spatial patterns | 图像 / image | 卷积 / pooling |
| `LSTM` | 顺序依赖 / sequential dependence | token 序列、时间序列 / token sequences, time series | recurrent state |
| `Transformer` | 全局依赖 / global dependencies | 文本、序列 / text, sequence | self-attention |

## 3. 最常见的排错顺序 / Default Debug Order

1. shape
2. dtype
3. device
4. loss 是否有限 / whether loss is finite
5. gradient 是否正常 / whether gradients look normal
6. 数据是否泄漏 / whether there is data leakage

## 4. 一个重要提醒 / One Important Reminder

`notebook 能跑通 != 知识已经内化`  
`A notebook running successfully does not mean the knowledge is internalized.`

真正的掌握通常要经过三步：

1. 跟着做 / follow
2. 关提示重写 / rewrite without hints
3. 独立改动并解释结果 / modify independently and explain the result
