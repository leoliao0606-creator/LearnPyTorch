# One-Page Learning Map

## The Core PyTorch Data Flow

`Tensor -> Dataset -> DataLoader -> Model -> Loss -> Optimizer -> Training Loop -> Metrics -> Analysis`

More concretely:

1. `Tensor` carries values, shapes, dtypes, and devices.
2. `Dataset` defines how to get one sample.
3. `DataLoader` handles batching, shuffling, and iteration.
4. `Model` turns inputs into predictions.
5. `Loss` defines how wrong the predictions are.
6. `Optimizer` updates parameters with gradients.
7. `Training Loop` connects all of the above in the right order.
8. `Metrics and Analysis` tell you whether the model is actually improving.

## Intuition for Three Model Families

| Family | Typical Input | Core Structure | Best Use Case |
| --- | --- | --- | --- |
| MLP | tabular features | stacked linear layers | small and medium tabular tasks |
| CNN | images | convolution plus pooling | vision tasks with local spatial patterns |
| LSTM / Transformer | token sequences or time series | recurrent state or self-attention | text and sequence modeling |

## Default Debug Order

1. shape
2. dtype
3. device
4. whether the loss is finite
5. whether gradients look normal
6. whether there is data leakage

## One Important Reminder

`A notebook running successfully does not mean the knowledge is internalized.`

Real mastery usually requires three steps:

1. follow
2. rewrite without hints
3. modify independently and explain the result
