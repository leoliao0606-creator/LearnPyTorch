from __future__ import annotations

import unittest

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from src.training import collect_predictions, run_classification_epoch


class TrainingHelperTests(unittest.TestCase):
    def test_run_epoch_trains_and_collects_predictions(self) -> None:
        x = torch.tensor(
            [
                [-2.0],
                [-1.0],
                [1.0],
                [2.0],
            ],
            dtype=torch.float32,
        )
        y = torch.tensor([0, 0, 1, 1], dtype=torch.long)
        loader = DataLoader(TensorDataset(x, y), batch_size=4, shuffle=False)
        torch.manual_seed(0)
        model = nn.Linear(1, 2)
        loss_fn = nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.2)

        for _ in range(30):
            train_loss, train_acc = run_classification_epoch(
                model,
                loader,
                loss_fn,
                optimizer=optimizer,
            )

        eval_loss, eval_acc = run_classification_epoch(model, loader, loss_fn)
        preds, targets = collect_predictions(model, loader)

        self.assertLess(train_loss, 0.8)
        self.assertGreaterEqual(train_acc, 0.75)
        self.assertLess(eval_loss, 0.8)
        self.assertGreaterEqual(eval_acc, 0.75)
        self.assertTrue(torch.equal(targets, y))
        self.assertEqual(preds.shape, y.shape)

    def test_empty_loader_raises_clear_error(self) -> None:
        x = torch.empty(0, 1)
        y = torch.empty(0, dtype=torch.long)
        loader = DataLoader(TensorDataset(x, y), batch_size=4)
        model = nn.Linear(1, 2)
        loss_fn = nn.CrossEntropyLoss()

        with self.assertRaisesRegex(ValueError, "loader did not yield"):
            run_classification_epoch(model, loader, loss_fn)
        with self.assertRaisesRegex(ValueError, "loader did not yield"):
            collect_predictions(model, loader)


if __name__ == "__main__":
    unittest.main()
