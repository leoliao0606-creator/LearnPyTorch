from __future__ import annotations

import unittest

import numpy as np
import torch

from src.tabular_data import load_breast_cancer_splits, make_tabular_loaders
from src.text_data import PAD_TOKEN, UNK_TOKEN, build_vocab, encode_text, make_text_loaders
from src.utils import resolve_device


class TabularDataTests(unittest.TestCase):
    def test_breast_cancer_splits_are_scaled_and_complete(self) -> None:
        split_data = load_breast_cancer_splits(seed=123)

        total_rows = (
            len(split_data["y_train"])
            + len(split_data["y_val"])
            + len(split_data["y_test"])
        )
        self.assertEqual(total_rows, 569)
        self.assertEqual(split_data["x_train"].shape[1], len(split_data["feature_names"]))
        np.testing.assert_allclose(split_data["x_train"].mean(axis=0), 0.0, atol=1e-6)
        np.testing.assert_allclose(split_data["x_train"].std(axis=0), 1.0, atol=1e-6)

    def test_tabular_loader_shuffle_is_reproducible_with_seed(self) -> None:
        split_data = load_breast_cancer_splits(seed=42)
        loaders_a = make_tabular_loaders(split_data, train_batch_size=8, seed=7)
        loaders_b = make_tabular_loaders(split_data, train_batch_size=8, seed=7)

        batch_a = next(iter(loaders_a["train"]))
        batch_b = next(iter(loaders_b["train"]))

        self.assertTrue(torch.equal(batch_a[0], batch_b[0]))
        self.assertTrue(torch.equal(batch_a[1], batch_b[1]))


class TextDataTests(unittest.TestCase):
    def test_vocab_min_freq_unknown_and_padding(self) -> None:
        _, stoi = build_vocab(["good movie", "bad movie"], min_freq=2)

        self.assertIn(PAD_TOKEN, stoi)
        self.assertIn(UNK_TOKEN, stoi)
        self.assertIn("movie", stoi)
        self.assertNotIn("good", stoi)

        encoded = encode_text("good movie", stoi, max_len=4)
        self.assertEqual(encoded[0], stoi[UNK_TOKEN])
        self.assertEqual(encoded[1], stoi["movie"])
        self.assertEqual(encoded[2:], [stoi[PAD_TOKEN], stoi[PAD_TOKEN]])

    def test_text_loader_shuffle_is_reproducible_with_seed(self) -> None:
        texts = ["good movie", "bad movie", "nice show", "weak plot"]
        labels = [1, 0, 1, 0]
        _, stoi = build_vocab(texts)

        loaders_a = make_text_loaders(
            texts,
            labels,
            texts,
            labels,
            texts,
            labels,
            stoi,
            max_len=3,
            train_batch_size=2,
            seed=99,
        )
        loaders_b = make_text_loaders(
            texts,
            labels,
            texts,
            labels,
            texts,
            labels,
            stoi,
            max_len=3,
            train_batch_size=2,
            seed=99,
        )

        batch_a = next(iter(loaders_a["train"]))
        batch_b = next(iter(loaders_b["train"]))

        self.assertTrue(torch.equal(batch_a[0], batch_b[0]))
        self.assertTrue(torch.equal(batch_a[1], batch_b[1]))


class DeviceTests(unittest.TestCase):
    def test_resolve_cpu_device(self) -> None:
        self.assertEqual(resolve_device("cpu"), torch.device("cpu"))


if __name__ == "__main__":
    unittest.main()
