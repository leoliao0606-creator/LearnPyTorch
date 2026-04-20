"""Reference solution for 05_01_capstone_tabular_classification.py."""

print("exercise 1:")
print("- LogisticRegression is simple, stable, fast, and often already very competitive on tabular data.")
print()
print("exercise 2:")
print("- The most likely problem is overfitting.")
print("- Next steps: stronger regularization, smaller hidden_dim, better feature processing, or earlier stopping.")
print()
print("exercise 3:")
print(
    [
        {"model": "LogisticRegression", "val_acc": 0.97, "test_acc": 0.98, "notes": "strong baseline"},
        {"model": "MLP-Small", "val_acc": 0.91, "test_acc": 0.90, "notes": "overfitting risk"},
        {"model": "MLP-Regularized", "val_acc": 0.95, "test_acc": 0.95, "notes": "better regularization"},
    ]
)
