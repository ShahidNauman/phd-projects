"""
Training script for Transformer-based sentiment classification.

Usage:
    python train.py [--epochs N] [--lr LR] [--batch-size BS]

The script trains a TransformerClassifier on the built-in movie-review
dataset, prints per-epoch metrics, and saves the best checkpoint to
``best_model.pt`` in the current working directory.
"""

from __future__ import annotations

import argparse
import time

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from dataset import get_dataloaders
from transformer import TransformerClassifier


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def accuracy(logits: torch.Tensor, labels: torch.Tensor) -> float:
    """Fraction of correct predictions."""
    preds = logits.argmax(dim=-1)
    return (preds == labels).float().mean().item()


def run_epoch(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
    train: bool = True,
) -> tuple[float, float]:
    """Run one epoch of training or evaluation.

    Args:
        model: The TransformerClassifier.
        loader: DataLoader yielding (input_ids, padding_mask, labels).
        criterion: Loss function (CrossEntropyLoss).
        optimizer: Optimizer — None during evaluation.
        device: CPU or CUDA device.
        train: If True, perform gradient updates.

    Returns:
        Tuple of (mean_loss, mean_accuracy) for the epoch.
    """
    model.train(train)
    total_loss = 0.0
    total_acc = 0.0

    with torch.set_grad_enabled(train):
        for input_ids, padding_mask, labels in loader:
            input_ids = input_ids.to(device)
            padding_mask = padding_mask.to(device)
            labels = labels.to(device)

            logits = model(input_ids, padding_mask)
            loss = criterion(logits, labels)

            if train and optimizer is not None:
                optimizer.zero_grad()
                loss.backward()
                # Gradient clipping stabilizes Transformer training.
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

            total_loss += loss.item()
            total_acc += accuracy(logits, labels)

    n = len(loader)
    return total_loss / n, total_acc / n


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train Transformer sentiment classifier"
    )
    parser.add_argument(
        "--epochs", type=int, default=30, help="Number of training epochs"
    )
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size")
    parser.add_argument(
        "--embed-dim", type=int, default=128, help="Model embedding dimension"
    )
    parser.add_argument(
        "--num-heads", type=int, default=4, help="Number of attention heads"
    )
    parser.add_argument(
        "--num-layers", type=int, default=2, help="Number of encoder layers"
    )
    parser.add_argument(
        "--ff-dim", type=int, default=256, help="Feed-forward inner dimension"
    )
    parser.add_argument(
        "--max-len", type=int, default=64, help="Maximum sequence length"
    )
    parser.add_argument(
        "--dropout", type=float, default=0.1, help="Dropout probability"
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    return parser.parse_args()


def train(args: argparse.Namespace) -> None:
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}\n")

    # ---- Data ---------------------------------------------------------------
    train_loader, val_loader, tokenizer = get_dataloaders(
        batch_size=args.batch_size,
        max_len=args.max_len,
    )
    print(f"Vocabulary size : {tokenizer.vocab_size}")
    print(f"Training samples: {len(train_loader.dataset)}")  # type: ignore
    print(f"Validation samples: {len(val_loader.dataset)}\n")  # type: ignore

    # ---- Model --------------------------------------------------------------
    model = TransformerClassifier(
        vocab_size=tokenizer.vocab_size,
        num_classes=2,
        embed_dim=args.embed_dim,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        ff_dim=args.ff_dim,
        max_len=args.max_len,
        dropout=args.dropout,
    ).to(device)

    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {num_params:,}\n")

    # ---- Optimisation -------------------------------------------------------
    criterion = nn.CrossEntropyLoss()
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=1e-2)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)

    # ---- Training loop ------------------------------------------------------
    best_val_acc = 0.0
    print(
        f"{'Epoch':>6} | {'Train Loss':>10} | {'Train Acc':>9} | {'Val Loss':>8} | {'Val Acc':>7} | {'Time':>6}"
    )
    print("-" * 65)

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_loss, train_acc = run_epoch(
            model, train_loader, criterion, optimizer, device, train=True
        )
        val_loss, val_acc = run_epoch(
            model, val_loader, criterion, None, device, train=False
        )
        scheduler.step()
        elapsed = time.time() - t0

        marker = " ✓" if val_acc > best_val_acc else ""
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), "best_model.pt")

        print(
            f"{epoch:>6} | {train_loss:>10.4f} | {train_acc:>9.4f} | "
            f"{val_loss:>8.4f} | {val_acc:>7.4f} | {elapsed:>5.1f}s{marker}"
        )

    print(f"\nBest validation accuracy: {best_val_acc:.4f}")
    print("Model checkpoint saved to best_model.pt")


if __name__ == "__main__":
    train(parse_args())
