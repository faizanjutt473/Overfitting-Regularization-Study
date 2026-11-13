"""
Overfitting and Regularization Study
====================================

A controlled experiment that:
  1. Deliberately overfits a large MLP on a small MNIST subset (baseline).
  2. Applies regularization remedies ONE AT A TIME to the same baseline:
        L2 (weight decay), L1, Dropout, Early Stopping,
        Data Augmentation, Simpler Architecture.
  3. Runs one combined model (optional bonus) using the best ideas together.
  4. Saves training/validation curves, comparison plots, a metrics table
     and raw histories.

Everything except the single remedy under test is held constant:
same data split, same seed (so same initial weights), same optimizer,
same learning rate, same batch size and same number of max epochs.

Usage:
    python main.py
    python main.py --epochs 100 --train-size 1000 --seed 42
"""

from __future__ import annotations

import argparse
import copy
import json
import random
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib

matplotlib.use("Agg")  # no GUI needed; plots are saved to disk
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms

# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #
ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
PLOTS_DIR = ROOT / "plots"
RESULTS_DIR = ROOT / "results"

MNIST_MEAN, MNIST_STD = 0.1307, 0.3081


# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
@dataclass
class Settings:
    """Global settings shared by every experiment."""

    seed: int = 42
    train_size: int = 1000
    val_size: int = 2000
    batch_size: int = 64
    epochs: int = 100
    lr: float = 1e-3
    patience: int = 10  # early stopping patience (epochs)
    min_delta: float = 1e-4  # minimum val-loss improvement to reset patience


@dataclass
class Experiment:
    """One experiment = baseline + (at most) one change."""

    name: str
    title: str
    hidden_dims: Tuple[int, ...] = (512, 512, 256)
    dropout: float = 0.0
    weight_decay: float = 0.0  # L2 regularization (via optimizer)
    l1_lambda: float = 0.0  # L1 regularization (added to loss)
    augment: bool = False
    early_stopping: bool = False
    description: str = field(default="", repr=False)


def build_experiments() -> List[Experiment]:
    """Baseline first, then each remedy applied alone, then a combined run."""
    return [
        Experiment(
            name="baseline",
            title="Baseline (no regularization)",
            description="Large MLP on a tiny dataset; expected to overfit.",
        ),
        Experiment(
            name="l2",
            title="Remedy 1: L2 regularization (weight decay)",
            weight_decay=5e-4,
            description="Penalizes large weights.",
        ),
        Experiment(
            name="l1",
            title="Remedy 2: L1 regularization",
            l1_lambda=5e-5,
            description="Penalizes |w|; pushes many weights to zero.",
        ),
        Experiment(
            name="dropout",
            title="Remedy 3: Dropout (p=0.5)",
            dropout=0.5,
            description="Randomly drops units so the net cannot co-adapt.",
        ),
        Experiment(
            name="early_stopping",
            title="Remedy 4: Early stopping",
            early_stopping=True,
            description="Stops when validation loss stops improving.",
        ),
        Experiment(
            name="augmentation",
            title="Remedy 5: Data augmentation",
            augment=True,
            description="Random rotation, shift and scale of training images.",
        ),
        Experiment(
            name="simpler_model",
            title="Remedy 6: Simpler architecture",
            hidden_dims=(64,),
            description="One small hidden layer instead of three large ones.",
        ),
        Experiment(
            name="combined",
            title="Bonus: Combined (L2 + Dropout + Augmentation + Early stopping)",
            weight_decay=5e-4,
            dropout=0.3,
            augment=True,
            early_stopping=True,
            description="Several remedies together.",
        ),
    ]


# --------------------------------------------------------------------------- #
# Reproducibility
# --------------------------------------------------------------------------- #
def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# --------------------------------------------------------------------------- #
# Data
# --------------------------------------------------------------------------- #
def normalize(images_uint8: torch.Tensor) -> torch.Tensor:
    """uint8 [N,28,28] -> normalized float [N,1,28,28]."""
    x = images_uint8.float().div(255.0)
    x = (x - MNIST_MEAN) / MNIST_STD
    return x.unsqueeze(1)


class AugmentableDataset(Dataset):
    """Training dataset with optional on-the-fly augmentation.

    Augmentation is applied on the raw [0, 1] image (background = 0) and
    normalization is applied afterwards, so rotated/shifted corners stay
    background-colored.
    """

    def __init__(self, images_uint8: torch.Tensor, labels: torch.Tensor, augment: bool):
        self.images = images_uint8.float().div(255.0).unsqueeze(1)  # [N,1,28,28]
        self.labels = labels.long()
        self.augment = augment
        self.aug = transforms.RandomAffine(
            degrees=15, translate=(0.1, 0.1), scale=(0.9, 1.1)
        )

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int):
        x = self.images[idx]
        if self.augment:
            x = self.aug(x)
        x = (x - MNIST_MEAN) / MNIST_STD
        return x, self.labels[idx]


def load_data(settings: Settings, device: torch.device) -> Dict[str, torch.Tensor]:
    """Create fixed train / validation / test splits (identical for all runs)."""
    train_full = datasets.MNIST(root=str(DATA_DIR), train=True, download=True)
    test_full = datasets.MNIST(root=str(DATA_DIR), train=False, download=True)

    needed = settings.train_size + settings.val_size
    if needed > len(train_full):
        raise ValueError("train_size + val_size exceeds the MNIST training set size.")

    generator = torch.Generator().manual_seed(settings.seed)
    perm = torch.randperm(len(train_full), generator=generator)
    train_idx = perm[: settings.train_size]
    val_idx = perm[settings.train_size : needed]

    x_train_raw = train_full.data[train_idx]
    y_train = train_full.targets[train_idx]
    x_val_raw = train_full.data[val_idx]
    y_val = train_full.targets[val_idx]

    return {
        "x_train_raw": x_train_raw,  # uint8, used by the (augmentable) loader
        "y_train": y_train,
        "x_train": normalize(x_train_raw).to(device),  # clean copy for evaluation
        "y_train_dev": y_train.to(device),
        "x_val": normalize(x_val_raw).to(device),
        "y_val": y_val.to(device),
        "x_test": normalize(test_full.data).to(device),
        "y_test": test_full.targets.to(device),
    }


# --------------------------------------------------------------------------- #
# Model
# --------------------------------------------------------------------------- #
class MLP(nn.Module):
    """Fully connected network with optional dropout after each hidden layer."""

    def __init__(
        self,
        hidden_dims: Tuple[int, ...],
        dropout: float = 0.0,
        in_features: int = 28 * 28,
        n_classes: int = 10,
    ):
        super().__init__()
        layers: List[nn.Module] = [nn.Flatten()]
        prev = in_features
        for h in hidden_dims:
            layers += [nn.Linear(prev, h), nn.ReLU()]
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev = h
        layers.append(nn.Linear(prev, n_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def weight_sparsity(model: nn.Module, threshold: float = 1e-3) -> float:
    """Percentage of weights (not biases) with |w| < threshold."""
    total, small = 0, 0
    for p in model.parameters():
        if p.ndim > 1:
            total += p.numel()
            small += (p.detach().abs() < threshold).sum().item()
    return 100.0 * small / max(total, 1)


def l1_penalty(model: nn.Module) -> torch.Tensor:
    return sum(p.abs().sum() for p in model.parameters() if p.ndim > 1)


# --------------------------------------------------------------------------- #
# Training / evaluation
# --------------------------------------------------------------------------- #
@torch.no_grad()
def evaluate(
    model: nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    criterion: nn.Module,
    batch_size: int = 2048,
) -> Tuple[float, float]:
    """Return (cross-entropy loss, accuracy) in eval mode (dropout off)."""
    model.eval()
    total_loss, correct = 0.0, 0
    for i in range(0, len(x), batch_size):
        xb, yb = x[i : i + batch_size], y[i : i + batch_size]
        out = model(xb)
        total_loss += criterion(out, yb).item() * len(xb)
        correct += (out.argmax(dim=1) == yb).sum().item()
    return total_loss / len(x), correct / len(x)


def run_experiment(
    exp: Experiment,
    settings: Settings,
    data: Dict[str, torch.Tensor],
    device: torch.device,
) -> Tuple[Dict[str, List[float]], Dict[str, float]]:
    """Train one experiment and return (history, final metrics)."""
    set_seed(settings.seed)  # identical initial weights for every experiment

    model = MLP(exp.hidden_dims, exp.dropout).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=settings.lr, weight_decay=exp.weight_decay
    )
    criterion = nn.CrossEntropyLoss()

    train_ds = AugmentableDataset(data["x_train_raw"], data["y_train"], exp.augment)
    loader = DataLoader(
        train_ds,
        batch_size=settings.batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(settings.seed),
        num_workers=0,
    )

    history: Dict[str, List[float]] = {
        "train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []
    }
    best_val_loss = float("inf")
    best_state = None
    best_epoch = 0
    wait = 0
    stopped_epoch = settings.epochs

    start = time.time()
    for epoch in range(1, settings.epochs + 1):
        model.train()
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            if exp.l1_lambda > 0:
                loss = loss + exp.l1_lambda * l1_penalty(model)
            loss.backward()
            optimizer.step()

        # Curves are measured on CLEAN data in eval mode so that every
        # experiment is compared on exactly the same footing.
        tr_loss, tr_acc = evaluate(model, data["x_train"], data["y_train_dev"], criterion)
        va_loss, va_acc = evaluate(model, data["x_val"], data["y_val"], criterion)
        history["train_loss"].append(tr_loss)
        history["train_acc"].append(tr_acc)
        history["val_loss"].append(va_loss)
        history["val_acc"].append(va_acc)

        if epoch == 1 or epoch % 10 == 0:
            print(
                f"  epoch {epoch:3d}/{settings.epochs} | "
                f"train loss {tr_loss:.4f} acc {tr_acc:.3f} | "
                f"val loss {va_loss:.4f} acc {va_acc:.3f}"
            )

        if exp.early_stopping:
            if va_loss < best_val_loss - settings.min_delta:
                best_val_loss = va_loss
                best_state = copy.deepcopy(model.state_dict())
                best_epoch = epoch
                wait = 0
            else:
                wait += 1
                if wait >= settings.patience:
                    stopped_epoch = epoch
                    print(f"  early stopping at epoch {epoch} (best epoch {best_epoch})")
                    break

    if exp.early_stopping and best_state is not None:
        model.load_state_dict(best_state)  # restore best weights

    # Final metrics of the model that would actually be deployed.
    criterion_eval = nn.CrossEntropyLoss()
    f_tr_loss, f_tr_acc = evaluate(model, data["x_train"], data["y_train_dev"], criterion_eval)
    f_va_loss, f_va_acc = evaluate(model, data["x_val"], data["y_val"], criterion_eval)
    f_te_loss, f_te_acc = evaluate(model, data["x_test"], data["y_test"], criterion_eval)

    min_val_epoch = int(np.argmin(history["val_loss"])) + 1
    metrics = {
        "experiment": exp.name,
        "epochs_run": len(history["val_loss"]),
        "stopped_epoch": stopped_epoch if exp.early_stopping else len(history["val_loss"]),
        "best_val_loss_epoch": best_epoch if exp.early_stopping else min_val_epoch,
        "train_acc": f_tr_acc,
        "val_acc": f_va_acc,
        "test_acc": f_te_acc,
        "train_loss": f_tr_loss,
        "val_loss": f_va_loss,
        "test_loss": f_te_loss,
        "acc_gap_train_minus_val": f_tr_acc - f_va_acc,
        "loss_gap_val_minus_train": f_va_loss - f_tr_loss,
        "min_val_loss": float(min(history["val_loss"])),
        "weights_near_zero_pct": weight_sparsity(model),
        "parameters": count_parameters(model),
        "train_time_sec": time.time() - start,
    }
    return history, metrics


# --------------------------------------------------------------------------- #
# Plotting
# --------------------------------------------------------------------------- #
def plot_experiment(
    exp: Experiment,
    history: Dict[str, List[float]],
    metrics: Dict[str, float],
    baseline_history: Dict[str, List[float]],
    path: Path,
) -> None:
    """Training vs validation loss and accuracy for one experiment."""
    epochs = range(1, len(history["train_loss"]) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))

    for ax, key, ylabel in (
        (axes[0], "loss", "Cross-entropy loss"),
        (axes[1], "acc", "Accuracy"),
    ):
        ax.plot(epochs, history[f"train_{key}"], label="Train", color="tab:blue", lw=2)
        ax.plot(epochs, history[f"val_{key}"], label="Validation", color="tab:orange", lw=2)
        if exp.name != "baseline":
            b_epochs = range(1, len(baseline_history[f"val_{key}"]) + 1)
            ax.plot(
                b_epochs, baseline_history[f"val_{key}"],
                label="Baseline validation", color="gray", ls="--", lw=1.5, alpha=0.8,
            )
        if exp.early_stopping:
            ax.axvline(
                metrics["best_val_loss_epoch"], color="green", ls=":", lw=1.5,
                label=f"Best epoch ({int(metrics['best_val_loss_epoch'])})",
            )
        ax.set_xlabel("Epoch")
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.3)
        ax.legend()

    axes[0].set_title("Loss")
    axes[1].set_title("Accuracy")
    fig.suptitle(
        f"{exp.title}\n"
        f"train acc {metrics['train_acc']:.3f} | val acc {metrics['val_acc']:.3f} | "
        f"test acc {metrics['test_acc']:.3f}",
        fontsize=12,
    )
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_comparison(
    experiments: List[Experiment],
    histories: Dict[str, Dict[str, List[float]]],
    metrics_df: pd.DataFrame,
) -> None:
    """Overlay validation curves and compare final metrics of all runs."""
    cmap = plt.get_cmap("tab10")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for i, exp in enumerate(experiments):
        h = histories[exp.name]
        ep = range(1, len(h["val_loss"]) + 1)
        style = "--" if exp.name == "baseline" else "-"
        axes[0].plot(ep, h["val_loss"], style, label=exp.name, color=cmap(i), lw=2)
        axes[1].plot(ep, h["val_acc"], style, label=exp.name, color=cmap(i), lw=2)
    axes[0].set_title("Validation loss")
    axes[1].set_title("Validation accuracy")
    for ax, yl in zip(axes, ("Loss", "Accuracy")):
        ax.set_xlabel("Epoch")
        ax.set_ylabel(yl)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "comparison_validation_curves.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    names = metrics_df["experiment"].tolist()
    colors = [cmap(i) for i in range(len(names))]
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    bars = axes[0].bar(names, metrics_df["test_acc"] * 100, color=colors)
    axes[0].set_title("Test accuracy (%)")
    axes[0].set_ylim(max(0, metrics_df["test_acc"].min() * 100 - 10), 100)
    axes[0].bar_label(bars, fmt="%.1f", fontsize=8)

    bars = axes[1].bar(names, metrics_df["acc_gap_train_minus_val"] * 100, color=colors)
    axes[1].set_title("Generalization gap: train acc - val acc (points)")
    axes[1].bar_label(bars, fmt="%.1f", fontsize=8)

    for ax in axes:
        ax.tick_params(axis="x", rotation=40)
        ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "comparison_final_metrics.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def parse_args() -> Settings:
    parser = argparse.ArgumentParser(description="Overfitting and Regularization Study")
    defaults = Settings()
    parser.add_argument("--seed", type=int, default=defaults.seed)
    parser.add_argument("--train-size", type=int, default=defaults.train_size)
    parser.add_argument("--val-size", type=int, default=defaults.val_size)
    parser.add_argument("--batch-size", type=int, default=defaults.batch_size)
    parser.add_argument("--epochs", type=int, default=defaults.epochs)
    parser.add_argument("--lr", type=float, default=defaults.lr)
    parser.add_argument("--patience", type=int, default=defaults.patience)
    a = parser.parse_args()
    return Settings(
        seed=a.seed, train_size=a.train_size, val_size=a.val_size,
        batch_size=a.batch_size, epochs=a.epochs, lr=a.lr, patience=a.patience,
    )


def main() -> None:
    settings = parse_args()
    PLOTS_DIR.mkdir(exist_ok=True)
    RESULTS_DIR.mkdir(exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device} | Settings: {settings}")

    set_seed(settings.seed)
    data = load_data(settings, device)
    print(
        f"Data: train={len(data['y_train'])}, val={len(data['y_val'])}, "
        f"test={len(data['y_test'])}\n"
    )

    experiments = build_experiments()
    histories: Dict[str, Dict[str, List[float]]] = {}
    all_metrics: List[Dict[str, float]] = []

    for idx, exp in enumerate(experiments):
        print(f"[{idx + 1}/{len(experiments)}] {exp.title}")
        history, metrics = run_experiment(exp, settings, data, device)
        histories[exp.name] = history
        all_metrics.append(metrics)
        plot_experiment(
            exp, history, metrics, histories["baseline"],
            PLOTS_DIR / f"{idx:02d}_{exp.name}.png",
        )
        print(
            f"  -> train acc {metrics['train_acc']:.4f} | val acc {metrics['val_acc']:.4f} "
            f"| test acc {metrics['test_acc']:.4f}\n"
        )

    metrics_df = pd.DataFrame(all_metrics)
    metrics_df.round(4).to_csv(RESULTS_DIR / "metrics.csv", index=False)
    with open(RESULTS_DIR / "histories.json", "w", encoding="utf-8") as f:
        json.dump(histories, f)

    plot_comparison(experiments, histories, metrics_df)

    summary_cols = [
        "experiment", "epochs_run", "train_acc", "val_acc", "test_acc",
        "acc_gap_train_minus_val", "min_val_loss", "weights_near_zero_pct", "parameters",
    ]
    print("=" * 100)
    print("SUMMARY")
    print("=" * 100)
    print(metrics_df[summary_cols].round(4).to_string(index=False))
    print(f"\nPlots saved in:   {PLOTS_DIR}")
    print(f"Metrics saved in: {RESULTS_DIR / 'metrics.csv'}")


if __name__ == "__main__":
    main()
