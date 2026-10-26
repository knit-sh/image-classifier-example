from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .model import CifarCNN, count_parameters


def _select_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def save_checkpoint(
    path: Path,
    model: CifarCNN,
    config: dict,
    dataset: str,
    class_names: list[str],
    epoch: int,
) -> None:
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "model_config": config,
            "dataset": dataset,
            "class_names": class_names,
        },
        path,
    )


def run_train(
    args: argparse.Namespace,
    model: CifarCNN,
    config: dict,
    train_loader: DataLoader,
    val_loader: DataLoader,
    class_names: list[str],
) -> None:
    device = _select_device()
    print(f"Device      : {device}")
    print(f"Parameters  : {count_parameters(model):,}")

    model = model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    criterion = nn.CrossEntropyLoss()

    model_path = Path(args.model)
    model_path.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        # --- training ---
        model.train()
        train_loss = train_correct = train_total = 0

        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            logits = model(inputs)
            loss = criterion(logits, targets)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * inputs.size(0)
            train_correct += (logits.argmax(1) == targets).sum().item()
            train_total += inputs.size(0)

        scheduler.step()

        # --- validation ---
        model.eval()
        val_loss = val_correct = val_total = 0

        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                logits = model(inputs)
                val_loss += criterion(logits, targets).item() * inputs.size(0)
                val_correct += (logits.argmax(1) == targets).sum().item()
                val_total += inputs.size(0)

        print(
            f"Epoch {epoch:3d}/{args.epochs}"
            f" | train loss {train_loss / train_total:.4f}"
            f"  acc {100 * train_correct / train_total:5.1f}%"
            f" | val loss {val_loss / val_total:.4f}"
            f"  acc {100 * val_correct / val_total:5.1f}%"
        )

        if args.checkpoint_every and epoch % args.checkpoint_every == 0:
            ckpt = model_path.with_name(
                f"{model_path.stem}_epoch{epoch:04d}{model_path.suffix}"
            )
            save_checkpoint(ckpt, model, config, args.dataset, class_names, epoch)
            print(f"  checkpoint -> {ckpt}")

    save_checkpoint(model_path, model, config, args.dataset, class_names, args.epochs)
    print(f"Model saved -> {model_path}")

    # Machine-readable final metric. The last epoch's validation counts are still
    # in scope here. Automated tools read this line to get the objective.
    print(f"RESULT val_accuracy={100 * val_correct / val_total:.4f}")
