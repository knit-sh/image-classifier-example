from __future__ import annotations

import argparse
import sys

import torch

from .classify import run_classify
from .data import get_loaders
from .model import build_model
from .train import run_train


def _add_train_subparser(subparsers) -> None:
    p = subparsers.add_parser("train", help="Train a CNN on CIFAR-10 or CIFAR-100")

    p.add_argument("--model", required=True, metavar="PATH",
                   help="file path where the trained model will be saved")
    p.add_argument("--dataset", choices=["cifar10", "cifar100"], default="cifar10",
                   help="dataset to use (default: cifar10)")
    p.add_argument("--data", required=True, metavar="DIR",
                   help="directory holding the extracted dataset (see README for download); "
                        "for cifar10 it must contain cifar-10-batches-py/, "
                        "for cifar100 it must contain cifar-100-python/")

    arch = p.add_argument_group("architecture")
    arch.add_argument("--num-blocks", type=int, default=3, metavar="N",
                      help="number of conv blocks; each block halves spatial dims and doubles channels (default: 3, max: 5)")
    arch.add_argument("--base-channels", type=int, default=32, metavar="N",
                      help="output channels of the first block, doubled each subsequent block (default: 32)")
    arch.add_argument("--kernel-size", type=int, default=3, choices=[3, 5],
                      help="conv kernel size — 3 for a tighter receptive field, 5 for a wider one (default: 3)")
    arch.add_argument("--dropout", type=float, default=0.3, metavar="P",
                      help="dropout probability applied before the final linear layer (default: 0.3)")
    arch.add_argument("--residual", action="store_true",
                      help="add residual skip connections between blocks (ResNet-style)")
    arch.add_argument("--activation", choices=["relu", "gelu", "leaky-relu"], default="relu",
                      help="activation function used throughout the network (default: relu)")

    trn = p.add_argument_group("training")
    trn.add_argument("--epochs", type=int, default=20, metavar="N",
                     help="number of training epochs (default: 20)")
    trn.add_argument("--lr", type=float, default=1e-3, metavar="LR",
                     help="initial learning rate for AdamW (default: 1e-3)")
    trn.add_argument("--batch-size", type=int, default=128, metavar="N",
                     help="mini-batch size (default: 128)")
    trn.add_argument("--checkpoint-every", type=int, default=None, metavar="N",
                     help="save an intermediate checkpoint every N epochs (default: disabled)")
    trn.add_argument("--seed", type=int, default=None, metavar="N",
                     help="random seed for reproducibility (default: none)")


def _add_classify_subparser(subparsers) -> None:
    p = subparsers.add_parser("classify", help="Classify image(s) with a trained model")

    p.add_argument("--model", required=True, metavar="PATH",
                   help="path to a model checkpoint produced by 'train'")
    p.add_argument("image", metavar="IMAGE",
                   help="image file to classify, or a directory of images")
    p.add_argument("--top-k", type=int, default=1, metavar="K",
                   help="print the top-K predictions with probabilities (default: 1)")


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="cifar-cnn",
        description="Train and run a configurable CNN on CIFAR-10/100.",
    )
    subparsers = parser.add_subparsers(dest="command", metavar="COMMAND")
    subparsers.required = True

    _add_train_subparser(subparsers)
    _add_classify_subparser(subparsers)

    args = parser.parse_args()

    if args.command == "train":
        if args.num_blocks > 5:
            parser.error("--num-blocks cannot exceed 5 (CIFAR images are 32x32; five halvings reach 1x1)")

        if args.seed is not None:
            torch.manual_seed(args.seed)

        config = {
            "num_blocks": args.num_blocks,
            "base_channels": args.base_channels,
            "kernel_size": args.kernel_size,
            "dropout": args.dropout,
            "activation": args.activation,
            "residual": args.residual,
        }

        try:
            train_loader, val_loader, num_classes, class_names = get_loaders(
                args.dataset, args.batch_size, args.data
            )
        except FileNotFoundError as exc:
            print(exc, file=sys.stderr)
            sys.exit(1)

        model = build_model(num_classes, config)
        run_train(args, model, config, train_loader, val_loader, class_names)

    elif args.command == "classify":
        run_classify(args)
