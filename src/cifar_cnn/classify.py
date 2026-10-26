from __future__ import annotations

import sys
from pathlib import Path

import torch
from PIL import Image

from .data import get_inference_transform
from .model import CifarCNN, build_model


_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp", ".gif"}


def load_model(model_path: str) -> tuple[CifarCNN, list[str], str]:
    checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
    config: dict = checkpoint["model_config"]
    class_names: list[str] = checkpoint["class_names"]
    dataset: str = checkpoint["dataset"]

    model = build_model(len(class_names), config)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, class_names, dataset


def _classify_one(
    image_path: Path,
    model: CifarCNN,
    class_names: list[str],
    transform,
    top_k: int,
    device: torch.device,
) -> str:
    image = Image.open(image_path).convert("RGB")
    tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        probs = torch.softmax(model(tensor), dim=1)[0]

    k = min(top_k, len(class_names))
    top_probs, top_indices = probs.topk(k)

    parts = [
        f"{class_names[idx]} ({prob:.1%})"
        for prob, idx in zip(top_probs.tolist(), top_indices.tolist())
    ]
    return ", ".join(parts)


def run_classify(args) -> None:
    model, class_names, dataset = load_model(args.model)

    device = torch.device("cpu")
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    model = model.to(device)

    transform = get_inference_transform(dataset)
    target = Path(args.image)

    if target.is_dir():
        image_files = sorted(
            f for f in target.iterdir() if f.suffix.lower() in _IMAGE_EXTENSIONS
        )
        if not image_files:
            print(f"No image files found in {target}", file=sys.stderr)
            sys.exit(1)
        for img_path in image_files:
            result = _classify_one(img_path, model, class_names, transform, args.top_k, device)
            print(f"{img_path.name}: {result}")
    elif target.is_file():
        result = _classify_one(target, model, class_names, transform, args.top_k, device)
        print(result)
    else:
        print(f"Path not found: {target}", file=sys.stderr)
        sys.exit(1)
