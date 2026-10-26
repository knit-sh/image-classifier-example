import torch
import torch.nn as nn


def make_activation(name: str) -> nn.Module:
    match name:
        case "relu":
            return nn.ReLU(inplace=True)
        case "gelu":
            return nn.GELU()
        case "leaky-relu":
            return nn.LeakyReLU(0.1, inplace=True)
        case _:
            raise ValueError(f"Unknown activation: {name!r}")


class ConvBlock(nn.Module):
    """One convolutional block: Conv -> BN -> activation -> MaxPool, with an optional residual shortcut."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        activation: str,
        use_residual: bool,
    ) -> None:
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = make_activation(activation)
        self.pool = nn.MaxPool2d(2, 2)

        self.shortcut: nn.Module | None
        if use_residual:
            # 1x1 conv with stride 2 matches the spatial downsampling of MaxPool2d(2,2)
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride=2, bias=False),
                nn.BatchNorm2d(out_channels),
            )
        else:
            self.shortcut = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.pool(self.act(self.bn(self.conv(x))))
        if self.shortcut is not None:
            out = out + self.shortcut(x)
        return out


class CifarCNN(nn.Module):
    """Configurable CNN for CIFAR classification."""

    def __init__(
        self,
        num_classes: int,
        num_blocks: int,
        base_channels: int,
        kernel_size: int,
        dropout: float,
        activation: str,
        residual: bool,
    ) -> None:
        super().__init__()

        blocks: list[nn.Module] = []
        in_ch = 3
        out_ch = base_channels

        for _ in range(num_blocks):
            blocks.append(ConvBlock(in_ch, out_ch, kernel_size, activation, residual))
            in_ch = out_ch
            out_ch *= 2

        self.features = nn.Sequential(*blocks)
        # Global average pool so the classifier head is independent of spatial size
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(in_ch, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.gap(x).flatten(1)
        return self.classifier(x)


def build_model(num_classes: int, config: dict) -> CifarCNN:
    return CifarCNN(
        num_classes=num_classes,
        num_blocks=config["num_blocks"],
        base_channels=config["base_channels"],
        kernel_size=config["kernel_size"],
        dropout=config["dropout"],
        activation=config["activation"],
        residual=config["residual"],
    )


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
