# cifar-cnn

A small, configurable convolutional neural network trained on CIFAR-10 or CIFAR-100,
presented as a command-line tool.

## Installation

```sh
uv pip install git+https://github.com/knit-sh/image-classifier-example.git
```

> **GPU note**: the above pulls the CPU-only PyTorch wheel from PyPI.
> For CUDA support install torch separately first:
> ```sh
> uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
> uv pip install git+https://github.com/knit-sh/image-classifier-example.git --no-deps
> ```

## Download the dataset

The tool does **not** download the dataset for you — fetch and extract it first.

**CIFAR-10**

```sh
mkdir -p data
curl -L https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz | tar -xz -C data
# -> data/cifar-10-batches-py/
```

**CIFAR-100**

```sh
mkdir -p data
curl -L https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz | tar -xz -C data
# -> data/cifar-100-python/
```

Then pass the containing directory (`data` above) via `--data`.

## Usage

### train

```
cifar-cnn train --model model.pt --data DIR [OPTIONS]
```

**Dataset / IO**

| Flag | Default | Description |
|---|---|---|
| `--dataset` | `cifar10` | `cifar10` or `cifar100` |
| `--model PATH` | *(required)* | Where to save the final model |
| `--data DIR` | *(required)* | Directory holding the extracted dataset (see above) |

**Architecture**

| Flag | Default | Description |
|---|---|---|
| `--num-blocks N` | `3` | Conv blocks (max 5 for 32×32 input) |
| `--base-channels N` | `32` | Channels in block 1; doubled each block |
| `--kernel-size` | `3` | Kernel size: `3` or `5` |
| `--dropout P` | `0.3` | Dropout before the classifier head |
| `--residual` | off | Enable ResNet-style skip connections |
| `--activation` | `relu` | `relu`, `gelu`, or `leaky-relu` |

**Training**

| Flag | Default | Description |
|---|---|---|
| `--epochs N` | `20` | Training epochs |
| `--lr LR` | `1e-3` | Initial learning rate (AdamW + cosine decay) |
| `--batch-size N` | `128` | Mini-batch size |
| `--checkpoint-every N` | off | Save `model_epochNNNN.pt` every N epochs |
| `--seed N` | none | Random seed for reproducibility |

**Example**

```sh
cifar-cnn train \
  --model runs/my_model.pt \
  --dataset cifar100 \
  --data data \
  --num-blocks 4 \
  --base-channels 64 \
  --residual \
  --activation gelu \
  --epochs 50 \
  --checkpoint-every 10
```

### classify

```
cifar-cnn classify --model model.pt IMAGE
```

`IMAGE` can be a single image file or a directory. For a directory, each image
is classified and printed as `filename: class (prob)`. Use `--top-k K` to show
the top K predictions.

```sh
# single image
cifar-cnn classify --model runs/my_model.pt photo.jpg

# directory
cifar-cnn classify --model runs/my_model.pt ./test_images/ --top-k 3
```

## Architecture

Each conv block contains:

```
Conv2d → BatchNorm2d → activation → MaxPool2d(2×2)
                                          ↑
                  optional residual shortcut (Conv1×1, stride 2)
```

After all blocks, a global average pool collapses the spatial dimensions,
followed by dropout and a single linear layer.

With default settings (`--num-blocks 3 --base-channels 32`) the channel
progression is 3 → 32 → 64 → 128, producing ~200 k parameters on CIFAR-10.
