"""Fashion-MNIST training code shared by the CLI and standalone notebook.

Section markers are consumed by scripts/build_notebook.py.
"""

# %% imports
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
import csv
import json
import math
import platform
import random
import time

import numpy as np
import torch
import torchvision
from torch import nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
import matplotlib.pyplot as plt


@dataclass
class Config:
    epochs: int = 5
    batch_size: int = 128
    learning_rate: float = 0.001
    seed: int = 42
    val_size: int = 6000
    num_workers: int = 0
    quick: bool = False
    require_gpu: bool = False

    def __post_init__(self):
        if self.epochs < 1 or self.batch_size < 1:
            raise ValueError("epochs and batch_size must be positive")
        if not math.isfinite(self.learning_rate) or self.learning_rate <= 0:
            raise ValueError("learning_rate must be finite and positive")
        if self.val_size < 1 or self.num_workers < 0:
            raise ValueError("val_size must be positive; num_workers cannot be negative")
        if self.quick:
            self.epochs = 1


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def get_device(require_gpu=False):
    if torch.cuda.is_available():
        return torch.device("cuda")
    if require_gpu:
        raise RuntimeError(
            "CUDA GPU unavailable. In Colab choose Runtime > Change runtime type > GPU, "
            "then reconnect. To intentionally use CPU set REQUIRE_GPU=False "
            "or omit --require-gpu. GPU availability is not guaranteed."
        )
    return torch.device("cpu")


def environment_info(device):
    return {
        "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": str(torch.__version__),
        "torchvision": str(torchvision.__version__),
        "numpy": str(np.__version__),
        "device": str(device),
        "cuda_available": torch.cuda.is_available(),
        "cuda_runtime": torch.version.cuda,
        "gpu_name": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "cpu_threads": torch.get_num_threads(),
    }


# %% data
CLASS_NAMES = [
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot",
]


class FashionMNIST(datasets.FashionMNIST):
    # Official dataset repository over HTTPS; torchvision still checks the MD5 hashes.
    mirrors = [
        "https://raw.githubusercontent.com/zalandoresearch/fashion-mnist/master/data/fashion/",
    ]


def split_indices(size, val_size, seed):
    if not 0 < val_size < size:
        raise ValueError("val_size must leave at least one training sample")
    order = torch.randperm(size, generator=torch.Generator().manual_seed(seed)).tolist()
    return order[val_size:], order[:val_size]


def make_loaders(config, data_dir="data"):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
    ])
    full_train = FashionMNIST(data_dir, train=True, download=True, transform=transform)
    test_set = FashionMNIST(data_dir, train=False, download=True, transform=transform)
    train_ids, val_ids = split_indices(len(full_train), config.val_size, config.seed)
    train_set = Subset(full_train, train_ids)
    val_set = Subset(full_train, val_ids)
    if config.quick:
        # Only for smoke testing, never report these as full-dataset results.
        train_set = Subset(train_set, range(min(2048, len(train_set))))
        val_set = Subset(val_set, range(min(512, len(val_set))))
        test_set = Subset(test_set, range(min(512, len(test_set))))
    options = {"batch_size": config.batch_size, "num_workers": config.num_workers,
               "pin_memory": torch.cuda.is_available()}
    train_loader = DataLoader(train_set, shuffle=True,
                              generator=torch.Generator().manual_seed(config.seed), **options)
    return (train_loader, DataLoader(val_set, shuffle=False, **options),
            DataLoader(test_set, shuffle=False, **options))


# %% model
class FashionCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 7 * 7, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 10),
        )

    def forward(self, images):
        # Return logits: CrossEntropyLoss applies log-softmax internally.
        return self.classifier(self.features(images))


# %% engine
def train_one_epoch(model, loader, optimizer, device):
    model.train()
    loss_fn = nn.CrossEntropyLoss()
    total_loss, correct, count = 0.0, 0, 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        loss = loss_fn(logits, labels)
        loss.backward()
        optimizer.step()
        count += labels.size(0)
        total_loss += loss.item() * labels.size(0)
        correct += (logits.argmax(1) == labels).sum().item()
    if count == 0:
        raise ValueError("Training loader must not be empty")
    return total_loss / count, correct / count


@torch.inference_mode()
def evaluate(model, loader, device):
    model.eval()
    loss_fn = nn.CrossEntropyLoss(reduction="sum")
    total_loss, count = 0.0, 0
    labels_all, predictions_all = [], []
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        logits = model(images)
        total_loss += loss_fn(logits, labels).item()
        count += labels.size(0)
        labels_all.append(labels.cpu())
        predictions_all.append(logits.argmax(1).cpu())
    if count == 0:
        raise ValueError("Evaluation loader must not be empty")
    y_true, y_pred = torch.cat(labels_all), torch.cat(predictions_all)
    return {"loss": total_loss / count,
            "accuracy": (y_true == y_pred).sum().item() / count,
            "y_true": y_true, "y_pred": y_pred}


def fit(model, train_loader, val_loader, config, device, run_dir):
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    history, best_loss, best_epoch = [], float("inf"), 0
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    started = time.perf_counter()
    for epoch in range(1, config.epochs + 1):
        epoch_started = time.perf_counter()
        train_loss, train_accuracy = train_one_epoch(model, train_loader, optimizer, device)
        validation = evaluate(model, val_loader, device)
        if not math.isfinite(train_loss) or not math.isfinite(validation["loss"]):
            raise RuntimeError("Non-finite loss. Check inputs and reduce the learning rate.")
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        row = {"epoch": epoch, "train_loss": train_loss, "train_accuracy": train_accuracy,
               "val_loss": validation["loss"], "val_accuracy": validation["accuracy"],
               "epoch_seconds": time.perf_counter() - epoch_started}
        history.append(row)
        if validation["loss"] < best_loss:
            best_loss, best_epoch = validation["loss"], epoch
            torch.save({"model_state_dict": model.state_dict(), "epoch": epoch,
                        "val_loss": best_loss, "config": asdict(config)},
                       run_dir / "best_model.pt")
        # Flush the learning history after every epoch so a disconnect loses less work.
        with (run_dir / "history.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(row))
            writer.writeheader()
            writer.writerows(history)
        print(f"Epoch {epoch:02d}/{config.epochs} | loss {train_loss:.4f} | "
              f"acc {train_accuracy:.2%} | val_loss {validation['loss']:.4f} | "
              f"val_acc {validation['accuracy']:.2%}", flush=True)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    return history, best_epoch, time.perf_counter() - started


# %% outputs
def confusion_counts(y_true, y_pred, num_classes=10):
    counts = np.zeros((num_classes, num_classes), dtype=np.int64)
    np.add.at(counts, (np.asarray(y_true), np.asarray(y_pred)), 1)
    return counts


def plot_history(history, path):
    epochs = [row["epoch"] for row in history]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for prefix, label in [("train", "Training"), ("val", "Validation")]:
        axes[0].plot(epochs, [row[prefix + "_loss"] for row in history], "o-", label=label)
        axes[1].plot(epochs, [row[prefix + "_accuracy"] for row in history], "o-", label=label)
    for axis, title in zip(axes, ["Cross-entropy loss", "Accuracy"]):
        axis.set(xlabel="Epoch", title=title, xticks=epochs)
        axis.grid(alpha=0.2)
        axis.legend()
    axes[1].set_ylim(0, 1)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_confusion(counts, path):
    fig, axis = plt.subplots(figsize=(10, 9))
    chart = axis.imshow(counts, cmap="Blues")
    fig.colorbar(chart, ax=axis, fraction=0.046, pad=0.04)
    axis.set(xticks=range(10), yticks=range(10), xticklabels=CLASS_NAMES,
             yticklabels=CLASS_NAMES, xlabel="Predicted class", ylabel="True class",
             title="Test confusion matrix (counts)")
    plt.setp(axis.get_xticklabels(), rotation=45, ha="right")
    for row in range(10):
        for col in range(10):
            axis.text(col, row, str(counts[row, col]), ha="center", va="center", fontsize=8,
                      color="white" if counts[row, col] > counts.max() / 2 else "black")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_predictions(dataset, y_true, y_pred, path, errors_only=False):
    indices = torch.where(y_true != y_pred)[0].tolist() if errors_only else list(range(len(y_true)))
    indices = indices[:12]
    if not indices:
        return False
    rows = math.ceil(len(indices) / 4)
    fig, axes = plt.subplots(rows, 4, figsize=(12, rows * 3), squeeze=False)
    for axis in axes.flat:
        axis.axis("off")
    for axis, index in zip(axes.flat, indices):
        image, _ = dataset[index]
        axis.imshow((image.squeeze().numpy() * 0.5 + 0.5).clip(0, 1), cmap="gray")
        true_id, predicted_id = y_true[index].item(), y_pred[index].item()
        axis.set_title(f"True: {CLASS_NAMES[true_id]}\nPred: {CLASS_NAMES[predicted_id]}",
                       color="green" if true_id == predicted_id else "crimson", fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return True


def save_artifacts(model, test_loader, history, config, device, run_dir,
                   best_epoch, train_seconds, split_sizes):
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    result = evaluate(model, test_loader, device)
    counts = confusion_counts(result["y_true"], result["y_pred"])
    per_class = []
    for index, name in enumerate(CLASS_NAMES):
        true_positive = int(counts[index, index])
        predicted = int(counts[:, index].sum())
        support = int(counts[index].sum())
        precision = true_positive / predicted if predicted else 0.0
        recall = true_positive / support if support else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        per_class.append({"class_id": index, "class_name": name, "precision": precision,
                          "recall": recall, "f1": f1, "support": support})
    metrics = {"test_loss": result["loss"], "test_accuracy": result["accuracy"],
               "best_epoch": best_epoch,
               "best_val_loss": min(row["val_loss"] for row in history),
               "train_seconds": train_seconds,
               "parameter_count": sum(p.numel() for p in model.parameters()),
               "split_sizes": split_sizes, "quick_mode": config.quick,
               "per_class": per_class,
               "macro_f1": sum(row["f1"] for row in per_class) / len(per_class)}
    for name, payload in [("config.json", asdict(config)),
                          ("environment.json", environment_info(device)), ("metrics.json", metrics)]:
        (run_dir / name).write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    plot_history(history, run_dir / "training_curves.png")
    plot_confusion(counts, run_dir / "confusion_matrix.png")
    plot_predictions(test_loader.dataset, result["y_true"], result["y_pred"], run_dir / "predictions.png")
    plot_predictions(test_loader.dataset, result["y_true"], result["y_pred"],
                     run_dir / "misclassified.png", errors_only=True)
    print(f"Best epoch: {best_epoch} | Test accuracy: {result['accuracy']:.2%}", flush=True)
    return metrics


# %% runner
def run_experiment(config, data_dir="data", output_dir=None):
    device = get_device(config.require_gpu)
    seed_everything(config.seed)
    if output_dir is None:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        run_dir = Path("outputs") / stamp
    else:
        run_dir = Path(output_dir)
    # Never combine a new run with an old checkpoint or misleading plots.
    if run_dir.exists() and any(run_dir.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {run_dir}. Choose a new directory.")
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.json").write_text(json.dumps(asdict(config), indent=2), encoding="utf-8")
    (run_dir / "environment.json").write_text(json.dumps(environment_info(device), indent=2), encoding="utf-8")
    print(f"Device: {device} | Output: {run_dir.resolve()}", flush=True)
    train_loader, val_loader, test_loader = make_loaders(config, data_dir)
    model = FashionCNN().to(device)
    history, best_epoch, seconds = fit(model, train_loader, val_loader, config, device, run_dir)
    checkpoint = torch.load(run_dir / "best_model.pt", map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"])
    save_artifacts(model, test_loader, history, config, device, run_dir, best_epoch, seconds,
                   {"train": len(train_loader.dataset), "validation": len(val_loader.dataset),
                    "test": len(test_loader.dataset)})
    return run_dir
