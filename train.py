"""Independent MNIST reproduction. See docs/reference-comparison.md for provenance.

Run: python train.py --device auto --output-dir results
The notebook embeds this implementation and needs no GitHub repository URL.
"""

import argparse
import csv
import hashlib
import json
import os
import platform
import random
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor


# Factual baseline transcribed from the author's saved notebook, not this run.
# Fixed source commit: f7b4405ad1bfcfcc994f00a964904d1d5e1544e7
REFERENCE = [
    (0, 2.3093, .0909), (1, .2435, .9313), (2, .1978, .9435),
    (3, .1683, .9519), (4, .1584, .9538), (5, .1379, .9583),
    (6, .1267, .9638), (7, .1186, .9655), (8, .1189, .9651),
    (9, .1250, .9641), (10, .1198, .9636),
]


class MnistMLP(nn.Module):
    """28x28 image -> 32 ReLU features -> 10 unnormalized class logits."""

    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            nn.Flatten(), nn.Linear(784, 32), nn.ReLU(), nn.Linear(32, 10)
        )

    def forward(self, images):
        return self.network(images)


def choose_device(requested='auto'):
    if requested not in ('auto', 'cpu', 'cuda'):
        raise ValueError('device must be auto, cpu, or cuda')
    if requested == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA is unavailable. Select a GPU runtime or use --device cpu.')
    return torch.device('cuda' if requested == 'auto' and torch.cuda.is_available()
                        else 'cpu' if requested == 'auto' else requested)


def synchronize(device):
    if device.type == 'cuda':
        torch.cuda.synchronize(device)


def evaluate(model, loader, device):
    """Return weighted metrics, article-style batch means, labels and predictions.

    Uneven last batches must not be overweighted in the main metrics.
    The two article_* fields intentionally preserve that original convention.
    """
    previous_mode = model.training
    model.eval()
    loss_sum = correct = count = 0
    batch_losses, batch_accuracies, targets, predictions = [], [], [], []
    try:
        with torch.inference_mode():
            for images, labels in loader:
                images = images.to(device, non_blocking=device.type == 'cuda')
                labels = labels.to(device, non_blocking=device.type == 'cuda')
                logits = model(images)
                loss = F.cross_entropy(logits, labels, reduction='sum').item()
                predicted = logits.argmax(dim=1)
                n = labels.numel()
                hits = (predicted == labels).sum().item()
                loss_sum += loss
                correct += hits
                count += n
                batch_losses.append(loss / n)
                batch_accuracies.append(hits / n)
                targets.append(labels.cpu())
                predictions.append(predicted.cpu())
    finally:
        model.train(previous_mode)
    if not count:
        raise ValueError('Cannot evaluate an empty data loader')
    metrics = {
        'loss': loss_sum / count,
        'accuracy': correct / count,
        'article_loss': float(np.mean(batch_losses)),
        'article_accuracy': float(np.mean(batch_accuracies)),
        'count': count,
    }
    return metrics, torch.cat(targets), torch.cat(predictions)


def train_one_epoch(model, loader, optimizer, device):
    model.train()
    loss_sum = count = 0
    for images, labels in loader:
        images = images.to(device, non_blocking=device.type == 'cuda')
        labels = labels.to(device, non_blocking=device.type == 'cuda')
        optimizer.zero_grad(set_to_none=True)
        loss = F.cross_entropy(model(images), labels)
        loss.backward()
        optimizer.step()
        loss_sum += loss.detach().item() * labels.numel()
        count += labels.numel()
    if not count:
        raise ValueError('Cannot train with an empty data loader')
    return loss_sum / count


def save_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def save_csv(path, rows):
    with path.open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def plot_results(history, labels, predicted, dataset, destination, runtime):
    plt.rcParams.update({'figure.facecolor': 'white', 'font.size': 10})
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2), layout='constrained')
    epochs = [r['epoch'] for r in history]
    axes[0].plot(epochs, [r['val_loss'] for r in history], 'o-', label='This run: sample weighted')
    axes[0].plot(epochs, [r['article_val_loss'] for r in history], '--', label='This run: batch mean')
    axes[0].plot(epochs, [r[1] for r in REFERENCE], ':', color='grey', label="Author's saved batch mean")
    axes[0].set(ylabel='Validation cross entropy', xlabel='Completed epochs')
    axes[1].plot(epochs, [r['val_accuracy'] * 100 for r in history], 'o-', label='This run: sample weighted')
    axes[1].plot(epochs, [r['article_val_accuracy'] * 100 for r in history], '--', label='This run: batch mean')
    axes[1].plot(epochs, [r[2] * 100 for r in REFERENCE], ':', color='grey', label="Author's saved batch mean")
    axes[1].set(ylabel='Validation accuracy (%)', xlabel='Completed epochs')
    for ax in axes:
        ax.axvline(5, color='#777777', alpha=.3)
        ax.grid(alpha=.2)
        ax.legend(fontsize=8)
    fig.suptitle(f'MNIST 784-32-10 | {runtime} | lr 0.5 (1-5), 0.1 (6-10)')
    fig.savefig(destination / 'training_curves.png', dpi=160)
    plt.close(fig)

    matrix = np.zeros((10, 10), dtype=int)
    np.add.at(matrix, (labels.numpy(), predicted.numpy()), 1)
    save_csv(destination / 'confusion_matrix.csv', [dict(actual=i, **{f'predicted_{j}': int(matrix[i, j]) for j in range(10)}) for i in range(10)])
    fig, ax = plt.subplots(figsize=(7, 6), layout='constrained')
    im = ax.imshow(matrix, cmap='Blues')
    for i in range(10):
        for j in range(10):
            ax.text(j, i, str(matrix[i, j]), ha='center', va='center', fontsize=8,
                    color='white' if matrix[i, j] > matrix.max() / 2 else '#222222')
    ax.set(xticks=range(10), yticks=range(10), xlabel='Predicted digit', ylabel='Actual digit',
           title=f'Official MNIST test set (10,000) | {runtime}')
    fig.colorbar(im, ax=ax, shrink=.8)
    fig.savefig(destination / 'confusion_matrix.png', dpi=160)
    plt.close(fig)

    for filename, indices, title in (
        ('predictions.png', list(range(20)), 'First 20 official test examples'),
        ('mistakes.png', (labels != predicted).nonzero().flatten().tolist()[:20], 'First 20 misclassified test examples'),
    ):
        fig, axes = plt.subplots(4, 5, figsize=(10, 8), layout='constrained')
        for ax in axes.flat:
            ax.axis('off')
        for ax, index in zip(axes.flat, indices):
            ax.imshow(dataset[index][0].squeeze().numpy(), cmap='gray')
            actual, guess = int(labels[index]), int(predicted[index])
            ax.set_title(f'#{index}: true {actual}, pred {guess}', fontsize=9,
                         color='#26734d' if actual == guess else '#b33a3a')
        fig.suptitle(f'{title} | {runtime}')
        fig.savefig(destination / filename, dpi=150)
        plt.close(fig)


def run_experiment(data_dir='data', output_dir='results', device='auto', seed=42, threads=2):
    if threads < 1:
        raise ValueError('threads must be at least 1')
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(threads)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    selected = choose_device(device)
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    # This records the real environment, not a label supplied by the caller.
    is_colab = bool(os.environ.get('COLAB_RELEASE_TAG') or os.environ.get('COLAB_BACKEND_VERSION'))
    runtime = 'Google Colab' if is_colab else f'Local {platform.system()}'
    environment = {
        'execution_platform': runtime,
        'created_at_utc': datetime.now(timezone.utc).isoformat(),
        'python': sys.version.split()[0], 'torch': torch.__version__,
        'torchvision': torchvision.__version__, 'numpy': np.__version__,
        'matplotlib': matplotlib.__version__, 'os': platform.system(),
        'device': str(selected), 'cuda_available': torch.cuda.is_available(),
        'gpu_name': torch.cuda.get_device_name(selected) if selected.type == 'cuda' else None,
        'cpu_threads': torch.get_num_threads(),
    }
    print(json.dumps(environment, indent=2), flush=True)
    print('Downloading/loading MNIST; this time is excluded from training timing.', flush=True)
    dataset = MNIST(data_dir, train=True, download=True, transform=ToTensor())
    test_set = MNIST(data_dir, train=False, download=True, transform=ToTensor())
    train_set, val_set = random_split(dataset, [50000, 10000], generator=torch.Generator().manual_seed(seed))
    pin = selected.type == 'cuda'
    train_loader = DataLoader(train_set, batch_size=128, shuffle=True, num_workers=0,
                              pin_memory=pin, generator=torch.Generator().manual_seed(seed))
    val_loader = DataLoader(val_set, batch_size=256, shuffle=False, num_workers=0,
                            pin_memory=pin, generator=torch.Generator().manual_seed(seed + 1))
    test_loader = DataLoader(test_set, batch_size=256, shuffle=False, num_workers=0,
                             pin_memory=pin, generator=torch.Generator().manual_seed(seed + 2))
    model = MnistMLP().to(selected)
    initial, _, _ = evaluate(model, val_loader, selected)
    history = [{
        'epoch': 0, 'learning_rate': None, 'train_loss': None,
        'val_loss': initial['loss'], 'val_accuracy': initial['accuracy'],
        'article_val_loss': initial['article_loss'], 'article_val_accuracy': initial['article_accuracy'],
        'epoch_seconds': 0.,
    }]
    print(f"Initial: val_loss={initial['loss']:.4f}, val_acc={initial['accuracy']:.4f}", flush=True)
    synchronize(selected)
    started = time.perf_counter()
    epoch = 0
    for learning_rate in (0.5, 0.1):
        # The source creates a new SGD optimizer for each five-epoch stage.
        optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
        for _ in range(5):
            epoch += 1
            synchronize(selected)
            epoch_start = time.perf_counter()
            train_loss = train_one_epoch(model, train_loader, optimizer, selected)
            validation, _, _ = evaluate(model, val_loader, selected)
            synchronize(selected)
            row = {
                'epoch': epoch, 'learning_rate': learning_rate, 'train_loss': train_loss,
                'val_loss': validation['loss'], 'val_accuracy': validation['accuracy'],
                'article_val_loss': validation['article_loss'], 'article_val_accuracy': validation['article_accuracy'],
                'epoch_seconds': time.perf_counter() - epoch_start,
            }
            history.append(row)
            save_csv(destination / 'history.csv', history)
            print(f"Epoch {epoch:02d}/10 lr={learning_rate:.1f} train_loss={train_loss:.4f} "
                  f"val_loss={row['val_loss']:.4f} val_acc={row['val_accuracy']:.4f} "
                  f"article_val_acc={row['article_val_accuracy']:.4f} seconds={row['epoch_seconds']:.2f}", flush=True)
    synchronize(selected)
    training_seconds = time.perf_counter() - started
    # Test data is first evaluated after all ten epochs; it never selects a model.
    test, labels, predicted = evaluate(model, test_loader, selected)
    config = {'seed': seed, 'train_size': 50000, 'validation_size': 10000, 'test_size': 10000,
              'train_batch_size': 128, 'eval_batch_size': 256, 'hidden_size': 32,
              'schedule': [{'epochs': 5, 'learning_rate': .5}, {'epochs': 5, 'learning_rate': .1}],
              'optimizer': 'SGD', 'momentum': 0, 'weight_decay': 0,
              'preprocessing': 'ToTensor only', 'validation_shuffle': False, 'num_workers': 0,
              'parameter_count': sum(p.numel() for p in model.parameters()),
              'train_split_sha256': hashlib.sha256(np.asarray(train_set.indices, dtype='<i8').tobytes()).hexdigest()}
    metrics = {'schema_version': 1, 'environment': environment, 'config': config,
               'initial_validation': initial, 'final_validation': validation, 'test': test,
               'training_and_validation_seconds': training_seconds,
               'timing_excludes': 'download, initial evaluation, test evaluation, plots, final serialization; per-epoch CSV writes are included',
               'reference_kind': 'original author saved notebook validation outputs; not this run'}
    save_json(destination / 'environment.json', environment)
    save_json(destination / 'metrics.json', metrics)
    comparison = []
    for row, (_, ref_loss, ref_accuracy) in zip(history, REFERENCE):
        comparison.append({
            'epoch': row['epoch'], 'author_saved_val_loss_batch_mean': ref_loss,
            'author_saved_val_accuracy_batch_mean': ref_accuracy,
            'this_run_val_loss_batch_mean': row['article_val_loss'],
            'this_run_val_accuracy_batch_mean': row['article_val_accuracy'],
            'accuracy_difference_percentage_points': 100 * (row['article_val_accuracy'] - ref_accuracy),
            'this_run_val_accuracy_sample_weighted': row['val_accuracy'],
        })
    save_csv(destination / 'source_comparison.csv', comparison)
    save_csv(destination / 'test_predictions.csv', [
        {'index': i, 'actual': int(a), 'predicted': int(p), 'correct': int(a == p)}
        for i, (a, p) in enumerate(zip(labels, predicted))
    ])
    plot_results(history, labels, predicted, test_set, destination, f'{runtime} / {selected.type.upper()}')
    # Store only tensors and simple metadata; no serialized model class.
    cpu_state = {key: value.detach().cpu() for key, value in model.state_dict().items()}
    torch.save({'state_dict': cpu_state, 'config': config}, destination / 'model.pt')
    print(f"Final test: loss={test['loss']:.4f}, accuracy={test['accuracy']:.4f} "
          f"({round(test['accuracy'] * test['count'])}/{test['count']})", flush=True)
    print(f'Training + validation time: {training_seconds:.2f}s. Outputs saved.', flush=True)
    return metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device', choices=['auto', 'cpu', 'cuda'], default='auto')
    parser.add_argument('--data-dir', default='data')
    parser.add_argument('--output-dir', default='results')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--threads', type=int, default=2)
    args = parser.parse_args()
    run_experiment(**vars(args))


if __name__ == '__main__':
    main()
