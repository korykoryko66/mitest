"""Offline regression tests for data isolation and the real training pipeline."""

import math
from pathlib import Path
import tempfile
import unittest

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

try:
    from src import training
except (ImportError, ModuleNotFoundError) as exc:
    if exc.name not in {"src", "src.training"}:
        raise
    training = None


class TrainingTests(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(training, "The training pipeline has not been implemented")
        torch.manual_seed(42)
        torch.set_num_threads(2)

    def test_split_is_disjoint_complete_and_repeatable(self):
        train, val = training.split_indices(100, 20, 42)
        self.assertEqual((len(train), len(val)), (80, 20))
        self.assertFalse(set(train) & set(val))
        self.assertEqual(set(train) | set(val), set(range(100)))
        self.assertEqual((train, val), training.split_indices(100, 20, 42))
        self.assertNotEqual(val, training.split_indices(100, 20, 43)[1])

    def test_invalid_configuration_is_rejected_before_data_download(self):
        for kwargs in ({"epochs": 0}, {"batch_size": 0},
                       {"learning_rate": float("nan")}, {"num_workers": -1}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                training.Config(**kwargs)
        with self.assertRaises(ValueError):
            training.split_indices(10, 10, 42)

    def test_quick_mode_is_always_a_single_epoch(self):
        self.assertEqual(training.Config(quick=True).epochs, 1)
        self.assertEqual(training.Config(quick=True, epochs=10).epochs, 1)

    def test_mnist_mlp_learns_from_a_real_batch(self):
        model = training.MnistModel()
        images = torch.rand(4, 1, 28, 28)
        labels = torch.tensor([0, 1, 2, 3])
        self.assertEqual(tuple(model(images).shape), (4, 10))
        self.assertEqual(sum(parameter.numel() for parameter in model.parameters()), 25450)
        before = model.linear2.weight.detach().clone()
        loader = DataLoader(TensorDataset(images, labels), batch_size=3)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.5)
        loss, accuracy = training.train_one_epoch(model, loader, optimizer, torch.device("cpu"))
        self.assertTrue(math.isfinite(loss))
        self.assertTrue(0 <= accuracy <= 1)
        self.assertFalse(torch.equal(before, model.linear2.weight))

    def test_nested_batches_and_models_move_to_the_selected_device(self):
        nested = [torch.ones(2), (torch.zeros(1), torch.ones(3))]
        selected = training.get_default_device()
        moved = training.to_device(nested, selected)
        self.assertIsInstance(moved, list)
        self.assertIsInstance(moved[1], tuple)
        self.assertEqual(moved[0].device.type, selected.type)
        self.assertEqual(moved[1][1].device.type, selected.type)
        model = training.to_device(training.MnistModel(), selected)
        self.assertEqual(next(model.parameters()).device.type, selected.type)

    def test_device_dataloader_preserves_all_samples_and_batch_count(self):
        dataset = TensorDataset(torch.arange(5), torch.arange(5))
        raw = DataLoader(dataset, batch_size=2)
        wrapped = training.DeviceDataLoader(raw, torch.device("cpu"))
        self.assertEqual(len(wrapped), 3)
        self.assertIs(wrapped.dataset, dataset)
        self.assertEqual(torch.cat([batch[0] for batch in wrapped]).tolist(), [0, 1, 2, 3, 4])

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is unavailable on this machine")
    def test_mnist_model_and_device_loader_train_on_real_cuda(self):
        device = training.get_device(require_gpu=True)
        model = training.to_device(training.MnistModel(), device)
        dataset = TensorDataset(torch.rand(4, 1, 28, 28), torch.arange(4))
        wrapped = training.DeviceDataLoader(DataLoader(dataset, batch_size=2), device)
        before = model.linear2.weight.detach().clone()
        loss, _ = training.train_one_epoch(
            model, wrapped, torch.optim.SGD(model.parameters(), lr=0.5), device)
        self.assertTrue(math.isfinite(loss))
        self.assertEqual(next(iter(wrapped))[0].device.type, "cuda")
        self.assertFalse(torch.equal(before, model.linear2.weight))

    def test_two_stage_sgd_changes_learning_rate_after_five_epochs(self):
        dataset = TensorDataset(torch.zeros(4, 1, 28, 28), torch.zeros(4, dtype=torch.long))
        loader = DataLoader(dataset, batch_size=2)
        model = nn.Sequential(nn.Flatten(), nn.Linear(784, 10))
        config = training.Config(epochs=6)
        with tempfile.TemporaryDirectory() as folder:
            history, _, _ = training.fit(model, loader, loader, config, torch.device("cpu"), Path(folder))
        self.assertEqual([row["learning_rate"] for row in history], [0.5] * 5 + [0.1])

    def test_evaluation_weights_the_short_last_batch_by_sample_count(self):
        logits = torch.tensor([[0., 0.], [0., 0.], [0., 2.]])
        labels = torch.tensor([0, 1, 0])
        loader = DataLoader(TensorDataset(logits, labels), batch_size=2)
        model = nn.Identity()
        model.train()
        result = training.evaluate(model, loader, torch.device("cpu"))
        expected = (2 * math.log(2) + math.log1p(math.exp(2))) / 3
        self.assertAlmostEqual(result["loss"], expected, places=6)
        self.assertAlmostEqual(result["accuracy"], 1 / 3, places=6)
        self.assertEqual(result["y_pred"].tolist(), [0, 0, 1])
        self.assertFalse(model.training)
        self.assertIsNone(logits.grad)

    def test_confusion_matrix_rows_are_true_labels(self):
        counts = training.confusion_counts([0, 0, 1, 2], [0, 1, 1, 0], 3)
        self.assertEqual(counts.tolist(), [[1, 1, 0], [0, 1, 0], [1, 0, 0]])

    def test_best_checkpoint_matches_the_best_validation_epoch_and_loads_on_cpu(self):
        # Conflicting labels force validation to worsen while training improves.
        # This catches saving final weights in place of the actual best weights.
        images = torch.zeros(12, 1, 28, 28)
        train_loader = DataLoader(TensorDataset(images, torch.zeros(12, dtype=torch.long)), batch_size=5)
        val_loader = DataLoader(TensorDataset(images, torch.ones(12, dtype=torch.long)), batch_size=5)
        model = nn.Sequential(nn.Flatten(), nn.Linear(784, 10))
        config = training.Config(epochs=3, learning_rate=0.1)
        with tempfile.TemporaryDirectory() as folder:
            history, best_epoch, seconds = training.fit(
                model, train_loader, val_loader, config, torch.device("cpu"), Path(folder))
            checkpoint = torch.load(Path(folder) / "best_model.pt", map_location="cpu", weights_only=True)
            self.assertLess(best_epoch, config.epochs)
            self.assertEqual(best_epoch, min(history, key=lambda row: row["val_loss"])["epoch"])
            self.assertEqual(checkpoint["epoch"], best_epoch)
            restored = nn.Sequential(nn.Flatten(), nn.Linear(784, 10))
            restored.load_state_dict(checkpoint["model_state_dict"])
            result = training.evaluate(restored, val_loader, torch.device("cpu"))
            self.assertAlmostEqual(result["loss"], min(row["val_loss"] for row in history), places=6)
            self.assertGreater(seconds, 0)


if __name__ == "__main__":
    unittest.main()
