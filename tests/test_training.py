"""Offline tests for metric correctness and a real optimizer update."""

import math
import unittest

from train import MnistMLP, evaluate, train_one_epoch

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

class EvaluationTests(unittest.TestCase):
    def test_partial_batch_is_weighted_by_number_of_examples(self):
        # Three predictions all choose 0; the final one-example batch is wrong.
        # An unweighted batch mean would incorrectly report 50%, rather than 2/3.
        logits = torch.tensor([[math.log(3), 0.0]] * 3)
        labels = torch.tensor([0, 0, 1])
        loader = DataLoader(TensorDataset(logits, labels), batch_size=2)
        metrics, targets, predictions = evaluate(nn.Identity(), loader, torch.device('cpu'))
        self.assertEqual(metrics['count'], 3)
        self.assertAlmostEqual(metrics['accuracy'], 2 / 3)
        self.assertAlmostEqual(metrics['article_accuracy'], 0.5)
        self.assertAlmostEqual(metrics['loss'], (2 * math.log(4 / 3) + math.log(4)) / 3, places=6)
        self.assertAlmostEqual(metrics['article_loss'], (math.log(4 / 3) + math.log(4)) / 2, places=6)
        self.assertEqual(targets.tolist(), [0, 0, 1])
        self.assertEqual(predictions.tolist(), [0, 0, 0])

    def test_evaluation_disables_gradients_and_restores_model_mode(self):
        class ObservedModel(nn.Module):
            def forward(self, images):
                self.observed = (self.training, torch.is_grad_enabled())
                return images

        model = ObservedModel()
        model.train()
        loader = DataLoader(TensorDataset(torch.tensor([[1., 0.]]), torch.tensor([0])))
        evaluate(model, loader, torch.device('cpu'))
        self.assertEqual(model.observed, (False, False))
        self.assertTrue(model.training)

    def test_empty_evaluation_is_rejected(self):
        loader = DataLoader(TensorDataset(torch.empty(0, 2), torch.empty(0, dtype=torch.long)))
        with self.assertRaisesRegex(ValueError, 'empty'):
            evaluate(nn.Identity(), loader, torch.device('cpu'))


class TrainingTests(unittest.TestCase):
    def test_optimizer_step_reduces_loss_on_separable_examples(self):
        torch.manual_seed(42)
        images = torch.tensor([[2., 0.], [0., 2.]] * 8)
        labels = torch.tensor([0, 1] * 8)
        loader = DataLoader(TensorDataset(images, labels), batch_size=4)
        model = nn.Linear(2, 2)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.2)
        before = evaluate(model, loader, torch.device('cpu'))[0]['loss']
        train_one_epoch(model, loader, optimizer, torch.device('cpu'))
        after = evaluate(model, loader, torch.device('cpu'))[0]['loss']
        self.assertLess(after, before * 0.7)

    def test_image_batch_produces_ten_logits_and_accepts_backpropagation(self):
        model = MnistMLP()
        logits = model(torch.zeros(3, 1, 28, 28))
        self.assertEqual(tuple(logits.shape), (3, 10))
        nn.functional.cross_entropy(logits, torch.tensor([0, 1, 2])).backward()
        self.assertTrue(all(p.grad is not None for p in model.parameters()))


if __name__ == '__main__':
    torch.set_num_threads(1)
    unittest.main()
