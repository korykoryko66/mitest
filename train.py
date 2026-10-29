"""Command-line entry point. See README.md for Colab and local instructions."""

import argparse

from src.training import Config, run_experiment


def main():
    parser = argparse.ArgumentParser(description="Train a CNN on Fashion-MNIST using CUDA or CPU")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--require-gpu", action="store_true")
    parser.add_argument("--quick", action="store_true", help="One epoch on 2048/512/512 samples; smoke test only")
    args = parser.parse_args()
    config = Config(epochs=1 if args.quick else args.epochs, batch_size=args.batch_size,
                    learning_rate=args.learning_rate, seed=args.seed, num_workers=args.num_workers,
                    quick=args.quick, require_gpu=args.require_gpu)
    run_experiment(config, data_dir=args.data_dir, output_dir=args.output_dir)


if __name__ == "__main__":
    main()
