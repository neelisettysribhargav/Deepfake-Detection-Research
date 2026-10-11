import argparse
import sys

def main():
    parser = argparse.ArgumentParser(description="Deepfake Detection using CSWin Transformer")
    parser.add_argument(
        "--mode",
        type=str,
        default="train",
        choices=["train", "test", "tune", "finetune"],
        help="Pipeline mode to execute: 'train' (100 epochs from scratch), 'test' (evaluate on test set), 'tune' (Bayesian optimization), 'finetune' (DFDC transfer learning)"
    )
    args = parser.parse_args()

    if args.mode == "train":
        from training.train import main as train_main
        train_main()
    elif args.mode == "test":
        from evaluation.test import main as test_main
        test_main()
    elif args.mode == "tune":
        from training.tune import main as tune_main
        tune_main()
    elif args.mode == "finetune":
        from training.finetune import main as finetune_main
        finetune_main()

if __name__ == "__main__":
    main()
