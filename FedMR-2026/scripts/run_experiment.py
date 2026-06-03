import argparse
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from fedmr2026.config import ExperimentConfig, LLMConfig
from fedmr2026.experiment import run_experiment


def parse_args():
    parser = argparse.ArgumentParser(description="Run FedMR  9-scenario experiment.")
    parser.add_argument("--train-file", required=True, type=Path, help="Path to train data.jsonl.")
    parser.add_argument("--eval-file", required=True, type=Path, help="Path to eval data.jsonl.")
    parser.add_argument("--output-dir", default=Path("outputs"), type=Path, help="Directory for generated figures.")
    parser.add_argument("--num-eval-samples", default=100, type=int)
    parser.add_argument("--max-conversations", default=500, type=int)
    parser.add_argument("--max-pairs-per-conv", default=500, type=int)
    parser.add_argument("--difficulty-threshold", default=0.3, type=float)
    parser.add_argument("--subset-size", default=15000, type=int)
    parser.add_argument("--epochs", default=80, type=int)
    parser.add_argument("--budget", default=30.0, type=float)
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--base-url", default="http://219.222.20.79:00000/v1")#The local port where you deployed the model
    parser.add_argument("--api-key", default="sk-no-key-required")
    parser.add_argument("--model", default="meta-llama-31-8b-instruct-q80")
    parser.add_argument("--tokenizer-name", default="meta-llama/Llama-3.1-8B-Instruct")
    parser.add_argument("--embedding-model", default="all-mpnet-base-v2")
    return parser.parse_args()


def main():
    args = parse_args()
    exp_config = ExperimentConfig(
        train_file=args.train_file,
        eval_file=args.eval_file,
        output_dir=args.output_dir,
        num_eval_samples=args.num_eval_samples,
        max_conversations=args.max_conversations,
        max_pairs_per_conversation=args.max_pairs_per_conv,
        difficulty_threshold=args.difficulty_threshold,
        subset_size=args.subset_size,
        epochs=args.epochs,
        budget=args.budget,
        seed=args.seed,
    )
    llm_config = LLMConfig(
        base_url=args.base_url,
        api_key=args.api_key,
        model=args.model,
        tokenizer_name=args.tokenizer_name,
        embedding_model=args.embedding_model,
    )
    run_experiment(exp_config, llm_config)


if __name__ == "__main__":
    main()
