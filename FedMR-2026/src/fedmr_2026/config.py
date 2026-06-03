from dataclasses import dataclass
from pathlib import Path


@dataclass
class LLMConfig:
    base_url: str = "http://219.222.20.79:00000/v1"#The local port where you deployed the model
    api_key: str = "sk-no-key-required"
    model: str = "meta-llama-31-8b-instruct-q80"
    tokenizer_name: str = "meta-llama/Llama-3.1-8B-Instruct"
    fallback_tokenizer: str = "gpt2"
    embedding_model: str = "all-mpnet-base-v2"
    request_timeout: float = 30.0


@dataclass
class ExperimentConfig:
    train_file: Path
    eval_file: Path
    output_dir: Path = Path("outputs")
    num_eval_samples: int = 20
    max_conversations: int = 500
    max_pairs_per_conversation: int = 500
    difficulty_threshold: float = 0.3
    subset_size: int = 15000
    epochs: int = 80
    budget: float = 30.0
    seed: int = 42
