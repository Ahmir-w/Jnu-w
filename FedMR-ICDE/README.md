# FedMR-KDD2026: Federated Memory Refinement for On-Device LLMs

This repository provides a modular implementation of the KDD2026 FedMR experiment. It evaluates nine memory-enhanced inference scenarios and trains Scene 8 with federated online learning, then plots the reward-cost convergence curves.

## Requirements

To run the code, install the following environment:

- Python 3.10 or higher
- PyTorch 2.0 or higher
- A running OpenAI-compatible LLM endpoint that supports completion logprobs
- Optional GPU for faster embedding/model operations

Install dependencies:

```bash
pip install -r requirements.txt
```

## Project Structure

```text
FedMR-KDD2026/
│
├── README.md                         # Project usage, dataset links, and execution order
├── requirements.txt                  # Python dependencies
├── .gitignore                        # Git ignore rules for caches, outputs, data, checkpoints
│
├── scripts/
│   ├── run_experiment.py             # Main CLI entrypoint for all experiments
│   └── run_natural_science.sh        # Example command for Natural_Science
│
└── src/
    └── fedmr_kdd2026/
        ├── __init__.py
        ├── config.py                 # Dataclass configs for LLM and experiment settings
        ├── llm_client.py             # OpenAI-compatible LLM client, tokenizer, embeddings, token probabilities
        ├── memory.py                 # EnhancedMemoryUnit and DynamicMemoryBank
        ├── builder.py                # Builds memory banks from JSONL conversations
        ├── federated.py              # FederatedCoordinator for hash-score global aggregation
        ├── selection.py              # Offline, online-immediate, and federated marginal-gain selection
        ├── inference.py              # kNN memory retrieval and LLM/kNN probability fusion
        ├── baselines.py              # Merge compression baseline and CREAM-style enhancement
        ├── system.py                 # KDD2026System wrapper and PPL evaluation
        ├── models.py                 # PolicyNetwork for dynamic mode and parameter decisions
        ├── feedback.py               # Feedback learning, weight update, distillation/pruning simulation
        ├── reward.py                 # Unified cost and reward formulas
        ├── state.py                  # Device/system state and subset health estimation
        ├── training.py               # Scene 8 federated online learning loop
        ├── plots.py                  # Scene 8 convergence and reward-cost figures
        └── experiment.py             # Full experiment orchestration for the nine scenarios
```

## Dataset

The code expects each dataset split to be stored as JSONL files:

```text
dataset_category/
├── Arts_and_Entertainment/
│   ├── train/data.jsonl
│   └── eval/data.jsonl
├── Business_and_Finance/
│   ├── train/data.jsonl
│   └── eval/data.jsonl
├── Business_Marketing/
│   ├── train/data.jsonl
│   └── eval/data.jsonl
├── Math/
│   ├── train/data.jsonl
│   └── eval/data.jsonl
└── Natural_Science/
    ├── train/data.jsonl
    └── eval/data.jsonl
```

Each JSONL line should contain a `conversation` list:

```json
{
  "category": "Natural_Science",
  "conversation": [
    {"human": "question text"},
    {"assistant": "answer text"}
  ]
}
```

Dataset links used or referenced by the paper:

- ShareGPT: https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered
- Chinese-Alpaca: https://github.com/ymcui/Chinese-LLaMA-Alpaca
- Chinese Medical Dialogue Data / ChineseMDD-style corpus: https://github.com/Toyhom/Chinese-medical-dialogue-data

If you use private or locally processed category splits, place them under `data/dataset_category/` or pass absolute paths through the command line.

## Execution Order

The complete experiment follows this order:

1. Configure LLM endpoint, model name, tokenizer, dataset paths, and experiment hyperparameters.
2. Build the external memory bank from the training JSONL file.
3. Compute the average evaluation-task embedding.
4. Evaluate nine scenarios:
   - Scene 1: no external memory.
   - Scene 2: full external memory bank.
   - Scene 3: offline optimal subset selection.
   - Scene 4: online-immediate selection.
   - Scene 5: online-batch selection.
   - Scene 6: random selection.
   - Scene 7: heuristic merge compression.
   - Scene 8: federated collaboration.
   - Scene 9: CREAM baseline.
5. Train Scene 8 with federated online learning for reward-cost convergence.
6. Estimate static reward/cost for baseline scenarios using the same formulas.
7. Save figures under `outputs/`.

## Running

Example for the Natural_Science split:

```bash
python scripts/run_experiment.py \
  --train-file /home/WangZP/DaiMa/kdd2026-dataset/dataset_category/Natural_Science/train/data.jsonl \
  --eval-file /home/WangZP/DaiMa/kdd2026-dataset/dataset_category/Natural_Science/eval/data.jsonl \
  --output-dir outputs/natural_science \
  --subset-size 15000 \
  --max-conversations 500 \
  --max-pairs-per-conv 500 \
  --epochs 80
```

You can also run the example shell script:

```bash
bash scripts/run_natural_science.sh
```

To use another model endpoint:

```bash
python scripts/run_experiment.py \
  --train-file data/dataset_category/Math/train/data.jsonl \
  --eval-file data/dataset_category/Math/eval/data.jsonl \
  --base-url http://YOUR_HOST:PORT/v1 \
  --model YOUR_MODEL_NAME \
  --tokenizer-name Qwen/Qwen2.5-7B-Instruct
```

## Outputs

The following figures are generated:

```text
outputs/
├── scene8_convergence.png        # Reward and cost convergence against static baselines
├── scene8_epoch_scatter.png      # Epoch-wise Scene 8 cost-reward scatter
└── scene8_vs_baselines.png       # Scene 8 reward-cost path vs baseline points
```

The terminal also prints PPL for all nine scenarios and final reward/cost summaries.

## Notes

- The LLM endpoint must support `completions.create(..., logprobs=top_k)`; otherwise token-probability estimation will fall back to the default probability.
- The original single-file implementation initialized Ray, but Ray was not used by the executed experiment logic. The dependency remains in `requirements.txt` for compatibility, but the refactored entrypoint does not require Ray initialization.
- Generated datasets, model checkpoints, and figures are ignored by Git by default.
