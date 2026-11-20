# FedMR — clipping and Gaussian perturbation revision

This is a scoped revision of the supplied `Jnu-w-main.zip` reference implementation. It replaces raw-memory exchange in the federated scoring module with the model-update protocol in paper Section 3.3, Eqs. (5)–(8). It does not certify that the remaining prototype implements the complete paper.

## New configuration (proposed, not recovered historical settings)

| Parameter | Default | Meaning |
|---|---:|---|
| mu | 0.4 | Global scorer weight; local influence weight is 0.6 |
| C / clip_c | 1.0 | L2 norm cap for the entire flattened client parameter update |
| sigma | 0.001 | Absolute Gaussian standard deviation per coordinate, variance 1e-6 |
| sigma/C | 0.001 | Equivalent noise multiplier for the above definitions |
| fed_rounds | 20 | Logical federated rounds |
| fed_clients | 4 | Logical clients in the single-process driver |
| fed_local_epochs | 3 | MLP epochs per client and round |
| fed_lr | 0.001 | Adam learning rate |
| fed_batch_size | 64 | Local batch size |
| fed_hidden_dim | 32 | MLP hidden width |

The client creates features `[embedding, weight, difficulty, privacy, age, frequency]`, uses Eq. (5) as the MSE target (positive frequency coefficient 0.5 as in Table 2), trains from the broadcast model, flattens the complete parameter delta, clips its L2 norm, adds independent zero-mean Gaussian noise to each coordinate, and transmits the perturbed vector and sample count. The server uses sample-count weighted averaging. Each client evaluates its downloaded global scorer on its own local memory; retrieval remains strictly local. Sigma is NOT a noise multiplier in the code. No formal DP guarantee is claimed; sample counts and participation metadata remain visible, and model updates can still leak information. Noise uses a separate OS-entropy-seeded generator by default, not the public experiment seed. This is an experimental, not hardened cryptographic, privacy implementation.

## Install and smoke tests

Python >= 3.10, a locally accessible OpenAI-compatible endpoint with completion logprobs, and the original dependencies are required. On Jetson use its platform-compatible torch build. The smoke tests require only numpy, torch and scikit-learn.

```bash
pip install -r requirements.txt
OMP_NUM_THREADS=1 python -m unittest discover -s tests -v
```

## A. Single-process logical-client experiment

```bash
python -u scripts/run_experiment.py \
  --train-file /path/to/train/data.jsonl \
  --eval-file /path/to/eval/data.jsonl \
  --output-dir outputs/natural_science_privacy \
  --base-url http://127.0.0.1:8000/v1 \
  --model YOUR_LOCAL_MODEL_NAME \
  --mu 0.4 --clip-c 1.0 --sigma 0.001 \
  --fed-clients 4 --fed-rounds 20 --fed-local-epochs 3
```

Use real dataset paths, endpoint and model ID. `scripts/run_natural_science.sh` is a configurable wrapper. The driver constructs one reference memory bank and partitions its retained units among logical clients for scorer training; the reference bank is used for evaluation. This is a **simulation, not evidence of physical four-Jetson deployment or data isolation**, and is not equivalent to partitioning raw dialogues across devices. For real distributed training use the following mode. Output includes `global_scorer.npz`, `federated_config.json` and measured PPL from the existing evaluation routine. The `round_history` records counts and vector bytes, not actual network traffic. `--fed-device cpu` is the default scorer training device; `--fed-device cuda` is optional. The LLM endpoint and embedding device remain separate settings.

## B. Separate processes on actual clients and server

`federated_round.py` provides an explicit parameter-exchange path without uploading raw memories. The caller must securely transfer files and coordinate rounds (for example, using authenticated SSH), and use disjoint local training data. No automatic network orchestration, secure aggregation, retry protocol or differential privacy accounting is provided. `--embedding-dim` must match the actual encoder output (768 for the configured all-mpnet-base-v2). Use locally hosted LLM endpoints if measuring on-device execution.

1. Server initialization:

```bash
python scripts/federated_round.py init --embedding-dim 768 --out round0.npz
```

2. Copy `round0.npz` to each Jetson. Run on each physical client with its own local data, client ID and local endpoint:

```bash
python scripts/federated_round.py client --checkpoint round0.npz \
  --train-file /path/to/local/train.jsonl --client-id jetson1 \
  --base-url http://127.0.0.1:8000/v1 --model YOUR_LOCAL_MODEL_NAME \
  --clip-c 1.0 --sigma 0.001 --out jetson1_round0.npz
```

The script rebuilds local memory from the source file each invocation; it is a minimal file-based integration, not a persistent production client. Retain `*.LOCAL_ONLY.json` locally: these diagnostics contain raw/clipped update norms and local MSE. Only transfer the `.npz` update packet (no raw memories, embeddings, token targets or raw deltas).

3. Server aggregation (after receiving same-round packets):

```bash
python scripts/federated_round.py aggregate --checkpoint round0.npz \
  --updates jetson1_round0.npz jetson2_round0.npz jetson3_round0.npz jetson4_round0.npz \
  --out round1.npz
```

Repeat with the new checkpoint for later rounds. The server writes an actual per-round `.round.json` manifest recording C and sigma. Do not discard these manifests. To evaluate the downloaded scorer with the existing driver on a client, use `--fed-checkpoint round20.npz --mu 0.4` and a client-specific dataset. The evaluation manifest deliberately does not infer C and sigma for externally trained models; use the actual training manifests.

## C. Scope and limitations of the original prototype

- The original 500 candidate cap, 5000 memory-bank capacity, truncated probability API evaluation and other unrelated mechanisms are retained. These do not by themselves establish fidelity to the paper's full-sequence, resource-policy, and probability definitions.
- Removed legacy scene-specific probability rescaling and evaluation-query embedding use in selection configuration; task embedding is now computed from training queries. Eq. (5) frequency sign was aligned with the paper. Raw global memory sharing and duplicate retrieval were removed. Scene 8 uses the configured subset limit rather than adding 30. These changes may change numerical outputs; rerun affected experiments. The original evaluation routine still has other approximations (e.g., top-k probability fallback, first 15 target tokens, adaptive fusion and averaging per response) and is not a fully audited paper-quality PPL evaluator.
- The original `training.py` deliberately synthesizes reward/cost convergence using epoch-dependent target curves. It is **disabled by default**. The optional `--run-legacy-policy-simulation` is retained only for compatibility/debugging and must not be reported as measured evidence. This revision does not implement or validate the complete policy or marginal-gain predictor.
- The new C/sigma values are reasonable initial engineering choices for a small MLP, not recovered from original experiments, nor validated optimal settings. Test them on actual validation workloads, report measured utility, and rerun with the selected values. Parameter overrides (including mu sensitivity settings) are supported. Do not retroactively attribute these parameters to old results.
- Local memory creation uses an OpenAI-compatible LLM API and a separate embedding encoder. Ensure both actually execute on-device for on-device claims; a remote call is not on-device execution. The server-side federation script does not require access to client data.

## Dataset format

JSONL, each line containing `{"category":"Natural_Science","conversation":[{"human":"question"},{"assistant":"answer"}]}`. Use separate training, validation and test splits. Do not tune on final test results.
