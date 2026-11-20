from dataclasses import dataclass
from datetime import datetime
import math
import secrets
from typing import Sequence

import numpy as np
import torch
from torch import nn
from torch.nn.utils import parameters_to_vector, vector_to_parameters

from .memory import EnhancedMemoryUnit

DEFAULT_C = 1.0
DEFAULT_SIGMA = 0.001
DEFAULT_MU = 0.4


def memory_features(unit: EnhancedMemoryUnit, now: datetime | None = None) -> np.ndarray:
    now = now or datetime.now(tz=unit.timestamp.tzinfo)
    age = min(1.0, max(0.0, (now - unit.timestamp).total_seconds() / 86400.0))
    freq = min(1.0, max(0.0, unit.retrieval_freq / 100.0))
    return np.concatenate((np.asarray(unit.key, dtype=np.float32).reshape(-1),
                           np.array([unit.weight, unit.difficulty, unit.privacy_score, age, freq], dtype=np.float32)))


def influence_from_features(x: np.ndarray) -> np.ndarray:
    # Eq. (5), with coefficients from Table 2, including positive frequency.
    w, difficulty, privacy, age, freq = np.moveaxis(x[..., -5:], -1, 0)
    score = w + 0.5 * difficulty + 0.5 * freq - 0.3 * privacy - 0.1 * age
    return 1.0 / (1.0 + np.exp(-np.clip(score, -80.0, 80.0)))


class MemoryScorer(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 32):
        super().__init__()
        self.layers = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.ReLU(),
                                    nn.Linear(hidden_dim, 1), nn.Sigmoid())

    def forward(self, x):
        return self.layers(x).squeeze(-1)


def perturb_update(delta: np.ndarray, C: float, sigma: float,
                   rng: np.random.Generator | None = None) -> tuple[np.ndarray, dict]:
    """Whole-vector L2 clipping, then iid N(0,sigma**2) per coordinate.

    sigma is the absolute standard deviation, NOT a multiplier of C.
    Diagnostics are local-only. rng injection is for tests, not deployment.
    """
    if not math.isfinite(C) or C <= 0 or not math.isfinite(sigma) or sigma < 0:
        raise ValueError('Require C > 0 and sigma >= 0, both finite')
    delta = np.asarray(delta, dtype=np.float64)
    if delta.ndim != 1 or not delta.size or not np.isfinite(delta).all():
        raise ValueError('Expected finite nonempty flat update')
    norm = float(np.linalg.norm(delta))
    factor = min(1.0, C / norm) if norm else 1.0
    clipped = delta * factor
    # Separate entropy source: the experiment seed never fixes privacy noise.
    rng = rng if rng is not None else np.random.default_rng(secrets.randbits(128))
    noise = rng.normal(0.0, sigma, size=delta.shape)
    noisy = (clipped + noise).astype(np.float32)
    return noisy, {'raw_l2': norm, 'clipped_l2': float(np.linalg.norm(clipped)),
                   'noise_l2': float(np.linalg.norm(noise)), 'was_clipped': factor < 1.0}


@dataclass(frozen=True)
class ClientUpdate:
    round_id: int
    sample_count: int
    delta: np.ndarray


def train_client_update(units: Sequence[EnhancedMemoryUnit], global_vector: np.ndarray,
                        round_id: int, input_dim: int, hidden_dim: int = 32,
                        C: float = DEFAULT_C, sigma: float = DEFAULT_SIGMA,
                        local_epochs: int = 3, lr: float = 0.001, batch_size: int = 64,
                        seed: int = 42, device: str = 'cpu') -> tuple[ClientUpdate, dict]:
    """Call on the client. Only ClientUpdate is sent; never send diagnostics."""
    if not units or local_epochs < 1 or batch_size < 1 or lr <= 0:
        raise ValueError('Require nonempty client memory and valid training settings')
    x_np = np.stack([memory_features(u) for u in units])
    if x_np.shape[1] != input_dim or not np.isfinite(x_np).all():
        raise ValueError('Invalid or inconsistent local embedding dimensions')
    model = MemoryScorer(input_dim, hidden_dim).to(device)
    initial = torch.as_tensor(np.asarray(global_vector, dtype=np.float32), device=device).clone()
    if initial.numel() != parameters_to_vector(model.parameters()).numel():
        raise ValueError('Global model shape mismatch')
    vector_to_parameters(initial.clone(), model.parameters())
    x = torch.as_tensor(x_np, device=device)
    y = torch.as_tensor(influence_from_features(x_np), device=device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    gen = torch.Generator().manual_seed(seed)
    model.train()
    for _ in range(local_epochs):
        order = torch.randperm(len(units), generator=gen)
        for ids in order.split(batch_size):
            ids = ids.to(device)
            opt.zero_grad()
            loss = nn.functional.mse_loss(model(x[ids]), y[ids])
            loss.backward()
            opt.step()
    with torch.no_grad():
        mse = float(nn.functional.mse_loss(model(x), y).cpu())
        delta = (parameters_to_vector(model.parameters()) - initial).cpu().numpy()
    noisy, diagnostics = perturb_update(delta, C, sigma)
    diagnostics['local_mse'] = mse
    return ClientUpdate(round_id, len(units), noisy), diagnostics


class FederatedCoordinator:
    """Server: broadcast parameters and aggregate already-perturbed updates (Eq. 7)."""
    def __init__(self, input_dim: int, hidden_dim: int = 32, seed: int = 42):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            self.model = MemoryScorer(input_dim, hidden_dim)
        self.model.eval()
        self.round_id = 0
        self.history = []

    def broadcast(self) -> np.ndarray:
        return parameters_to_vector(self.model.parameters()).detach().cpu().numpy().copy()

    def aggregate(self, updates: Sequence[ClientUpdate]) -> dict:
        if not updates:
            raise ValueError('No client updates')
        current = self.broadcast()
        accumulator = np.zeros_like(current, dtype=np.float64)
        count = 0
        for u in updates:
            if u.round_id != self.round_id or u.sample_count <= 0:
                raise ValueError('Stale update or invalid sample count')
            delta = np.asarray(u.delta)
            if delta.shape != current.shape or not np.isfinite(delta).all():
                raise ValueError('Invalid noisy parameter update')
            accumulator += u.sample_count * delta.astype(np.float64)
            count += u.sample_count
        new = (current.astype(np.float64) + accumulator / count).astype(np.float32)
        with torch.no_grad():
            vector_to_parameters(torch.from_numpy(new.copy()), self.model.parameters())
        self.round_id += 1
        report = {'round': self.round_id, 'clients': len(updates), 'samples': count,
                  'parameter_count': int(current.size),
                  'uplink_vector_bytes': int(sum(u.delta.nbytes for u in updates))}
        self.history.append(report)
        return report

    def save(self, path):
        np.savez_compressed(path, vector=self.broadcast(), input_dim=self.input_dim,
                            hidden_dim=self.hidden_dim, round_id=self.round_id)

    @classmethod
    def load(cls, path):
        with np.load(path, allow_pickle=False) as f:
            obj = cls(int(f['input_dim']), int(f['hidden_dim']))
            vector = np.asarray(f['vector'], dtype=np.float32)
            if vector.shape != obj.broadcast().shape or not np.isfinite(vector).all():
                raise ValueError('Invalid global checkpoint')
            with torch.no_grad():
                vector_to_parameters(torch.from_numpy(vector.copy()), obj.model.parameters())
            obj.round_id = int(f['round_id'])
        return obj

    def local_scores(self, units: Sequence[EnhancedMemoryUnit]) -> np.ndarray:
        # On a real client this is called on the locally downloaded model.
        if not units:
            return np.empty(0, dtype=np.float32)
        x = np.stack([memory_features(u) for u in units])
        with torch.no_grad():
            return self.model(torch.from_numpy(x)).cpu().numpy()

    def get_global_score(self, unit: EnhancedMemoryUnit) -> float:
        return float(self.local_scores([unit])[0])


def run_simulated_federation(units, *, num_clients=4, rounds=20, hidden_dim=32,
                             C=DEFAULT_C, sigma=DEFAULT_SIGMA, local_epochs=3,
                             lr=0.001, batch_size=64, seed=42, device='cpu'):
    """Single-process logical-client simulation, NOT physical device federation."""
    if num_clients < 2 or rounds < 1 or len(units) < num_clients:
        raise ValueError('Need at least two nonempty logical clients and one round')
    order = np.random.default_rng(seed).permutation(len(units))
    shards = [[units[int(i)] for i in indices] for indices in np.array_split(order, num_clients)]
    coordinator = FederatedCoordinator(np.asarray(units[0].key).size + 5, hidden_dim, seed)
    for _ in range(rounds):
        vector = coordinator.broadcast()
        packets = []
        for i, shard in enumerate(shards):
            packet, _ = train_client_update(shard, vector, coordinator.round_id,
                coordinator.input_dim, hidden_dim, C, sigma, local_epochs, lr,
                batch_size, seed + 1009 * coordinator.round_id + i, device)
            packets.append(packet)
        report = coordinator.aggregate(packets)
        print(f"[FED] round={report['round']}/{rounds} clients={report['clients']} "
              f"samples={report['samples']} C={C:g} sigma={sigma:g}")
    return coordinator
