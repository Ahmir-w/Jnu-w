import math
from datetime import datetime

import numpy as np

from .federated import FederatedCoordinator
from .memory import DynamicMemoryBank, EnhancedMemoryUnit


class SubsetSelector:
    def __init__(
        self,
        memory_bank: DynamicMemoryBank,
        federated_coordinator: FederatedCoordinator | None = None,
        mu: float = 0.9,
    ):
        self.memory_bank = memory_bank
        self.fed = federated_coordinator
        self.mu = mu

    def compute_local_value(self, unit: EnhancedMemoryUnit) -> float:
        age_norm = min(1.0, (datetime.now() - unit.timestamp).total_seconds() / (3600 * 24))
        ret_norm = min(1.0, unit.retrieval_freq / 100.0)
        score = (
            1.0 * unit.weight
            + 0.5 * unit.difficulty
            - 0.3 * unit.privacy_score
            - 0.1 * age_norm
            - 0.05 * ret_norm
        )
        return 1.0 / (1.0 + math.exp(-score))

    def compute_federated_score(self, unit: EnhancedMemoryUnit) -> float:
        local_score = self.compute_local_value(unit)
        if self.fed is None:
            return local_score
        return (1 - self.mu) * local_score + self.mu * self.fed.get_global_score(unit.get_hash_key())

    def compute_marginal_gain(self, unit: EnhancedMemoryUnit, lam1: float, lam2: float, task_emb: np.ndarray) -> float:
        fed_score = self.compute_federated_score(unit)
        dist_to_task = np.linalg.norm(task_emb - unit.key)
        p_ti = math.exp(-dist_to_task / 0.5)
        gain = math.log(1 + fed_score * p_ti) - lam1 * unit.energy_cost - lam2 * unit.privacy_score
        return max(gain, 0.0)

    def offline_selection(self, subset_size: int, lam1: float, lam2: float, task_emb: np.ndarray) -> set[int]:
        candidates: list[tuple[int, EnhancedMemoryUnit]] = [(i, unit) for i, unit in enumerate(self.memory_bank.units)]
        if self.fed is not None:
            for j, unit in enumerate(self.fed.get_global_units(top_k=200)):
                candidates.append((-(j + 1), unit))
        scored = [
            (idx, self.compute_marginal_gain(unit, lam1, lam2, task_emb))
            for idx, unit in candidates
        ]
        scored.sort(key=lambda x: x[1], reverse=True)
        return {idx for idx, _ in scored[:subset_size]}

    def online_immediate_selection(
        self,
        new_idx: int,
        current_subset: set[int],
        subset_size: int,
        task_emb: np.ndarray,
    ) -> set[int]:
        unit = self.memory_bank.units[new_idx]
        delta = self.compute_marginal_gain(unit, 0.5, 0.5, task_emb)
        if not current_subset:
            current_subset.add(new_idx)
        else:
            avg_gain = np.mean(
                [
                    self.compute_marginal_gain(self.memory_bank.units[i], 0.5, 0.5, task_emb)
                    for i in current_subset
                    if i < len(self.memory_bank.units)
                ]
            )
            if delta > 0.8 * avg_gain:
                current_subset.add(new_idx)
        if len(current_subset) > subset_size:
            items = [
                (i, self.compute_marginal_gain(self.memory_bank.units[i], 0.5, 0.5, task_emb))
                for i in current_subset
                if i < len(self.memory_bank.units)
            ]
            items.sort(key=lambda x: x[1], reverse=True)
            current_subset = {i for i, _ in items[:subset_size]}
        return current_subset
