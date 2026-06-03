import math

import numpy as np
from sklearn.neighbors import NearestNeighbors

from .federated import FederatedCoordinator
from .memory import DynamicMemoryBank, EnhancedMemoryUnit


class EnhancedInferenceEngine:
    def __init__(
        self,
        memory_bank: DynamicMemoryBank,
        federated_coordinator: FederatedCoordinator | None = None,
        k: int = 20,
        temperature: float = 0.05,
    ):
        self.memory_bank = memory_bank
        self.fed = federated_coordinator
        self.k = k
        self.temperature = temperature
        self.local_knn: NearestNeighbors | None = None
        self.global_vectors: np.ndarray | None = None
        self.global_units_list: list[EnhancedMemoryUnit] = []

    def update_models(self):
        subset = self.memory_bank.get_subset_units()
        if subset:
            keys = np.array([unit.key for unit in subset])
            self.local_knn = NearestNeighbors(
                n_neighbors=min(self.k, len(subset)),
                metric="euclidean",
            )
            self.local_knn.fit(keys)
        else:
            self.local_knn = None

        if self.fed is not None:
            global_units = self.fed.get_global_units(top_k=100)
            if global_units:
                self.global_vectors = np.array([unit.key for unit in global_units])
                self.global_units_list = global_units

    def knn_search_with_global(self, query_vector: np.ndarray) -> list[tuple[str, float]]:
        all_results: list[tuple[str, float]] = []
        if self.local_knn is not None:
            dists, idxs = self.local_knn.kneighbors(query_vector.reshape(1, -1))
            subset = self.memory_bank.get_subset_units()
            for i, idx in enumerate(idxs[0]):
                if idx < len(subset):
                    all_results.append((subset[idx].target, dists[0][i]))

        if self.global_vectors is not None and len(self.global_vectors) > 0:
            dists_global = np.linalg.norm(self.global_vectors - query_vector, axis=1)
            for idx in np.argsort(dists_global)[: self.k]:
                all_results.append((self.global_units_list[idx].target, dists_global[idx]))
        return all_results

    def compute_knn_probability(self, query_vector: np.ndarray, eval_token: str) -> float:
        all_results = self.knn_search_with_global(query_vector)
        prob_knn = 0.0
        total_weight = 0.0
        for target_str, dist in all_results:
            weight = math.exp(-dist / self.temperature)
            total_weight += weight
            if eval_token.strip().lower() in target_str.strip().lower():
                prob_knn += weight
        if total_weight > 0:
            prob_knn /= total_weight
        return prob_knn

    def fuse_probabilities(self, p_llm: float, p_knn: float, alpha_base: float) -> float:
        alpha_t = alpha_base if p_knn > p_llm else alpha_base * 0.05
        return (1.0 - alpha_t) * p_llm + alpha_t * p_knn
