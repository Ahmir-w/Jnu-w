import numpy as np

from .llm_client import LLMClient
from .memory import DynamicMemoryBank, EnhancedMemoryUnit


class FeedbackLearner:
    def __init__(
        self,
        memory_bank: DynamicMemoryBank,
        llm_client: LLMClient,
        learning_rate: float = 0.01,
        mastery_threshold: float = 0.99,
    ):
        self.memory_bank = memory_bank
        self.llm_client = llm_client
        self.eta = learning_rate
        self.theta_m = mastery_threshold

    def update_weights(self, retrieved_units: list[EnhancedMemoryUnit], satisfaction: float, query_emb: np.ndarray):
        avg_sat = 0.5
        for unit in retrieved_units:
            sim = np.dot(query_emb, unit.key) / (
                np.linalg.norm(query_emb) * np.linalg.norm(unit.key) + 1e-8
            )
            unit.weight += self.eta * (satisfaction - avg_sat) * sim
            unit.weight = max(0.1, min(unit.weight, 5.0))
            unit.mastery_score = min(
                1.0,
                unit.mastery_score + 0.01 * max(0.0, satisfaction - 0.5),
            )

    def distill_and_prune(self):
        candidates = [unit for unit in self.memory_bank.units if unit.mastery_score > self.theta_m]
        if not candidates:
            return
        candidates.sort(key=lambda unit: unit.mastery_score, reverse=True)
        to_remove = candidates[:2]
        for unit in to_remove:
            if unit in self.memory_bank.units:
                self.memory_bank.units.remove(unit)
        if to_remove:
            print(f"  [反馈学习] 移除 {len(to_remove)} 个已掌握记忆单元（候选 {len(candidates)}）")
