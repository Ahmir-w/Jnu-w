import hashlib
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

import numpy as np


@dataclass
class EnhancedMemoryUnit:
    key: np.ndarray
    target: str
    timestamp: datetime
    scene_tag: str
    weight: float = 1.0
    energy_cost: float = 0.0
    privacy_score: float = 0.0
    difficulty: float = 0.0
    retrieval_freq: int = 0
    mastery_score: float = 0.0
    hash_key: str = field(init=False)

    def __post_init__(self):
        self.hash_key = hashlib.sha256(self.key.tobytes()).hexdigest()

    def get_hash_key(self) -> str:
        return self.hash_key


class DynamicMemoryBank:
    def __init__(self, max_size: int = 5000):
        self.units: list[EnhancedMemoryUnit] = []
        self.max_size = max_size
        self.current_subset_indices: set[int] = set()
        self.current_subset_health: dict[str, float] = {
            "diversity": 0.0,
            "avg_weight": 0.0,
            "avg_age": 0.0,
        }

    def add_unit(self, unit: EnhancedMemoryUnit):
        if len(self.units) < self.max_size:
            self.units.append(unit)
        else:
            min_idx = min(range(len(self.units)), key=lambda i: self.units[i].weight)
            self.units[min_idx] = unit

    def get_subset_units(self) -> list[EnhancedMemoryUnit]:
        return [self.units[i] for i in self.current_subset_indices if i < len(self.units)]
