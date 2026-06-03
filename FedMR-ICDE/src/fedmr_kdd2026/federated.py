from .memory import EnhancedMemoryUnit


class FederatedCoordinator:
    def __init__(self):
        self.global_units: dict[str, EnhancedMemoryUnit] = {}
        self.aggregated_scores: dict[str, float] = {}

    def upload_units_deduplicate(self, device_id: str, units: list[EnhancedMemoryUnit]):
        for unit in units:
            key = unit.get_hash_key()
            if key not in self.global_units:
                self.global_units[key] = unit
                self.aggregated_scores[key] = 0.9

    def get_global_score(self, hash_key: str) -> float:
        return self.aggregated_scores.get(hash_key, 0.5)

    def get_global_units(self, top_k: int = 100) -> list[EnhancedMemoryUnit]:
        return list(self.global_units.values())[:top_k]
