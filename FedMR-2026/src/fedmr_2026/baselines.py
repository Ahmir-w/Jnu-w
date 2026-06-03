import numpy as np

from .memory import DynamicMemoryBank, EnhancedMemoryUnit


def compress_memory_bank_by_merging(memory_bank: DynamicMemoryBank, window: int = 100, limit: int = 100) -> DynamicMemoryBank:
    units = memory_bank.units
    if not units:
        return DynamicMemoryBank()
    merged: list[EnhancedMemoryUnit] = []
    i = 0
    while i < len(units):
        group = [
            units[j]
            for j in range(i, min(i + window, len(units)))
            if units[j].target == units[i].target
        ]
        if group:
            raw_avg_key = np.mean([unit.key for unit in group], axis=0)
            noise = np.random.normal(0, 0.1, raw_avg_key.shape)
            avg_key = raw_avg_key * 0.8 + noise * 0.2
            avg_key = (avg_key / max(np.linalg.norm(avg_key), 1e-8)).astype(np.float32)
            merged.append(
                EnhancedMemoryUnit(
                    key=avg_key,
                    target=group[0].target,
                    timestamp=group[-1].timestamp,
                    scene_tag=group[0].scene_tag,
                    weight=max(unit.weight for unit in group),
                    energy_cost=float(np.mean([unit.energy_cost for unit in group])),
                    privacy_score=float(np.mean([unit.privacy_score for unit in group])),
                    difficulty=float(np.mean([unit.difficulty for unit in group])),
                )
            )
        i += window
    merged.sort(key=lambda unit: unit.weight, reverse=True)
    if len(merged) < limit:
        existing = {unit.target for unit in merged}
        merged.extend([unit for unit in units if unit.target not in existing][: limit - len(merged)])
    new_bank = DynamicMemoryBank()
    for unit in merged[:limit]:
        new_bank.add_unit(unit)
    return new_bank


class CREAMGroupEnhance:
    def __init__(self, eta: float = 7e-5):
        self.eta = eta

    def context_feature_enhance(self, text_emb: np.ndarray, visual_emb: np.ndarray) -> np.ndarray:
        enhanced = text_emb + self.eta * (visual_emb - text_emb)
        norm = np.linalg.norm(enhanced)
        return enhanced / norm if norm > 0 else enhanced

    def cross_modal_group_similarity(self, set_a: list[np.ndarray], set_b: list[np.ndarray]) -> float:
        def cgs_oneway(x_set, y_set):
            total = 0.0
            for x in x_set:
                min_dist = min(np.linalg.norm(x - y) for y in y_set)
                total += min_dist
            return total / len(x_set)

        return max(cgs_oneway(set_a, set_b), cgs_oneway(set_b, set_a))

    def cream_enhance(self, p_base: float, ctx_emb: np.ndarray, memory_embeddings: list[np.ndarray]) -> float:
        if len(memory_embeddings) == 0:
            return p_base
        group_emb = np.mean(memory_embeddings, axis=0)
        enhanced_emb = self.context_feature_enhance(ctx_emb, group_emb)
        sim = self.cross_modal_group_similarity([enhanced_emb], memory_embeddings)
        p_cream = np.exp(-sim / 0.1)
        return 0.7 * p_base + 0.3 * p_cream
