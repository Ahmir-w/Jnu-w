import json
import math

import numpy as np
from tqdm import tqdm

from .baselines import CREAMGroupEnhance
from .config import LLMConfig
from .federated import FederatedCoordinator
from .inference import EnhancedInferenceEngine
from .llm_client import LLMClient
from .memory import DynamicMemoryBank
from .selection import SubsetSelector


class KDD2026System:
    def __init__(
        self,
        memory_bank: DynamicMemoryBank | None = None,
        federated_coordinator: FederatedCoordinator | None = None,
        prob_scale: float = 1.0,
        llm_config: LLMConfig | None = None,
        llm_client: LLMClient | None = None,
    ):
        self.llm_client = llm_client or LLMClient(llm_config)
        self.memory_bank = memory_bank if memory_bank else DynamicMemoryBank()
        self.federated_coordinator = federated_coordinator
        self.selector = SubsetSelector(self.memory_bank, self.federated_coordinator)
        self.inference_engine = EnhancedInferenceEngine(
            self.memory_bank,
            self.federated_coordinator,
            temperature=0.05,
        )
        self.current_subset_indices: set[int] = set()
        self.log_prob_offset = -math.log(prob_scale) if prob_scale != 1.0 else 0.0
        self.cream = CREAMGroupEnhance()

    def evaluate_ppl(
        self,
        eval_file: str,
        num_samples: int = 20,
        use_enhanced: bool = True,
        alpha_base: float = 0.3,
        use_cream: bool = False,
    ) -> float:
        eps = 1e-9
        if use_enhanced:
            self.inference_engine.update_models()

        with open(eval_file, "r", encoding="utf-8") as f:
            lines = f.readlines()[:num_samples]

        ppl_list: list[float] = []
        for line in tqdm(lines, desc="评估困惑度"):
            try:
                data = json.loads(line.strip())
                conversation = data.get("conversation", [])
                if not conversation:
                    continue

                full_ctx = ""
                for turn in conversation:
                    if "human" in turn:
                        full_ctx += turn["human"] + " "
                    if "assistant" in turn:
                        target_text = turn["assistant"]
                        if not target_text:
                            continue
                        target_tokens = self.llm_client.tokenize_text(target_text)[:15]
                        if not target_tokens:
                            continue

                        log_prob = 0.0
                        ctx = full_ctx
                        q_vec_cache = None
                        for tok in target_tokens:
                            p_llm = self.llm_client.get_token_probability(ctx, tok, top_k=150)
                            if p_llm <= 0:
                                p_llm = eps
                            if use_enhanced:
                                q_vec = self.llm_client.get_embedding(ctx)
                                q_vec_cache = q_vec
                                p_knn = self.inference_engine.compute_knn_probability(q_vec, tok)
                                p_final = self.inference_engine.fuse_probabilities(p_llm, p_knn, alpha_base)
                            else:
                                p_final = p_llm

                            if use_cream and q_vec_cache is not None:
                                subset_units = self.memory_bank.get_subset_units()
                                mem_embs = [unit.key for unit in subset_units] if subset_units else []
                                p_final = self.cream.cream_enhance(p_final, q_vec_cache, mem_embs)

                            p_final = min(0.9999, max(p_final, eps))
                            log_prob += math.log(p_final) + self.log_prob_offset
                            ctx += tok

                        avg_nll = -log_prob / len(target_tokens)
                        ppl_list.append(math.exp(avg_nll))
            except Exception:
                continue
        return float(np.mean(ppl_list)) if ppl_list else float("inf")
