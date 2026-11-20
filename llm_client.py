import math
import time
from typing import Dict, List

import numpy as np
import openai
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer

from .config import LLMConfig


class LLMClient:
    def __init__(self, config: LLMConfig | None = None):
        self.config = config or LLMConfig()
        self.client = openai.OpenAI(
            base_url=self.config.base_url,
            api_key=self.config.api_key,
            timeout=self.config.request_timeout,
        )
        self.model = self.config.model
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.config.tokenizer_name,
                trust_remote_code=True,
            )
        except Exception:
            self.tokenizer = AutoTokenizer.from_pretrained(self.config.fallback_tokenizer)

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.embedding_model = SentenceTransformer(self.config.embedding_model)

    def get_embedding(self, context: str) -> np.ndarray:
        vec = self.embedding_model.encode(context, convert_to_numpy=True)
        norm = np.linalg.norm(vec)
        return (vec / norm).astype(np.float32) if norm > 0 else vec.astype(np.float32)

    def get_next_token_probabilities(self, context: str, top_k: int = 150) -> Dict[str, float]:
        for _ in range(3):
            try:
                response = self.client.completions.create(
                    model=self.model,
                    prompt=context,
                    max_tokens=1,
                    logprobs=top_k,
                    echo=False,
                    timeout=self.config.request_timeout,
                )
                choice = response.choices[0]
                logprobs_obj = choice.logprobs
                probs: Dict[str, float] = {}
                if logprobs_obj and hasattr(logprobs_obj, "content") and logprobs_obj.content:
                    first_content = logprobs_obj.content[0]
                    if "top_logprobs" in first_content and first_content["top_logprobs"]:
                        for item in first_content["top_logprobs"]:
                            probs[item["token"]] = float(math.exp(item["logprob"]))
                    else:
                        token_str = first_content.get("token", "")
                        logprob = first_content.get("logprob", -1e4)
                        if token_str:
                            probs[token_str] = float(math.exp(logprob))
                total = sum(probs.values())
                if total > 0:
                    for key in probs:
                        probs[key] /= total
                return probs
            except Exception:
                time.sleep(1)
        return {}

    def get_token_probability(self, context: str, target_token: str, top_k: int = 150) -> float:
        probs = self.get_next_token_probabilities(context, top_k=top_k)
        if not probs:
            return 0.1
        if target_token in probs:
            return probs[target_token]
        target_clean = target_token.strip()
        for token, prob in probs.items():
            if token.strip() == target_clean:
                return prob
        return 0.1

    def tokenize_text(self, text: str) -> List[str]:
        token_ids = self.tokenizer.encode(text, add_special_tokens=False)
        return [self.tokenizer.decode([tid]) for tid in token_ids]
