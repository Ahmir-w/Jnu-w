import json
import os
from datetime import datetime

from tqdm import tqdm

from .llm_client import LLMClient
from .memory import DynamicMemoryBank, EnhancedMemoryUnit


class MemoryBankBuilder:
    def __init__(
        self,
        llm_client: LLMClient,
        max_pairs_per_conv: int = 20,
        difficulty_threshold: float = 0.3,
    ):
        self.llm_client = llm_client
        self.memory_bank = DynamicMemoryBank()
        self.beta = 0.001
        self.max_pairs_per_conv = max_pairs_per_conv
        self.difficulty_threshold = difficulty_threshold

    def build_memory_bank_from_file(self, filepath: str, max_conversations: int = 20):
        print("\n======= 开始构建记忆库 =======")
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"文件不存在: {filepath}")

        with open(filepath, "r", encoding="utf-8") as f:
            lines = f.readlines()[:max_conversations]

        processed = 0
        for line in tqdm(lines, desc="处理训练对话"):
            try:
                data = json.loads(line.strip())
                conversation = data.get("conversation", [])
                if not conversation:
                    continue

                conv_text = ""
                for turn in conversation:
                    if "human" in turn:
                        conv_text += turn["human"] + " "
                    if "assistant" in turn:
                        conv_text += turn["assistant"] + " "
                if not conv_text.strip():
                    continue

                scene = data.get("category", "train")
                if isinstance(scene, list):
                    scene = scene[0] if scene else "train"

                tokens = self.llm_client.tokenize_text(conv_text)
                max_pairs = min(len(tokens) - 1, self.max_pairs_per_conv)
                now = datetime.now()

                for m in range(max_pairs):
                    context = " ".join(tokens[: m + 1])
                    target = tokens[m + 1]
                    key = self.llm_client.get_embedding(context)
                    energy_cost = self.beta * len(context.split())
                    privacy_score = min(
                        sum(
                            0.15
                            for word in ["password", "credit card"]
                            if word in (context + target).lower()
                        ),
                        1.0,
                    )
                    difficulty = 1.0 - self.llm_client.get_token_probability(context, target)
                    if difficulty >= self.difficulty_threshold:
                        unit = EnhancedMemoryUnit(
                            key,
                            target,
                            now,
                            scene,
                            weight=1.0,
                            energy_cost=energy_cost,
                            privacy_score=privacy_score,
                            difficulty=difficulty,
                        )
                        self.memory_bank.add_unit(unit)
                processed += 1
            except Exception:
                continue

        print(f"\n记忆库构建完成: 处理对话 {processed}，总记忆单元 M = {len(self.memory_bank.units)}")
