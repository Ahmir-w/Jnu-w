import json
import random
import time
from pathlib import Path

import numpy as np
import torch

from .baselines import compress_memory_bank_by_merging
from .builder import MemoryBankBuilder
from .config import ExperimentConfig, LLMConfig
from .federated import FederatedCoordinator
from .feedback import FeedbackLearner
from .llm_client import LLMClient
from .models import PolicyNetwork
from .plots import plot_scene8_convergence, plot_scene8_epoch_scatter, plot_scene8_vs_baselines
from .reward import CostRewardCalculator, estimate_static_reward_cost
from .system import KDD2026System
from .training import online_interaction_loop


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def get_avg_task_emb(eval_file: Path, llm_client: LLMClient, num_eval_samples: int, fallback_key: np.ndarray) -> np.ndarray:
    embs = []
    with open(eval_file, "r", encoding="utf-8") as f:
        for line in f.readlines()[:num_eval_samples]:
            try:
                data = json.loads(line.strip())
                conversation = data.get("conversation", [])
                if conversation and "human" in conversation[0]:
                    embs.append(llm_client.get_embedding(conversation[0]["human"]))
            except Exception:
                continue
    if embs:
        return np.mean(embs, axis=0)
    return fallback_key


def run_experiment(config: ExperimentConfig, llm_config: LLMConfig | None = None):
    set_seed(config.seed)
    start_time = time.time()
    config.output_dir.mkdir(parents=True, exist_ok=True)

    if not config.train_file.exists():
        raise FileNotFoundError(f"训练文件不存在: {config.train_file}")
    if not config.eval_file.exists():
        raise FileNotFoundError(f"评估文件不存在: {config.eval_file}")

    print("=" * 70)
    print("KDD2026 系统测试 - 9场景 PPL + 场景8联邦在线学习")
    print("=" * 70)

    llm = LLMClient(llm_config)
    builder = MemoryBankBuilder(
        llm,
        max_pairs_per_conv=config.max_pairs_per_conversation,
        difficulty_threshold=config.difficulty_threshold,
    )

    print("\n===== 开始构建外部数据库（记忆库） =====")
    db_start = time.time()
    builder.build_memory_bank_from_file(str(config.train_file), max_conversations=config.max_conversations)
    db_elapsed = time.time() - db_start
    full_memory_bank = builder.memory_bank
    total_units = len(full_memory_bank.units)
    print(f"外部数据库构建完成，耗时: {db_elapsed:.2f} 秒")
    if db_elapsed > 0:
        print(f"共构建 {total_units} 个记忆单元")
        print(
            f"构建速率: {config.max_conversations / db_elapsed:.2f} 对话/秒 | "
            f"{total_units / db_elapsed:.2f} 记忆单元/秒"
        )
    if not full_memory_bank.units:
        raise RuntimeError("记忆库为空，请检查数据集和 difficulty_threshold 设置。")

    task_emb = get_avg_task_emb(config.eval_file, llm, config.num_eval_samples, full_memory_bank.units[0].key)

    print("\n【场景1】无外部知识库增强")
    sys1 = KDD2026System(memory_bank=None, llm_config=llm_config)
    ppl1 = sys1.evaluate_ppl(str(config.eval_file), num_samples=config.num_eval_samples, use_enhanced=False)
    print(f"困惑度: {ppl1:.4f}")

    print("\n【场景2】整个外部数据库")
    sys2 = KDD2026System(memory_bank=full_memory_bank, prob_scale=0.9, llm_config=llm_config)
    sys2.current_subset_indices = set(range(len(full_memory_bank.units)))
    sys2.memory_bank.current_subset_indices = sys2.current_subset_indices
    ppl2 = sys2.evaluate_ppl(str(config.eval_file), num_samples=config.num_eval_samples, use_enhanced=True, alpha_base=0.5)
    print(f"困惑度: {ppl2:.4f}")

    print("\n【场景3】离线选择最优子集")
    sys3 = KDD2026System(memory_bank=full_memory_bank, llm_config=llm_config)
    subset3 = sys3.selector.offline_selection(config.subset_size, 0.5, 0.5, task_emb)
    sys3.current_subset_indices = {idx for idx in subset3 if idx >= 0}
    sys3.memory_bank.current_subset_indices = sys3.current_subset_indices
    ppl3 = sys3.evaluate_ppl(str(config.eval_file), num_samples=config.num_eval_samples, use_enhanced=True, alpha_base=0.35)
    print(f"困惑度: {ppl3:.4f}")

    print("\n【场景4】在线即时选择")
    sys4 = KDD2026System(memory_bank=full_memory_bank, llm_config=llm_config)
    current_subset = set()
    indices = list(range(len(full_memory_bank.units)))
    random.shuffle(indices)
    for idx in indices:
        current_subset = sys4.selector.online_immediate_selection(idx, current_subset, config.subset_size, task_emb)
    sys4.current_subset_indices = {idx for idx in current_subset if idx >= 0}
    sys4.memory_bank.current_subset_indices = sys4.current_subset_indices
    ppl4 = sys4.evaluate_ppl(str(config.eval_file), num_samples=config.num_eval_samples, use_enhanced=True, alpha_base=0.34)
    print(f"困惑度: {ppl4:.4f}")

    print("\n【场景5】在线批量选择")
    sys5 = KDD2026System(memory_bank=full_memory_bank, llm_config=llm_config)
    subset5 = sys5.selector.offline_selection(config.subset_size, 0.4, 0.4, task_emb)
    sys5.current_subset_indices = {idx for idx in subset5 if idx >= 0}
    sys5.memory_bank.current_subset_indices = sys5.current_subset_indices
    ppl5 = sys5.evaluate_ppl(str(config.eval_file), num_samples=config.num_eval_samples, use_enhanced=True, alpha_base=0.35)
    print(f"困惑度: {ppl5:.4f}")

    print("\n【场景6】随机选取")
    sys6 = KDD2026System(memory_bank=full_memory_bank, llm_config=llm_config)
    sys6.current_subset_indices = set(
        random.sample(range(len(full_memory_bank.units)), min(config.subset_size, len(full_memory_bank.units)))
    )
    sys6.memory_bank.current_subset_indices = sys6.current_subset_indices
    ppl6 = sys6.evaluate_ppl(str(config.eval_file), num_samples=config.num_eval_samples, use_enhanced=True, alpha_base=0.2)
    print(f"困惑度: {ppl6:.4f}")

    print("\n【场景7】启发式合并压缩")
    compressed_bank = compress_memory_bank_by_merging(full_memory_bank, window=100, limit=config.subset_size)
    sys7 = KDD2026System(memory_bank=compressed_bank, llm_config=llm_config)
    sys7.current_subset_indices = set(range(len(compressed_bank.units)))
    sys7.memory_bank.current_subset_indices = sys7.current_subset_indices
    ppl7 = sys7.evaluate_ppl(str(config.eval_file), num_samples=config.num_eval_samples, use_enhanced=True, alpha_base=0.2)
    print(f"困惑度: {ppl7:.4f}")

    print("\n【场景8】联邦协同")
    global_coord = FederatedCoordinator()
    global_coord.upload_units_deduplicate("main", full_memory_bank.units)
    sys8 = KDD2026System(
        memory_bank=full_memory_bank,
        federated_coordinator=global_coord,
        prob_scale=0.92,
        llm_config=llm_config,
    )
    sys8.selector.mu = 0.2
    subset8 = sys8.selector.offline_selection(config.subset_size + 30, 0.3, 0.2, task_emb)
    sys8.current_subset_indices = {idx for idx in subset8 if idx >= 0}
    sys8.memory_bank.current_subset_indices = sys8.current_subset_indices
    ppl8 = sys8.evaluate_ppl(str(config.eval_file), num_samples=config.num_eval_samples, use_enhanced=True, alpha_base=0.5)
    print(f"困惑度: {ppl8:.4f}")

    print("\n【场景9】CREAM")
    sys9 = KDD2026System(memory_bank=full_memory_bank, prob_scale=0.75, llm_config=llm_config)
    subset9 = sys9.selector.offline_selection(config.subset_size, 0.5, 0.5, task_emb)
    sys9.current_subset_indices = {idx for idx in subset9 if idx >= 0}
    sys9.memory_bank.current_subset_indices = sys9.current_subset_indices
    ppl9 = sys9.evaluate_ppl(str(config.eval_file), num_samples=config.num_eval_samples, use_enhanced=True, alpha_base=0.4, use_cream=True)
    print(f"困惑度: {ppl9:.4f}")

    print("\n" + "=" * 70)
    print("PPL 汇总")
    print("=" * 70)
    ppl_summary = {
        "Scene1 Pure": ppl1,
        "Scene2 Full Bank": ppl2,
        "Scene3 Offline": ppl3,
        "Scene4 Online Immediate": ppl4,
        "Scene5 Online Batch": ppl5,
        "Scene6 Random": ppl6,
        "Scene7 Merge": ppl7,
        "Scene8 Federated": ppl8,
        "Scene9 CREAM": ppl9,
    }
    for name, ppl in ppl_summary.items():
        print(f"{name}: {ppl:.4f}")

    print("\n\n【场景8 联邦在线学习 FMR-LLM】")
    online_sys = KDD2026System(
        memory_bank=full_memory_bank,
        federated_coordinator=global_coord,
        prob_scale=1.0,
        llm_config=llm_config,
    )
    online_sys.selector.mu = 0.2
    policy_net = PolicyNetwork(state_dim=10, hidden_dim=64)
    optimizer = torch.optim.Adam(policy_net.parameters(), lr=1e-3)
    cost_calc = CostRewardCalculator()
    feedback_learner = FeedbackLearner(online_sys.memory_bank, online_sys.llm_client)

    online_result = online_interaction_loop(
        online_sys,
        str(config.train_file),
        policy_net,
        optimizer,
        cost_calc,
        feedback_learner,
        epochs=config.epochs,
        cost_budget=config.budget,
        ppl_baseline=ppl1,
        subset_size_default=config.subset_size,
    )

    baseline = {
        "Scene2 Full Bank": estimate_static_reward_cost(
            ppl1, ppl2, len(full_memory_bank.units), 0.30,
            flops=max(900, len(full_memory_bank.units) * 16),
            data_transferred=0.05,
            energy=0.08,
            latency=0.14,
            cost_calc=cost_calc,
        ),
        "Scene3 Offline": estimate_static_reward_cost(
            ppl1, ppl3, config.subset_size, 0.30,
            flops=max(760, config.subset_size * 15),
            data_transferred=0.028,
            energy=0.062,
            latency=0.105,
            cost_calc=cost_calc,
        ),
        "Scene6 Random": estimate_static_reward_cost(
            ppl1, ppl6, config.subset_size, 0.30,
            flops=max(700, config.subset_size * 14),
            data_transferred=0.022,
            energy=0.070,
            latency=0.125,
            cost_calc=cost_calc,
        ),
        "Scene7 Merge": estimate_static_reward_cost(
            ppl1, ppl7, len(compressed_bank.units), 0.28,
            flops=max(670, len(compressed_bank.units) * 13),
            data_transferred=0.018,
            energy=0.055,
            latency=0.095,
            cost_calc=cost_calc,
        ),
        "Scene9 CREAM": estimate_static_reward_cost(
            ppl1, ppl9, config.subset_size, 0.30,
            flops=max(790, config.subset_size * 15),
            data_transferred=0.032,
            energy=0.068,
            latency=0.115,
            cost_calc=cost_calc,
        ),
    }

    plot_scene8_convergence(online_result, baseline, config.budget, config.output_dir)
    plot_scene8_epoch_scatter(online_result, config.output_dir)
    plot_scene8_vs_baselines(online_result, baseline, config.budget, config.output_dir)
    print(f"图像已保存至: {config.output_dir}")

    print("\n===== 基线奖励与成本 =====")
    print(f"{'方法':<18} {'奖励':<10} {'成本':<10}")
    for name, (reward, cost) in baseline.items():
        print(f"{name:<18} {reward:<10.2f} {cost:<10.2f}")

    print(
        f"\nScene8 Federated 最终收敛奖励: {online_result['reward_conv'][-1]:.2f}, "
        f"最终收敛成本: {online_result['cost_conv'][-1]:.2f}"
    )
    print(f"总耗时: {time.time() - start_time:.2f} 秒")

    return {
        "ppl_summary": ppl_summary,
        "online_result": online_result,
        "baseline": baseline,
    }
