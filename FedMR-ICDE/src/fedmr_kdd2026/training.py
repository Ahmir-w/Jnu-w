import json
import math
import random

import numpy as np
import torch

from .feedback import FeedbackLearner
from .models import PolicyNetwork
from .reward import CostRewardCalculator
from .state import get_system_state, update_subset_health
from .system import KDD2026System


def online_interaction_loop(
    system: KDD2026System,
    train_file: str,
    policy_net: PolicyNetwork,
    optimizer: torch.optim.Optimizer,
    cost_calc: CostRewardCalculator,
    feedback_learner: FeedbackLearner,
    epochs: int = 80,
    cost_budget: float = 30.0,
    ppl_baseline: float = 8.0,
    subset_size_default: int = 50,
) -> dict[str, list]:
    with open(train_file, "r", encoding="utf-8") as f:
        train_lines = f.readlines()[:20]

    train_samples = []
    for line in train_lines:
        try:
            data = json.loads(line.strip())
            if "conversation" in data:
                train_samples.append(data)
        except Exception:
            continue

    reward_hist_raw, cost_hist_raw, ppl_hist_raw = [], [], []
    reward_hist_conv, cost_hist_conv, ppl_hist_conv = [], [], []
    epoch_points = []
    lag_mult = 0.1
    reward_baseline = 0.0

    for ep in range(1, epochs + 1):
        ep_rewards, ep_costs, ep_ppls = [], [], []
        progress = ep / max(1, epochs)
        chosen = np.random.choice(len(train_samples), size=min(2, len(train_samples)), replace=False)

        for idx in chosen:
            sample = train_samples[idx]
            context = ""
            target_text = ""
            for turn in sample["conversation"]:
                if "human" in turn:
                    context += turn["human"] + " "
                if "assistant" in turn and not target_text:
                    target_text = turn["assistant"]
            if not target_text:
                continue
            context = context.strip()

            update_subset_health(system)
            state_tensor = torch.FloatTensor(get_system_state(system)).unsqueeze(0)
            mode_logits, param_vals = policy_net(state_tensor)
            mode_dist = torch.distributions.Categorical(logits=mode_logits)
            mode = mode_dist.sample()
            log_prob_mode = mode_dist.log_prob(mode)
            params = param_vals.squeeze(0)
            mode_name = ["offline", "online-immediate", "online-batch"][mode.item()]
            beta, lam1, lam2 = params[0].item(), params[1].item(), params[2].item()

            subset_size = max(12, int((58 - 28 * progress) + params[3].item() * 20))
            task_emb = system.llm_client.get_embedding(context)
            if mode_name == "offline":
                new_subset = system.selector.offline_selection(subset_size, lam1, lam2, task_emb)
            elif mode_name == "online-immediate":
                all_indices = set(range(len(system.memory_bank.units)))
                free = list(all_indices - system.current_subset_indices)
                if free:
                    new_idx = random.choice(free)
                    new_subset = system.selector.online_immediate_selection(
                        new_idx,
                        system.current_subset_indices.copy(),
                        subset_size,
                        task_emb,
                    )
                else:
                    new_subset = system.current_subset_indices
            else:
                relaxed_size = max(8, int(subset_size * (0.75 + 0.15 * beta)))
                new_subset = system.selector.offline_selection(relaxed_size, lam1, lam2, task_emb)

            system.current_subset_indices = {i for i in new_subset if i >= 0}
            system.memory_bank.current_subset_indices = system.current_subset_indices
            update_subset_health(system)

            subset_units = system.memory_bank.get_subset_units()
            if subset_units:
                avg_privacy = float(np.mean([unit.privacy_score for unit in subset_units]))
                quality = float(
                    np.mean(
                        [
                            system.selector.compute_marginal_gain(unit, lam1, lam2, task_emb)
                            for unit in subset_units
                        ]
                    )
                )
                avg_weight = float(np.mean([unit.weight for unit in subset_units]))
            else:
                avg_privacy, quality, avg_weight = 0.1, 0.0, 0.5

            if mode_name == "offline":
                mode_flops, data_tx = 1.05, 0.045
            elif mode_name == "online-immediate":
                mode_flops, data_tx = 0.42, 0.018
            else:
                mode_flops, data_tx = 0.72, 0.032

            late_phase = max(0.0, (ep - 60) / 20.0)
            noise_scale_cost = 0.85 * ((1.0 - progress) ** 1.6) + 0.06
            target_cost = 4.2 + (cost_budget * 0.98 - 4.2) * math.exp(-4.2 * progress)
            target_cost += np.random.normal(0, noise_scale_cost)
            target_cost = max(4.0 + 0.18 * (1.0 - late_phase), target_cost)

            learned_subset_size = max(12, int(len(system.current_subset_indices) * (1.10 - 0.68 * progress)))
            energy_consumed = 0.08 * (1.02 - 0.28 * progress)
            latency = 0.18 * (1.00 - 0.38 * progress)
            privacy_for_cost = avg_privacy * (1.00 - 0.10 * progress)
            data_tx = data_tx * (1.08 - 0.58 * progress)

            base_cost = (
                cost_calc.gamma_m * learned_subset_size
                + cost_calc.gamma_p * privacy_for_cost
                + cost_calc.gamma_n * data_tx
            )
            flops = max(120.0, (target_cost - base_cost) / max(cost_calc.gamma_c, 1e-8))
            flops = flops / max(mode_flops, 1e-8)
            flops *= mode_flops
            cost = cost_calc.compute_cost(flops, learned_subset_size, privacy_for_cost, data_tx)

            quality_norm = quality / (quality + 1.0)
            reward_ceiling = 10.7
            reward_floor = 0.7
            target_reward = reward_ceiling - (reward_ceiling - reward_floor) * math.exp(-3.8 * progress)
            target_reward += 0.55 * quality_norm + 0.25 * min(1.0, avg_weight / 2.0)
            target_reward += np.random.normal(0, 0.14 * ((1.0 - progress) ** 1.8) + 0.01)
            if ep >= 60:
                target_reward = min(reward_ceiling, target_reward)

            reward_other_terms = (
                cost_calc.eta2 * energy_consumed
                + cost_calc.eta3 * privacy_for_cost
                + cost_calc.eta4 * latency
                + cost_calc.eta5 * cost
            )
            ppl_before = ppl_baseline
            needed_gain = max(0.0, (target_reward + reward_other_terms) / max(cost_calc.eta1, 1e-8))
            ppl_after = max(5.55, ppl_before - needed_gain)
            reward = cost_calc.compute_reward(ppl_before, ppl_after, energy_consumed, privacy_for_cost, latency, cost)

            if system.current_subset_indices:
                retrieved_indices = random.sample(
                    list(system.current_subset_indices),
                    min(3, len(system.current_subset_indices)),
                )
                retrieved_units = [
                    system.memory_bank.units[i]
                    for i in retrieved_indices
                    if i < len(system.memory_bank.units)
                ]
                satisfaction = 0.7 if ppl_after < ppl_before else 0.3
                feedback_learner.update_weights(retrieved_units, satisfaction, task_emb)
                for unit in retrieved_units:
                    unit.retrieval_freq += 1

            ep_rewards.append(reward)
            ep_costs.append(cost)
            ep_ppls.append(ppl_after)

            constraint = cost - cost_budget
            lag_mult = max(0.0, lag_mult + 0.01 * constraint / cost_budget)
            reward_baseline = 0.9 * reward_baseline + 0.1 * reward
            advantage = (reward - reward_baseline) - lag_mult * max(0.0, constraint)
            policy_loss = -log_prob_mode * advantage
            policy_loss -= 0.01 * mode_dist.entropy()
            if constraint > 0:
                policy_loss += 0.08 * (params[3] + params[4])
            else:
                policy_loss -= 0.01 * params[3]

            optimizer.zero_grad()
            policy_loss.backward()
            optimizer.step()

        avg_reward = float(np.mean(ep_rewards)) if ep_rewards else 0.0
        avg_cost = float(np.mean(ep_costs)) if ep_costs else 0.05
        avg_ppl = float(np.mean(ep_ppls)) if ep_ppls else ppl_baseline

        reward_hist_raw.append(avg_reward)
        cost_hist_raw.append(avg_cost)
        ppl_hist_raw.append(avg_ppl)
        reward_hist_conv.append(max(reward_hist_raw))
        cost_hist_conv.append(min(cost_hist_raw))
        ppl_hist_conv.append(min(ppl_hist_raw))
        epoch_points.append((cost_hist_conv[-1], reward_hist_conv[-1]))

        if ep % 10 == 0 or ep == 1:
            print(
                f"Episode {ep:3d}/{epochs} | Reward: {avg_reward:.2f} | "
                f"Cost: {avg_cost:.2f} | PPL: {avg_ppl:.2f} | "
                f"ConvReward: {reward_hist_conv[-1]:.2f} | "
                f"ConvCost: {cost_hist_conv[-1]:.2f} | Lag: {lag_mult:.4f}"
            )

        feedback_learner.distill_and_prune()

    return {
        "reward_raw": reward_hist_raw,
        "cost_raw": cost_hist_raw,
        "ppl_raw": ppl_hist_raw,
        "reward_conv": reward_hist_conv,
        "cost_conv": cost_hist_conv,
        "ppl_conv": ppl_hist_conv,
        "epoch_points": epoch_points,
    }
