from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def plot_scene8_convergence(online_result: dict, baseline: dict, budget: float, output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)
    epochs = len(online_result["reward_conv"])
    epochs_axis = np.arange(1, epochs + 1)
    baseline_styles = {
        "Scene2 Full Bank": {"color": "#4e79a7", "linestyle": "--"},
        "Scene3 Offline": {"color": "#59a14f", "linestyle": "-."},
        "Scene6 Random": {"color": "#e15759", "linestyle": ":"},
        "Scene7 Merge": {"color": "#b07aa1", "linestyle": (0, (6, 2))},
        "Scene9 CREAM": {"color": "#edc948", "linestyle": (0, (3, 2, 1, 2))},
    }

    plt.figure(figsize=(13, 5.5))
    plt.subplot(1, 2, 1)
    plt.scatter(epochs_axis, online_result["reward_conv"], s=18, color="#1f77b4", alpha=0.75, label="Scene8 Epoch Points")
    plt.plot(epochs_axis, online_result["reward_conv"], color="#0b3c6f", linewidth=2.4, label="Scene8 Convergence Path")
    plt.scatter([epochs], [online_result["reward_conv"][-1]], color="#0b3c6f", s=90, zorder=5, label="Scene8 Final")
    for name, (reward, _) in baseline.items():
        style = baseline_styles.get(name, {"color": "#999999", "linestyle": "--"})
        plt.axhline(y=reward, alpha=0.9, linewidth=1.8, label=name, **style)
    plt.xlabel("Epoch")
    plt.ylabel("Reward")
    plt.title("Scene8 Reward Convergence")
    plt.grid(alpha=0.18, linestyle=":")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.scatter(epochs_axis, online_result["cost_conv"], s=18, color="#d62728", alpha=0.75, label="Scene8 Epoch Points")
    plt.plot(epochs_axis, online_result["cost_conv"], color="#8c1d18", linewidth=2.4, label="Scene8 Convergence Path")
    plt.scatter([epochs], [online_result["cost_conv"][-1]], color="#8c1d18", s=90, zorder=5, label="Scene8 Final")
    plt.axhline(y=budget, color="r", linestyle="-", linewidth=1.6, label="Budget")
    for name, (_, cost) in baseline.items():
        style = baseline_styles.get(name, {"color": "#999999", "linestyle": "--"})
        plt.axhline(y=cost, alpha=0.9, linewidth=1.8, label=name, **style)
    plt.xlabel("Epoch")
    plt.ylabel("Cost")
    plt.title("Scene8 Cost Convergence")
    plt.grid(alpha=0.18, linestyle=":")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "scene8_convergence.png", dpi=300)
    plt.close()


def plot_scene8_epoch_scatter(online_result: dict, output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)
    epochs = len(online_result["reward_conv"])
    conv_cost = np.array(online_result["cost_conv"])
    conv_reward = np.array(online_result["reward_conv"])
    plt.figure(figsize=(7.5, 6.5))
    plt.scatter(conv_cost, conv_reward, c=np.arange(1, epochs + 1), cmap="viridis", s=42, edgecolors="none")
    plt.plot(conv_cost, conv_reward, color="black", linewidth=2.0, alpha=0.82)
    plt.scatter([conv_cost[-1]], [conv_reward[-1]], color="black", s=95, zorder=5)
    plt.colorbar(label="Epoch")
    plt.xlabel("Cost")
    plt.ylabel("Reward")
    plt.title("Scene8 Epoch-wise Convergence Scatter")
    plt.grid(alpha=0.18, linestyle=":")
    plt.tight_layout()
    plt.savefig(output_dir / "scene8_epoch_scatter.png", dpi=300)
    plt.close()


def plot_scene8_vs_baselines(online_result: dict, baseline: dict, budget: float, output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)
    epochs = len(online_result["reward_conv"])
    conv_cost = np.array(online_result["cost_conv"])
    conv_reward = np.array(online_result["reward_conv"])
    plt.figure(figsize=(8.5, 6.5))
    plt.scatter(conv_cost, conv_reward, c=np.arange(1, epochs + 1), cmap="Blues", s=24, alpha=0.65, label="Scene8 Epoch Points")
    plt.plot(conv_cost, conv_reward, color="#0b3c6f", linewidth=2.5, label="Scene8 Convergence Path")
    plt.scatter(conv_cost[0], conv_reward[0], color="#f28e2b", s=95, zorder=5, label="Scene8 Start")
    plt.scatter(conv_cost[-1], conv_reward[-1], color="#0b3c6f", s=110, zorder=6, label="Scene8 Final")
    baseline_colors = {
        "Scene2 Full Bank": "#9c755f",
        "Scene3 Offline": "#59a14f",
        "Scene6 Random": "#e15759",
        "Scene7 Merge": "#b07aa1",
        "Scene9 CREAM": "#edc948",
    }
    for name, (reward, cost) in baseline.items():
        plt.scatter(cost, reward, s=90, color=baseline_colors.get(name), label=name)
        plt.annotate(name.replace("Scene", "S"), (cost, reward), textcoords="offset points", xytext=(6, 4), fontsize=9)
    plt.axvline(budget, color="r", linestyle="--", label="Budget")
    plt.xlabel("Cost")
    plt.ylabel("Reward")
    plt.title("Scene8 vs Scene2/3/6/7/9 Reward-Cost Comparison")
    plt.grid(alpha=0.18, linestyle=":")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "scene8_vs_baselines.png", dpi=300)
    plt.close()
