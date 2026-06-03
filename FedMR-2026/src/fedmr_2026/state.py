from datetime import datetime
import socket

import numpy as np
import psutil


def get_system_state(system) -> np.ndarray:
    battery = psutil.sensors_battery()
    battery_level = battery.percent / 100.0 if battery else 1.0
    cpu = psutil.cpu_percent(interval=0.1) / 100.0
    mem = psutil.virtual_memory().percent / 100.0
    try:
        socket.create_connection(("8.8.8.8", 53), timeout=1)
        net = 1.0
    except OSError:
        net = 0.0
    health = system.memory_bank.current_subset_health
    diversity = health.get("diversity", 0.5)
    avg_weight = health.get("avg_weight", 0.5)
    avg_age = health.get("avg_age", 0.5)
    subset_fill = len(system.current_subset_indices) / max(1, system.memory_bank.max_size)
    return np.array(
        [battery_level, cpu, mem, net, 0.5, diversity, avg_weight, avg_age, 0.5, subset_fill],
        dtype=np.float32,
    )


def update_subset_health(system):
    subset = system.memory_bank.get_subset_units()
    if not subset:
        system.memory_bank.current_subset_health = {
            "diversity": 0.0,
            "avg_weight": 0.0,
            "avg_age": 0.0,
        }
        return
    weights = [unit.weight for unit in subset]
    ages = [
        min(1.0, (datetime.now() - unit.timestamp).total_seconds() / (3600 * 24))
        for unit in subset
    ]
    if len(subset) > 1:
        keys = np.array([unit.key for unit in subset])
        centroid = np.mean(keys, axis=0)
        diversity = float(np.mean(np.linalg.norm(keys - centroid, axis=1)))
    else:
        diversity = 0.0
    system.memory_bank.current_subset_health = {
        "diversity": diversity,
        "avg_weight": float(np.mean(weights)),
        "avg_age": float(np.mean(ages)),
    }
