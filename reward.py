class CostRewardCalculator:
    def __init__(
        self,
        gamma_c: float = 0.002,
        gamma_m: float = 0.15,
        gamma_p: float = 2.0,
        gamma_n: float = 0.1,
        eta1: float = 4.0,
        eta2: float = 0.2,
        eta3: float = 0.3,
        eta4: float = 0.1,
        eta5: float = 0.02,
    ):
        self.gamma_c = gamma_c
        self.gamma_m = gamma_m
        self.gamma_p = gamma_p
        self.gamma_n = gamma_n
        self.eta1 = eta1
        self.eta2 = eta2
        self.eta3 = eta3
        self.eta4 = eta4
        self.eta5 = eta5

    def compute_cost(
        self,
        flops: float,
        subset_size: int,
        avg_privacy: float,
        data_transferred: float = 0.1,
    ) -> float:
        cost = (
            self.gamma_c * flops
            + self.gamma_m * subset_size
            + self.gamma_p * avg_privacy
            + self.gamma_n * data_transferred
        )
        return max(0.05, cost)

    def compute_reward(
        self,
        ppl_before: float,
        ppl_after: float,
        energy_consumed: float,
        privacy_leak: float,
        latency: float,
        cost: float,
    ) -> float:
        ppl_gain = max(0.0, ppl_before - ppl_after)
        reward = (
            self.eta1 * ppl_gain
            - self.eta2 * energy_consumed
            - self.eta3 * privacy_leak
            - self.eta4 * latency
            - self.eta5 * cost
        )
        return max(0.0, reward)


def estimate_static_reward_cost(
    ppl_before: float,
    ppl_after: float,
    subset_size: int,
    avg_privacy: float,
    flops: float,
    data_transferred: float,
    energy: float,
    latency: float,
    cost_calc: CostRewardCalculator,
) -> tuple[float, float]:
    cost = cost_calc.compute_cost(flops, subset_size, avg_privacy, data_transferred)
    reward = cost_calc.compute_reward(ppl_before, ppl_after, energy, avg_privacy, latency, cost)
    return reward, cost
