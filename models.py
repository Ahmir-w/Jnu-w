import torch
import torch.nn as nn


class PolicyNetwork(nn.Module):
    def __init__(self, state_dim: int = 10, hidden_dim: int = 64):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.mode_head = nn.Linear(hidden_dim, 3)
        self.param_head = nn.Linear(hidden_dim, 5)
        nn.init.constant_(self.mode_head.bias, 0.0)
        nn.init.constant_(self.param_head.bias, 0.5)

    def forward(self, state):
        x = self.fc(state)
        return self.mode_head(x), torch.sigmoid(self.param_head(x))
