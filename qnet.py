import torch
import torch.nn as nn
import torch.nn.functional as F

class QNet(nn.Module):
    def __init__(self, state_size, action_size, dueling=False):
        super().__init__()
        self.dueling = dueling

        self.l1 = nn.Linear(state_size, 64)
        self.l2 = nn.Linear(64, 32)

        if self.dueling:
            self.l3 = nn.Linear(32, 1)  # Value stream
            self.l4 = nn.Linear(32, action_size)  # Advantage stream
        else:
            self.l3 = nn.Linear(32, action_size)

    def forward(self, x):
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        if self.dueling:
            value = self.l3(x)
            advantage = self.l4(x)
            q_values = value + (advantage - advantage.mean(dim=1, keepdim=True))
        else:
            q_values = self.l3(x)
        return q_values
