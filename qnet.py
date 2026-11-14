import torch
import torch.nn as nn
import torch.nn.functional as F

class QNet(nn.Module):
    def __init__(self, state_size, action_size):
        super().__init__()

        self.l1 = nn.Linear(state_size, 64)
        self.l2 = nn.Linear(64, 32)
        self.l3 = nn.Linear(32, action_size)

    def forward(self, x):
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        x = self.l3(x)
        return x
