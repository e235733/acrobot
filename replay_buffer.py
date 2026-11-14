from collections import deque
import random
import numpy as np
import torch

class ReplayBuffer:
    def __init__(self, buffer_size, batch_size):
        self.buffer = deque(maxlen=buffer_size)
        self.batch_size = batch_size

    def add(self, state, action, reward, next_state, done):
        data = (state, action, reward, next_state, done)
        self.buffer.append(data)

    def __len__(self):
        return len(self.buffer)

    def get_batch(self):
        data = random.sample(self.buffer, self.batch_size)

        states = torch.as_tensor(
            np.stack([x[0] for x in data]),dtype=torch.float32
        )
        actions = torch.tensor(
            [x[1] for x in data], dtype=torch.long
        )
        rewards = torch.tensor(
            [x[2] for x in data], dtype=torch.float32
        )
        next_states = torch.as_tensor(
            np.stack([x[3] for x in data]), dtype=torch.float32
        )
        dones = torch.tensor(
            [x[4] for x in data], dtype=torch.float32
        )

        return states, actions, rewards, next_states, dones