from collections import deque
import random
import numpy as np
import torch
import math

class ReplayBuffer:
    def __init__(self, buffer_size, batch_size):
        self.buffer = deque(maxlen=buffer_size)
        self.batch_size = batch_size

    def add(self, state, action, reward, next_state, done, discount=1.0):
        data = (state, action, reward, next_state, done, discount)
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
        discounts = torch.tensor(
            [x[5] for x in data], dtype=torch.float32
        )

        return states, actions, rewards, next_states, dones, discounts

class PERBuffer(ReplayBuffer):
    def __init__(self, buffer_size, batch_size, alpha=0.6, beta=0.4):
        super().__init__(buffer_size, batch_size)
        self.alpha = alpha
        self.beta = beta
        self.epsilon = 1e-6
        self.priorities = deque(maxlen=buffer_size)

    def add(
        self, state, action, reward, next_state, done, td_error, discount=1.0
    ):
        super().add(state, action, reward, next_state, done, discount)
        priority = (math.fabs(td_error) + self.epsilon)
        self.priorities.append(priority)

    def get_batch(self):
        priorities = np.array(self.priorities)
        probabilities = priorities ** self.alpha
        probabilities /= probabilities.sum()

        indices = np.random.choice(
            len(self.buffer), size=self.batch_size, p=probabilities
        )
        data = [self.buffer[idx] for idx in indices]

        states = torch.as_tensor(
            np.stack([x[0] for x in data]), dtype=torch.float32
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
        discounts = torch.tensor(
            [x[5] for x in data], dtype=torch.float32
        )
        weights = (len(self.buffer) * probabilities[indices]) ** (-self.beta)
        weights /= weights.max()
        weights = torch.tensor(weights, dtype=torch.float32)

        return states, actions, rewards, next_states, dones, discounts, weights
