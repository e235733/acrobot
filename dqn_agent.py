import random
import torch
import torch.nn.functional as F

from replay_buffer import ReplayBuffer
from qnet import QNet

class DQNAgent:
    def __init__(self, state_size=6, action_size=3):
        self.gamma = 0.98
        self.lr = 0.005
        self.epsilon = 0.05
        self.batch_size = 512

        self.action_size = action_size

        self.replay_buffer = ReplayBuffer(10000, self.batch_size)

        self.qnet = QNet(state_size, action_size)
        self.qnet_target = QNet(state_size, action_size)

        self.optimizer = torch.optim.Adam(
            self.qnet.parameters(), lr=self.lr
        )

        self.sync_qnet()

    def get_action(self, state):
        if random.random() < self.epsilon:
            return random.randrange(self.action_size)

        with torch.no_grad():
            x = torch.as_tensor(
                state, dtype=torch.float32
            ).unsqueeze(0)
            q_values = self.qnet(x)
            return int(q_values.argmax(dim=1).item())

    def update(self, state, action, reward, next_state, done):
        self.replay_buffer.add(
            state, action, reward, next_state, done
        )

        if len(self.replay_buffer) < self.batch_size:
            return

        states, actions, rewards, next_states, dones = (
            self.replay_buffer.get_batch()
        )

        q_values = self.qnet(states)
        q = q_values.gather(
            1, actions.unsqueeze(1)
        ).squeeze(1)

        with torch.no_grad():
            next_q_values = self.qnet_target(next_states)
            next_q = next_q_values.max(dim=1).values
            target = rewards + (1 - dones) * self.gamma * next_q

        loss = F.mse_loss(q, target)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def sync_qnet(self):
        self.qnet_target.load_state_dict(
            self.qnet.state_dict()
        )

class DoubleDQNAgent(DQNAgent):
    def __init__(self, state_size=6, action_size=3):
        super().__init__(state_size, action_size)
        self.qnet = QNet(state_size, action_size, dueling=True)
        self.qnet_target = QNet(state_size, action_size, dueling=True)
        self.optimizer = torch.optim.Adam(
            self.qnet.parameters(), lr=self.lr
        )
        self.sync_qnet()

    def update(self, state, action, reward, next_state, done):
        self.replay_buffer.add(
            state, action, reward, next_state, done
        )

        if len(self.replay_buffer) < self.batch_size:
            return

        states, actions, rewards, next_states, dones = (
            self.replay_buffer.get_batch()
        )

        q_values = self.qnet(states)
        q = q_values.gather(
            1, actions.unsqueeze(1)
        ).squeeze(1)

        with torch.no_grad():
            next_q_values = self.qnet(next_states)
            next_actions = next_q_values.argmax(dim=1)
            next_q_target_values = self.qnet_target(next_states)
            next_q = next_q_target_values.gather(
                1, next_actions.unsqueeze(1)
            ).squeeze(1)
            target = rewards + (1 - dones) * self.gamma * next_q

        loss = F.mse_loss(q, target)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()