import random
import torch

from replay_buffer import ReplayBuffer, PERBuffer
from qnet import QNet

class DQNAgent:
    def __init__(self, state_size=6, action_size=3, dueling=False, use_per=True):
        self.gamma = 0.99
        self.lr = 0.001
        self.batch_size = 128

        self.epsilon_start = 1.0
        self.epsilon_end = 0.05
        self.epsilon_decay_steps = 20000
        self.epsilon = self.epsilon_start
        self.steps_done = 0

        self.action_size = action_size
        self.loss_history = []

        self.use_per = use_per
        buffer_cls = PERBuffer if use_per else ReplayBuffer
        self.replay_buffer = buffer_cls(20000, self.batch_size)

        self.qnet = QNet(state_size, action_size, dueling=dueling)
        self.qnet_target = QNet(state_size, action_size, dueling=dueling)

        self.optimizer = torch.optim.Adam(self.qnet.parameters(), lr=self.lr)

        self.sync_qnet()

    def _decay_epsilon(self):
        progress = min(1.0, self.steps_done / self.epsilon_decay_steps)
        self.epsilon = (
            self.epsilon_start
            + (self.epsilon_end - self.epsilon_start) * progress
        )
        self.steps_done += 1

    def get_action(self, state):
        self._decay_epsilon()

        if random.random() < self.epsilon:
            return random.randrange(self.action_size)

        with torch.no_grad():
            x = torch.as_tensor(
                state, dtype=torch.float32
            ).unsqueeze(0)
            q_values = self.qnet(x)
            return int(q_values.argmax(dim=1).item())

    def _compute_td_error(self, state, action, reward, next_state, done):
        with torch.no_grad():
            s = torch.as_tensor(state, dtype=torch.float32).unsqueeze(0)
            ns = torch.as_tensor(next_state, dtype=torch.float32).unsqueeze(0)

            q = self.qnet(s)[0, action]
            next_q = self.qnet_target(ns).max(dim=1).values[0]
            target = reward + (1 - done) * self.gamma * next_q

        return float((target - q).item())

    def update(self, state, action, reward, next_state, done):
        if self.use_per:
            td_error = self._compute_td_error(
                state, action, reward, next_state, done
            )
            self.replay_buffer.add(
                state, action, reward, next_state, done, td_error
            )
        else:
            self.replay_buffer.add(
                state, action, reward, next_state, done
            )

        if len(self.replay_buffer) < self.batch_size:
            return

        if self.use_per:
            states, actions, rewards, next_states, dones, weights = (
                self.replay_buffer.get_batch()
            )
        else:
            states, actions, rewards, next_states, dones = (
                self.replay_buffer.get_batch()
            )
            weights = 1.0

        q_values = self.qnet(states)
        q = q_values.gather(
            1, actions.unsqueeze(1)
        ).squeeze(1)

        with torch.no_grad():
            next_q_values = self.qnet_target(next_states)
            next_q = next_q_values.max(dim=1).values
            target = rewards + (1 - dones) * self.gamma * next_q

        loss = (weights * (q - target) ** 2).mean()

        self.loss_history.append(loss.item())

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def sync_qnet(self):
        self.qnet_target.load_state_dict(
            self.qnet.state_dict()
        )

class DoubleDQNAgent(DQNAgent):
    def __init__(self, state_size=6, action_size=3, dueling=False, use_per=True):
        super().__init__(state_size, action_size, dueling, use_per)

    def _compute_td_error(self, state, action, reward, next_state, done):
        with torch.no_grad():
            s = torch.as_tensor(state, dtype=torch.float32).unsqueeze(0)
            ns = torch.as_tensor(next_state, dtype=torch.float32).unsqueeze(0)

            q = self.qnet(s)[0, action]
            next_action = self.qnet(ns).argmax(dim=1)
            next_q = self.qnet_target(ns).gather(
                1, next_action.unsqueeze(1)
            ).squeeze(1)[0]
            target = reward + (1 - done) * self.gamma * next_q

        return float((target - q).item())

    def update(self, state, action, reward, next_state, done):
        if self.use_per:
            td_error = self._compute_td_error(
                state, action, reward, next_state, done
            )
            self.replay_buffer.add(
                state, action, reward, next_state, done, td_error
            )
        else:
            self.replay_buffer.add(
                state, action, reward, next_state, done
            )

        if len(self.replay_buffer) < self.batch_size:
            return

        if self.use_per:
            states, actions, rewards, next_states, dones, weights = (
                self.replay_buffer.get_batch()
            )
        else:
            states, actions, rewards, next_states, dones = (
                self.replay_buffer.get_batch()
            )
            weights = 1.0

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

        loss = (weights * (q - target) ** 2).mean()

        self.loss_history.append(loss.item())

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()