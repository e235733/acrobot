import unittest

from dqn_agent import DQNAgent


class NStepDQNTests(unittest.TestCase):
    def setUp(self):
        self.agent = DQNAgent(use_per=False, n_steps=3)

    @staticmethod
    def state(value):
        return [float(value)] * 6

    def add_step(self, index, reward, terminated=False, episode_end=None):
        self.agent.update(
            self.state(index),
            0,
            reward,
            self.state(index + 1),
            terminated,
            episode_end=episode_end
        )

    def test_emits_n_step_reward_and_discount(self):
        self.add_step(0, 1.0)
        self.add_step(1, 2.0)
        self.add_step(2, 3.0)

        transition = self.agent.replay_buffer.buffer[0]
        self.assertAlmostEqual(transition[2], 1 + 0.99 * 2 + 0.99**2 * 3)
        self.assertEqual(transition[3], self.state(3))
        self.assertFalse(transition[4])
        self.assertAlmostEqual(transition[5], 0.99**3)

    def test_flushes_shorter_returns_at_termination(self):
        self.add_step(0, 1.0)
        self.add_step(1, 2.0)
        self.add_step(2, 3.0, terminated=True)

        transitions = list(self.agent.replay_buffer.buffer)
        self.assertEqual(len(transitions), 3)
        self.assertAlmostEqual(transitions[1][2], 2 + 0.99 * 3)
        self.assertAlmostEqual(transitions[1][5], 0.99**2)
        self.assertTrue(all(transition[4] for transition in transitions))
        self.assertEqual(len(self.agent.n_step_buffer), 0)

    def test_truncation_flushes_but_keeps_bootstrap(self):
        self.add_step(0, 1.0)
        self.add_step(1, 2.0, episode_end=True)

        transitions = list(self.agent.replay_buffer.buffer)
        self.assertEqual(len(transitions), 2)
        self.assertAlmostEqual(transitions[0][2], 1 + 0.99 * 2)
        self.assertAlmostEqual(transitions[0][5], 0.99**2)
        self.assertFalse(transitions[0][4])
        self.assertFalse(transitions[1][4])


if __name__ == "__main__":
    unittest.main()
