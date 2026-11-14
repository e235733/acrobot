
import math

import gymnasium as gym
import numpy as np
import pygame
import torch

from dqn_agent import DQNAgent


def render_agent(agent):
    env = gym.make("Acrobot-v1", render_mode="rgb_array")

    qnet = agent.qnet
    was_training = qnet.training
    qnet.eval()
    device = next(qnet.parameters()).device

    pygame.init()

    state, _ = env.reset()
    frame = env.render()
    height, width = frame.shape[:2]

    panel_width = 300
    screen = pygame.display.set_mode(
        (width + panel_width, height)
    )
    pygame.display.set_caption("Acrobot DQN")

    font = pygame.font.Font(None, 25)
    clock = pygame.time.Clock()
    fps = env.metadata["render_fps"]  # Default: 15

    step_count = 0
    total_reward = 0.0
    last_action = None

    def write(text, x, y, color=(235, 240, 245)):
        image = font.render(str(text), True, color)
        screen.blit(image, (x, y))

    try:
        running = True

        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        running = False

            if not running:
                break

            # One forward pass for the current state.
            # No gradients, hooks, or parameter updates.
            with torch.inference_mode():
                x = torch.as_tensor(
                    state,
                    dtype=torch.float32,
                    device=device
                ).unsqueeze(0)

                qs = qnet(x)[0].cpu().numpy()

            action = int(np.argmax(qs))

            # Use Gymnasium's original Acrobot renderer.
            frame = env.render()

            # Reuse its Pygame Surface when available.
            source = getattr(env.unwrapped, "screen", None)

            if isinstance(source, pygame.Surface):
                screen.blit(source, (0, 0))
            else:
                surface = pygame.surfarray.make_surface(
                    frame.swapaxes(0, 1)
                )
                screen.blit(surface, (0, 0))

            # Simple information panel.
            pygame.draw.rect(
                screen, (28, 36, 49),
                (width, 0, panel_width, height)
            )

            left = width + 20

            theta1 = math.atan2(state[1], state[0])
            theta2 = math.atan2(state[3], state[2])
            tip_height = (
                -math.cos(theta1)
                -math.cos(theta1 + theta2)
            )

            write("ACROBOT MONITOR", left, 20)
            write(f"Step: {step_count}", left, 55)
            write(f"Reward: {total_reward:.0f}", left, 82)

            write("PHYSICS", left, 125, (80, 220, 165))
            write(
                f"Theta 1: {math.degrees(theta1):+.1f} deg",
                left, 160
            )
            write(
                f"Theta 2: {math.degrees(theta2):+.1f} deg",
                left, 190
            )
            write(f"Omega 1: {state[4]:+.2f} rad/s", left, 220)
            write(f"Omega 2: {state[5]:+.2f} rad/s", left, 250)
            write(f"Tip height: {tip_height:+.2f} m", left, 280)

            write("Q VALUES", left, 330, (80, 220, 165))

            for i in range(3):
                color = (
                    (80, 220, 165)
                    if i == action
                    else (235, 240, 245)
                )

                marker = ">" if i == action else " "
                write(
                    f"{marker} Torque {i - 1:+d}: {qs[i]:+.2f}",
                    left, 365 + i * 30, color
                )

            torque = "-" if last_action is None else f"{last_action - 1:+d}"
            write(f"Last torque: {torque}", left, 465)

            pygame.display.flip()

            # Match Gymnasium's original rendering FPS.
            # No skipped steps or custom physics.
            clock.tick(fps)

            # Take exactly one action.
            state, reward, terminated, truncated, _ = (
                env.step(action)
            )

            last_action = action
            step_count += 1
            total_reward += reward

            if terminated or truncated:
                print(
                    f"Evaluation: "
                    f"Steps={step_count}, "
                    f"Reward={total_reward:.0f}"
                )
                break

    finally:
        qnet.train(was_training)
        env.close()
        pygame.quit()


def main():
    env = gym.make("Acrobot-v1")
    agent = DQNAgent(
        state_size=env.observation_space.shape[0],
        action_size=env.action_space.n
    )

    episodes = 100
    sync_interval = 100
    reward_history = []

    try:
        for episode in range(1, episodes + 1):
            state, _ = env.reset()
            episode_done = False
            total_reward = 0

            while not episode_done:
                action = agent.get_action(state)

                next_state, reward, terminated, truncated, _ = (
                    env.step(action)
                )

                agent.update(
                    state, action, reward,
                    next_state, terminated
                )

                state = next_state
                total_reward += reward
                episode_done = terminated or truncated

            reward_history.append(total_reward)

            if episode % sync_interval == 0:
                agent.sync_qnet()

                recent = reward_history[-sync_interval:]
                print(
                    f"Episode {episode}: "
                    f"Reward={total_reward:.0f}, "
                    f"Average={np.mean(recent):.1f}"
                )

    finally:
        env.close()

    # Show the trained agent once after training.
    render_agent(agent)


if __name__ == "__main__":
    main()
