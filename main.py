import math

import gymnasium as gym
import numpy as np
import pygame
import torch

from dqn_agent import DQNAgent


def render_agent(agent, episode):
    env = gym.make("Acrobot-v1", render_mode="rgb_array")

    pygame.init()

    state, _ = env.reset(seed=42)
    frame = env.render()
    height, width = frame.shape[:2]

    panel_width = 310
    screen = pygame.display.set_mode(
        (width + panel_width, height)
    )
    pygame.display.set_caption(
        f"Acrobot DQN - Episode {episode}"
    )

    font = pygame.font.Font(None, 24)
    clock = pygame.time.Clock()
    fps = env.metadata["render_fps"]

    qnet = agent.qnet
    device = next(qnet.parameters()).device

    step_count = 0
    total_reward = 0.0

    def write(text, x, y, color=(235, 240, 245)):
        image = font.render(str(text), True, color)
        screen.blit(image, (x, y))

    try:
        running = True
        done = False

        while running and not done:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False

                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        running = False

            if not running:
                break

            # Evaluate Q-values without changing parameters.
            with torch.inference_mode():
                x = torch.as_tensor(
                    state,
                    dtype=torch.float32,
                    device=device
                ).unsqueeze(0)

                qs = qnet(x)[0].cpu().numpy()

            action = int(np.argmax(qs))

            # Gymnasium's original physics rendering.
            frame = env.render()

            # Reuse the internal Pygame Surface.
            source = getattr(env.unwrapped, "screen", None)

            if isinstance(source, pygame.Surface):
                screen.blit(source, (0, 0))
            else:
                surface = pygame.surfarray.make_surface(
                    frame.swapaxes(0, 1)
                )
                screen.blit(surface, (0, 0))

            # Information panel
            pygame.draw.rect(
                screen,
                (28, 36, 49),
                (width, 0, panel_width, height)
            )

            left = width + 20

            theta1 = math.atan2(state[1], state[0])
            theta2 = math.atan2(state[3], state[2])

            tip_height = (
                -math.cos(theta1)
                -math.cos(theta1 + theta2)
            )

            write("ACROBOT DQN", left, 20)
            write(f"Training episode: {episode}", left, 55)
            write(f"Evaluation step: {step_count}", left, 85)
            write(f"Reward: {total_reward:.0f}", left, 115)

            write("PHYSICS", left, 160, (80, 220, 165))
            write(f"Theta 1: {math.degrees(theta1):+.1f} deg",left, 195)
            write(f"Theta 2: {math.degrees(theta2):+.1f} deg",left, 225)
            write(f"Omega 1: {state[4]:+.2f} rad/s",left, 255)
            write(f"Omega 2: {state[5]:+.2f} rad/s",left, 285)
            write(f"Tip height: {tip_height:+.2f} m",left, 315)

            write("Q VALUES", left, 355, (80, 220, 165))

            for i in range(3):
                selected = i == action
                color = (
                    (80, 220, 165)
                    if selected
                    else (235, 240, 245)
                )
                marker = ">" if selected else " "

                write(
                    f"{marker} Torque {i - 1:+d}: {qs[i]:+.2f}",
                    left, 390 + i * 30, color
                )

            write("ESC: Return to training", left, 475)

            pygame.display.flip()
            clock.tick(fps)

            # Advance the original physics simulation.
            state, reward, terminated, truncated, _ = env.step(action)

            step_count += 1
            total_reward += reward
            done = terminated or truncated

        print(
            f"Evaluation after episode {episode}: "
            f"Steps={step_count}, "
            f"Reward={total_reward:.0f}"
        )

    finally:
        env.close()
        pygame.quit()


def main():
    env = gym.make("Acrobot-v1")

    agent = DQNAgent(
        state_size=env.observation_space.shape[0],
        action_size=env.action_space.n
    )

    episodes = 1000
    sync_interval = 100
    reward_history = []

    try:
        for episode in range(1, episodes + 1):
            state, _ = env.reset()
            episode_done = False
            total_reward = 0.0

            while not episode_done:
                action = agent.get_action(state)

                next_state, reward, terminated, truncated, _ = env.step(action)

                agent.update(state,action,reward,next_state,terminated)

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

                render_agent(agent, episode)

    finally:
        env.close()

if __name__ == "__main__":
    main()
