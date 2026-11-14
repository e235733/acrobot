import math

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import pygame
import torch

from dqn_agent import DQNAgent, DoubleDQNAgent


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
                    f"{marker} Torque {i - 1:+d}: {qs[i]:+.4f}",
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


def plot_training_results(reward_history, loss_history, qnet):
    weights = np.concatenate([
        param.detach().cpu().numpy().ravel()
        for name, param in qnet.named_parameters()
        if "weight" in name
    ])

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    fig.patch.set_facecolor("#fcfcfb")

    panels = (
        (axes[0], reward_history, "Reward per episode", "Episode", "Reward", "#2a78d6", "plot"),
        (axes[1], loss_history, "Training loss", "Update step", "Loss", "#1baf7a", "plot"),
    )

    for ax, data, title, xlabel, ylabel, color, kind in panels:
        ax.plot(data, color=color, linewidth=1.5)
        ax.set_title(title, color="#0b0b0b")
        ax.set_xlabel(xlabel, color="#52514e")
        ax.set_ylabel(ylabel, color="#52514e")
        ax.set_facecolor("#fcfcfb")
        ax.grid(color="#e1e0d9", linewidth=0.8)
        for spine in ax.spines.values():
            spine.set_color("#c3c2b7")
        ax.tick_params(colors="#898781")

    ax = axes[2]
    ax.hist(weights, bins=50, color="#2a78d6")
    ax.set_title("QNet weight distribution", color="#0b0b0b")
    ax.set_xlabel("Weight value", color="#52514e")
    ax.set_ylabel("Count", color="#52514e")
    ax.set_facecolor("#fcfcfb")
    ax.grid(color="#e1e0d9", linewidth=0.8)
    for spine in ax.spines.values():
        spine.set_color("#c3c2b7")
    ax.tick_params(colors="#898781")

    fig.tight_layout()
    fig.savefig("training_results.png", dpi=150, facecolor=fig.get_facecolor())
    plt.show()


def main():
    env = gym.make("Acrobot-v1")

    agent = DQNAgent(
        state_size=env.observation_space.shape[0],
        action_size=env.action_space.n,
        dueling=True,
        use_per=False,
        n_steps=3
    )

    episodes = 500
    sync_interval_steps = 500
    render_interval_episodes = 100
    reward_history = []
    total_steps = 0

    try:
        for episode in range(1, episodes + 1):
            state, _ = env.reset()
            episode_done = False
            total_reward = 0.0

            while not episode_done:
                action = agent.get_action(state)

                next_state, reward, terminated, truncated, _ = env.step(action)

                agent.update(
                    state,
                    action,
                    reward,
                    next_state,
                    terminated,
                    episode_end=terminated or truncated
                )

                state = next_state
                total_reward += reward
                episode_done = terminated or truncated

                total_steps += 1
                if total_steps % sync_interval_steps == 0:
                    agent.sync_qnet()

            reward_history.append(total_reward)

            if episode % render_interval_episodes == 0:
                recent = reward_history[-render_interval_episodes:]
                print(
                    f"Episode {episode}: "
                    f"Reward={total_reward:.0f}, "
                    f"Average={np.mean(recent):.1f}"
                )

                render_agent(agent, episode)

        plot_training_results(reward_history, agent.loss_history, agent.qnet)

    finally:
        env.close()

if __name__ == "__main__":
    main()
