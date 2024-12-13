"""
main.py
Runs the training loop, evaluates performance, and visualizes results.

1. Initializes the environment and GNN model (based on config).
2. Runs a training loop for DQN:
  a) Collects experiences by interacting with the environment.
  b) Trains the model using replay buffer samples.
  c) Updates the target network periodically.
3. Saves the model and generates a graph animation of the last episodes.

"""

import torch
from models.gcn import GCN
from models.gat import GATModel
from models.graphSAGE import GraphSAGE
from game import Environment
from dqn import DQN
from config import config
import copy
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import linregress


def sim_random(num_episodes):
    env = Environment()
    max_steps = config["max_steps"]
    graph_states = []
    all_rewards = []  # Track total rewards per episode

    for episode in range(num_episodes):
        state = env.reset()
        state = copy.deepcopy(state)
        total_reward = 0
        done = False

        for step in range(max_steps):
            # Select actions
            pacman_action_set = env.get_pacman_action_set()
            pacman_action = pacman_action_set[torch.randint(
                len(pacman_action_set), (1,)).item()]

            ghost_actions = env.get_ghost_action_set()
            random_action_idx = np.random.randint(len(ghost_actions))
            ghost_action = ghost_actions[random_action_idx]

            # Take a step in the environment
            next_state, reward, done, score = env.step(
                pacman_action, ghost_action)

            # Save graph state for visualization
            graph_states.append({
                "episode": episode,
                "graph": copy.deepcopy(env.field.graph),
                "pacman_positions": [env.pacman.get_pos(i) for i in range(env.num_pacman)],
                "ghost_positions": [env.ghosts.get_pos(i) for i in range(env.num_ghosts)],
            })

            # Store experience in replay buffer
            state = copy.deepcopy(next_state)

            # Accumulate reward
            total_reward += reward

            # Break if game ends
            if done:
                break
        # Log rewards
        all_rewards.append(reward)
        if episode % 10 == 0:
            print(f"Episode {episode}/{num_episodes} - Reward: {reward:.2f}")

    plt.figure(figsize=(12, 6))

    # Plot the raw losses
    plt.plot(all_rewards, color='blue', linewidth=1.5, label="Training Loss")

    # Calculate and plot the trendline
    x = np.arange(len(all_rewards))  # Replay steps as x-axis values
    y = np.array(all_rewards)  # Losses as y-axis values
    slope, intercept, _, _, _ = linregress(x, y)  # Linear regression
    trendline = slope * x + intercept  # Calculate trendline
    plt.plot(x, trendline, color='red', linestyle='--',
             linewidth=2, label=f"Trendline (slope={slope:.5f})")

    # Labels and title
    plt.xlabel("Replay Steps", fontsize=14)
    plt.ylabel("Loss", fontsize=14)
    plt.title("Training Loss Over Time", fontsize=16)

    # Grid and legend
    plt.grid(alpha=0.4)
    plt.legend(fontsize=12)
    plt.tight_layout()
    plt.show()


def main():
    # --- Initialize Environment ---
    env = Environment()

    # --- Initialize GNN Model ---
    if config["gnn_type"] == "GCN":
        model = GCN()
    elif config["gnn_type"] == "GAT":
        model = GATModel()
    elif config["gnn_type"] == "GraphSage":
        model = GraphSAGE()
    else:
        raise ValueError(f"Unsupported GNN type: {config['gnn_type']}")

    # --- Initialize DQN ---
    dqn = DQN(
        model=model,
        state_shape=config["state_shape"],
        action_size=config["action_size"],
        learning_rate_max=config["learning_rate_max"],
        learning_rate_decay=config["learning_rate_decay"],
        gamma=config["gamma"],
        memory_size=config["memory_size"],
        batch_size=config["batch_size"],
        exploration_max=config["exploration_max"],
        exploration_min=config["exploration_min"],
        exploration_decay=config["exploration_decay"],
    )

    # Move models to the same device
    device = torch.device(config["device"])
    dqn.model = dqn.model.to(device)
    dqn.target_model = dqn.target_model.to(device)

    # --- Training Loop ---
    num_episodes = config["num_episodes"]
    max_steps = config["max_steps"]
    target_update_freq = config["target_update_freq"]
    graph_states = []

    all_rewards = []  # Track total rewards per episode
    all_losses = []  # Track total losses per episode

    for episode in range(num_episodes):
        state = env.reset()
        state = copy.deepcopy(state)
        total_reward = 0
        done = False

        for step in range(max_steps):
            # # Select actions
            pacman_action = dqn.pacman_act(env)
            ghost_action = dqn.act(env)

            # Heuristic as baseline
            # ghost_action = dqn.ghost_heuristic_action(env)

            # Take a step in the environment
            next_state, reward, done, score = env.step(
                pacman_action, ghost_action)

            # Save graph state for visualization
            graph_states.append({
                "episode": episode,
                "graph": copy.deepcopy(env.field.graph),
                "pacman_positions": [env.pacman.get_pos(i) for i in range(env.num_pacman)],
                "ghost_positions": [env.ghosts.get_pos(i) for i in range(env.num_ghosts)],
            })

            # Store experience in replay buffer
            dqn.remember(state, copy.deepcopy(ghost_action),
                         reward, copy.deepcopy(next_state), done)
            state = copy.deepcopy(next_state)

            # Accumulate reward
            total_reward += reward

            # Break if game ends
            if done:
                break

        # Train the model after collecting enough experiences
        if dqn.memory.length() >= 5 * dqn.batch_size:
            losses = dqn.replay(env)
            all_losses.extend(losses)

        # Update the target network periodically
        if episode % target_update_freq == 0:
            dqn.update_target_model()

        # Decay epsilon
        dqn.exploration_rate = max(
            dqn.exploration_min, dqn.exploration_rate * dqn.exploration_decay)

        # Log rewards
        all_rewards.append(reward)
        if episode % 10 == 0:
            print(f"Episode {episode}/{num_episodes} - Reward: {reward:.2f}")

    # --- Save Animation ---
    print("Creating animation...")
    last_5_episodes = list(range(num_episodes - 5, num_episodes))
    # Filter graph_states to include only the last 5 episodes
    last_5_graph_states = [
        gs for gs in graph_states if gs["episode"] in last_5_episodes
    ]
    # Create the animation with the filtered states
    anim = env.create_animation(last_5_graph_states)
    anim.save("graph_animation.gif", writer="pillow")
    print("Animation saved.")

    # --- Plot the Losses ---
    plt.figure(figsize=(12, 6))

    # Plot the raw losses
    plt.plot(all_losses, color='blue', linewidth=1.5, label="Training Loss")

    # Calculate and plot the trendline
    x = np.arange(len(all_losses))  # Replay steps as x-axis values
    y = np.array(all_losses)  # Losses as y-axis values
    slope, intercept, _, _, _ = linregress(x, y)  # Linear regression
    trendline = slope * x + intercept  # Calculate trendline
    plt.plot(x, trendline, color='red', linestyle='--',
             linewidth=2, label=f"Trendline (slope={slope:.5f})")

    # Labels and title
    plt.xlabel("Replay Steps", fontsize=14)
    plt.ylabel("Loss", fontsize=14)
    plt.title("Training Loss Over Time", fontsize=16)

    # Grid and legend
    plt.grid(alpha=0.4)
    plt.legend(fontsize=12)
    plt.tight_layout()

    # Save the plot
    plt.savefig("loss_plot.png", dpi=300)
    plt.show()

    # --- Plot the Rewards ---
    plt.figure(figsize=(12, 6))

    # Plot the rewards
    plt.plot(all_rewards, color='green', linewidth=1.5, label="Rewards")

    # Calculate and plot the trendline for rewards
    x_rewards = np.arange(len(all_rewards))
    y_rewards = np.array(all_rewards)
    slope_rewards, intercept_rewards, _, _, _ = linregress(
        x_rewards, y_rewards)
    trendline_rewards = slope_rewards * x_rewards + intercept_rewards
    plt.plot(x_rewards, trendline_rewards, color='orange', linestyle='--',
             linewidth=2, label=f"Trendline (slope={slope_rewards:.5f})")

    # Labels and title
    plt.xlabel("Episodes", fontsize=14)
    plt.ylabel("Total Reward", fontsize=14)
    plt.title("Total Reward Over Episodes", fontsize=16)

    # Grid and legend
    plt.grid(alpha=0.4)
    plt.legend(fontsize=12)
    plt.tight_layout()

    # Save the plot
    plt.savefig("rewards_plot.png", dpi=300)
    plt.show()


if __name__ == "__main__":
    # sim_random(5000)
    main()
