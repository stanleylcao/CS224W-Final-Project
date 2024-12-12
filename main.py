import torch
from models.gcn import GCN
from models.gat import GAT
from models.graphSAGE import GraphSAGE
from game import Environment
from dqn import DQN
from config import config
import copy
from IPython.display import HTML
import matplotlib.pyplot as plt
import numpy as np

def main():
    # --- Initialize Environment ---
    env = Environment()

    # --- Initialize GNN Model ---
    if config["gnn_type"] == "GCN":
        model = GCN()
    elif config["gnn_type"] == "GAT":
        model = GAT()
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
    all_losses = [] # Track total losses per episode

    for episode in range(num_episodes):
        state = env.reset()
        state = copy.deepcopy(state)
        total_reward = 0
        done = False

        for step in range(max_steps):
            # Select actions
            pacman_action = dqn.pacman_act(env)
            ghost_action = dqn.act(env)

            # Take a step in the environment
            next_state, reward, done, score = env.step(pacman_action, ghost_action)

            # Save graph state for visualization
            graph_states.append({
                "episode": episode,
                "graph": copy.deepcopy(env.field.graph),
                "pacman_positions": [env.pacman.get_pos(i) for i in range(env.num_pacman)],
                "ghost_positions": [env.ghosts.get_pos(i) for i in range(env.num_ghosts)],
            })

            # Store experience in replay buffer
            dqn.remember(state, copy.deepcopy(ghost_action), reward, copy.deepcopy(next_state), done)
            state = copy.deepcopy(next_state)

            # Accumulate reward
            total_reward += reward

            # Break if game ends
            if done:
                break

        # Train the model after collecting enough experiences
        if dqn.memory.length() >= dqn.batch_size:
            losses = dqn.replay(env)
            all_losses.extend(losses)

        # Update the target network periodically
        if episode % target_update_freq == 0:
            dqn.update_target_model()

        # Decay epsilon
        dqn.exploration_rate = max(dqn.exploration_min, dqn.exploration_rate * dqn.exploration_decay)

        # Log rewards
        all_rewards.append(total_reward)
        if episode % 10 == 0:
            print(f"Episode {episode}/{num_episodes} - Reward: {total_reward}")

    print(f"Unique states in replay buffer: {len(set([str(exp[0]) for exp in dqn.memory.data]))}")

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
    # plt.plot(all_losses)
    # plt.xlabel("Replay Steps")
    # plt.ylabel("Loss")
    # plt.title("Training Loss Over Time")
    # plt.savefig("loss_plot.png", dpi=300) 
    # plt.show()
    # Plot the losses
    plt.figure(figsize=(12, 6))
    plt.plot(all_losses, color='blue', alpha=0.7, label="Raw Loss")

    # Optional: Add a smoothed version of the loss curve
    window_size = 50  # Adjust the window size for smoothing
    if len(all_losses) > window_size:
        smoothed_losses = np.convolve(all_losses, np.ones(window_size)/window_size, mode='valid')
        plt.plot(range(window_size-1, len(all_losses)), smoothed_losses, color='red', linewidth=2, label="Smoothed Loss")

    # Labels and title
    plt.xlabel("Replay Steps", fontsize=14)
    plt.ylabel("Loss", fontsize=14)
    plt.title("Training Loss Over Time", fontsize=16)

    # Grid, legend, and style
    plt.grid(alpha=0.4)
    plt.legend(fontsize=12)
    plt.tight_layout()

    # Save the plot
    plt.savefig("loss_plot.png", dpi=300)
    plt.show()

    # --- Save Model ---
    torch.save(dqn.model.state_dict(), "dqn_model.pth")
    print("Model saved.")

if __name__ == "__main__":
    main()
