"""
config.py
Defines global configuration values and hyperparameters for the project:
- General settings: Device (GPU/CPU), number of episodes, steps per episode.
- Environment: Number of Pac-Man agents, ghosts, their spawn positions, and grid edges.
- DQN: Parameters for learning, exploration, and replay memory.
- Model: Specifies the GNN type (e.g., GAT, GraphSAGE) and architecture settings.
"""

import torch

config = {
    # General settings
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "num_episodes": 500,
    "max_steps": 8,

    # Environment settings
    "num_pacman": 1,
    "num_ghosts": 2,
    "pacman_spawn_pos": [5],
    "ghosts_spawn_pos": [18, 20, 23, 21, 22],
    "pacman_idx_start": 0,
    "ghosts_idx_start": 1,

    # Game settings
    "time_step_score": -0.1,          # Penalty for each time step
    # Reward for catching Pac-Man (ghosts' perspective)
    "win_score": 1.0,
    "loss_score": -1.0,                # Penalty if Pac-Man wins
    "distance_reward_scale": 0.1,      # Reward for reducing distance to Pac-Man

    # Graph settings
    "edges": [
        [0, 1], [0, 10], [1, 2], [2, 3], [2, 17], [2, 18], [3, 4], [4, 5],
        [5, 6], [5, 19], [6, 7], [7, 8], [8, 9], [8, 11], [8, 20], [9, 10],
        [11, 12], [12, 13], [13, 14], [14, 15], [15, 16], [16, 17],
        [18, 19], [18, 23], [19, 20], [20, 21], [21, 22], [22, 23]
    ],
    "num_nodes": 24,

    # DQN settings
    "state_shape": (10, 3),  # Example placeholder
    "action_size": 4,
    "learning_rate_max": 0.001,
    "learning_rate_decay": 0.995,
    "gamma": 0.85,
    "memory_size": 2000,
    "batch_size": 32,
    "exploration_max": 0.8,
    "exploration_min": 0.1,
    "exploration_decay": 0.995,

    # Model settings
    "gnn_type": "GCN",  # Options are "GAT", "GraphSage", or "GCN"
    "input_dim": 3,
    "hidden_dim": 128,
    # Note that this isn't the action set, but rather the final node embeddings
    "output_dim": 4,
    "num_layers": 3,
    "heads": 2,
    "dropout": 0.0,
    "negative_slope": 0.2,
    "target_update_freq": 5,  # Frequency of target network updates
}
