import torch

config = {
    # General settings
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "num_episodes": 1,
    "max_steps": 50,

    # Environment settings
    "num_pacman": 1,
    "num_ghosts": 2,
    "pacman_spawn_pos": [5],
    "ghosts_spawn_pos": [18, 20, 23, 21, 22],
    "pacman_idx_start": 0,
    "ghosts_idx_start": 1,

    # Game settings
    "action_space": [0, 1, 2],
    "punishment": -1,
    "reward": 1,
    "loss_score": float('-inf'),
    "win_score": 5,
    "max_val": 2,

    # Graph settings
    "edges": [
        [0, 1], [0, 10], [1, 2], [2, 3], [2, 17], [3, 4], [4, 5],
        [5, 6], [5, 19], [6, 7], [7, 8], [8, 9], [8, 11], [9, 10],
        [11, 12], [12, 13], [13, 14], [14, 15], [15, 16], [16, 17],
        [18, 19], [18, 23], [19, 20], [19, 22], [20, 21], [21, 22], [22, 23]
    ],

    # DQN settings
    "state_shape": (10, 3),  # Example placeholder
    "action_size": 4,
    "learning_rate_max": 0.001,
    "learning_rate_decay": 0.995,
    "gamma": 0.75,
    "memory_size": 2000,
    "batch_size": 32,
    "exploration_max": 1.0,
    "exploration_min": 0.01,
    "exploration_decay": 0.995,

    # Model settings
    "gnn_type": "GCN",  # Change to "GAT" for testing GATModel
    "input_dim": 3,
    "hidden_dim": 128,
    # Note that this isn't the action set, but rather the final node embeddings
    "output_dim": 4,
}
