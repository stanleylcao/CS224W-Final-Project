import torch
from models.gcn import GCNModel
from models.gat import GATModel
from models.graphSAGE import GraphSAGEModel
from game import Environment
from dqn import DQN
from config import config


def main():
    # --- Initialize Environment ---
    env = Environment()

    # --- Initialize GNN Model ---
    if config["gnn_type"] == "GCN":
        model = GCNModel(
            input_dim=config["input_dim"], hidden_dim=config["hidden_dim"], output_dim=config["output_dim"])
    elif config["gnn_type"] == "GAT":
        model = GATModel(
            input_dim=config["input_dim"], hidden_dim=config["hidden_dim"], output_dim=config["output_dim"])
    elif config["gnn_type"] == "GraphSAGE":
        model = GraphSAGEModel(
            input_dim=config["input_dim"], hidden_dim=config["hidden_dim"], output_dim=config["output_dim"])
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

    env.reset()
    state = env.get_state()
    while dqn.memory.length() < dqn.batch_size:
        done = False
        # print('BEGIN--------------------')
        while not done:
            pacman_action = dqn.pacman_act(env)
            ghost_action = dqn.act(env)
            next_state, reward, done, score = env.step(
                pacman_action, ghost_action)
            env.dump()
            # print(done)
            print('-' * 20)
            # TODO: This is an issue because we are pushing references
            # rather than copies, which means all states will be the same
            # So even right here, state already equals next_state, which is bad.
            dqn.remember(state, ghost_action, reward, next_state, done)
            state = next_state
        env.reset()
    dqn.replay(env)
    return

    # --- Training Loop ---
    for episode in range(config["num_episodes"]):
        state = env.reset()  # Reset environment
        total_reward = 0
        done = False

        for step in range(config["max_steps"]):
            # Get graph data
            # state_graph = env.get_state()
            # node_features = state_graph.x.to(device)
            # edge_index = state_graph.edge_index.to(device)

            # Get action for pacman
            pacman_action = dqn.pacman_act(env)

            # Choose an action using the DQN
            ghost_action = dqn.act(env)

            # Take a step in the environment
            next_state, reward, done, _ = env.step(pacman_action, ghost_action)

            # Remember the experience
            dqn.remember(state, action, reward, next_state, done)

            # Update state
            state = next_state
            total_reward += reward

            # Train DQN
            loss = dqn.replay(episode)

            if done:
                break

        print(
            f"Episode {episode + 1}/{config['num_episodes']} - Total Reward: {total_reward}")

        # Update target network periodically
        if episode % 10 == 0:
            dqn.update_target_model()

    # --- Save Model ---
    dqn.save("dqn_model.pth")
    print("Model saved!")

    # --- Evaluation ---
    # TO DO


if __name__ == "__main__":
    main()
