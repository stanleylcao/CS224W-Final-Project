import torch
from models.gcn import GCN
from models.gat import GAT
from models.graphSAGE import GraphSAGE
from game import Environment
from dqn import DQN
from config import config
import copy
from IPython.display import HTML

def main():
    # --- Initialize Environment ---
    env = Environment()

    # --- Initialize GNN Model ---
    if config["gnn_type"] == "GCN":
        model = GCN()
    elif config["gnn_type"] == "GAT":
        model = GAT()
    elif config["gnn_type"] == "GraphSAGE":
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

    env.reset()

    # List to store graph states for visualization
    graph_states = []

    state = copy.deepcopy(env.get_state())
    num_games_played = 0
    while dqn.memory.length() < dqn.batch_size:
        done = False
        # print('BEGIN--------------------')
        counter = 0
        while not done:
            pacman_action = dqn.pacman_act(env)
            ghost_action = dqn.act(env)
            next_state, reward, done, score = env.step(
                pacman_action, ghost_action)
            env.dump()

            # Collect the current state of the graph to save for visualization
            graph_states.append({
                "game_number": num_games_played,
                "graph": copy.deepcopy(env.field.graph),
                "pacman_positions": [env.pacman.get_pos(i) for i in range(env.num_pacman)],
                "ghost_positions": [env.ghosts.get_pos(i) for i in range(env.num_ghosts)],
            })

            # print(done)
            print('-' * 20)
            dqn.remember(state, copy.deepcopy(ghost_action), reward, copy.deepcopy(next_state), done)
            state = copy.deepcopy(next_state)
        num_games_played += 1
        env.reset()

    print("num games played: ", num_games_played)
    print("creating animation...")
    # Create the animation
    # anim = env.create_animation(graph_states)
    # Display or save the animation
    # anim.save("graph_animation.gif", writer="pillow")
    print("animation created")

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
