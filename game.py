import torch
from torch_geometric.data import Data
from config import config
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import copy
import random
import numpy as np

class Field:
    """
    Represents the state of the game as a graph.
    """

    def __init__(self):
        self.num_pacman = config["num_pacman"]
        self.num_ghosts = config["num_ghosts"]
        self.pacman_idx_start = config["pacman_idx_start"]
        self.ghosts_idx_start = config["ghosts_idx_start"]
        self.pacman_spawn_pos = config["pacman_spawn_pos"]
        self.ghosts_spawn_pos = config["ghosts_spawn_pos"]

        # Create the graph
        edge_index = torch.tensor(config["edges"], dtype=torch.long)
        # add other direction to represent undirected graph
        flipped_edges = edge_index.flip(dims=[1])
        undirected_edge_index = torch.cat([edge_index, flipped_edges], dim=0)

        x = torch.zeros(
            size=(torch.max(edge_index) + 1, self.num_pacman + self.num_ghosts), dtype=torch.float
        )

        # Precompute distances between nodes in the graph
        self.distance_matrix = self.precompute_distances(edge_index, config["num_nodes"])

        # Initialize positions
        for i in range(self.num_pacman):
            x[self.pacman_spawn_pos[i], self.pacman_idx_start + i] = 1
        for i in range(self.num_ghosts):
            x[self.ghosts_spawn_pos[i], self.ghosts_idx_start + i] = 1

        self.graph = Data(
            x=x, edge_index=undirected_edge_index.t().contiguous())
        self.graph.validate(raise_on_error=True)

    def precompute_distances(self, edge_index, num_nodes):
        """
        Precomputes the shortest distances between all nodes using Floyd-Warshall.

        Args:
            edge_index (torch.Tensor): The edge list of the graph.
            num_nodes (int): The total number of nodes in the graph.

        Returns:
            np.ndarray: A distance matrix with the shortest distances between nodes.
        """
        # Initialize distance matrix
        distances = np.full((num_nodes, num_nodes), np.inf)
        np.fill_diagonal(distances, 0)  # Distance to self is 0

        # Populate direct distances
        for edge in edge_index.tolist():
            u, v = edge
            distances[u][v] = 1  # Assuming all edges have weight 1
            distances[v][u] = 1  # Undirected graph

        # Floyd-Warshall Algorithm
        for k in range(num_nodes):
            for i in range(num_nodes):
                for j in range(num_nodes):
                    distances[i][j] = min(distances[i][j], distances[i][k] + distances[k][j])

        return distances
    
    def update_field(self, pacman_positions, ghost_positions):
        """
        Updates the graph attributes based on Pac-Man and Ghosts' positions.
        """
        self.graph.x[:, :] = 0  # Clear all positions

        # TODO: Maybe we need to vectorize this
        for i, pos in enumerate(pacman_positions):
            self.graph.x[pos, i] = 1  # Update Pac-Man positions

        for i, pos in enumerate(ghost_positions):
            # Update Ghost positions
            self.graph.x[pos, self.ghosts_idx_start + i] = 1


class Agent:
    """
    Base class for agents in the game (e.g., Pac-Man or Ghosts).
    """

    def __init__(self, field: Field, is_pacman=True):
        self.field = field
        self.graph = field.graph
        self.action_vec = None
        self.is_pacman = is_pacman
        self.num_agents = self.field.num_pacman if is_pacman else self.field.num_ghosts
        self.idx_start = self.field.pacman_idx_start if is_pacman else self.field.ghosts_idx_start


    def get_action_set(self, data: Data = None):
        """
        Retrieves all possible actions for the agent(s), which is the neighbors of the current node(s).
        """
        if data is None:
            data = self.graph
        edge_index = data.edge_index
        x = data.x
        neighbors = []

        for i in range(self.num_agents):
            pos = (x[:, self.idx_start + i] == 1).nonzero().item()
            mask = edge_index[0] == pos
            neighbors.append(edge_index[1, mask])

        actions = torch.cartesian_prod(*neighbors)
        return actions if self.num_agents > 1 else actions.unsqueeze(dim=-1)

    def get_pos(self, agent_idx):
        return (self.graph.x[:, self.idx_start + agent_idx] == 1).nonzero().item()

    def set_action(self, action_vec):
        """
        Sets the action vector for the agent(s), which will be used to update the game state.
        """
        self.action_vec = action_vec

class RewardNormalizer:
    def __init__(self, alpha=0.1):
        self.mean = 0
        self.var = 1
        self.alpha = alpha  # Smoothing factor for running mean and variance

    def normalize(self, reward):
        # Update running mean and variance
        self.mean = self.alpha * reward + (1 - self.alpha) * self.mean
        self.var = self.alpha * (reward - self.mean) ** 2 + (1 - self.alpha) * self.var

        # Normalize reward
        return (reward - self.mean) / (self.var ** 0.5 + 1e-8)

class Environment:
    """
    Represents the Pac-Man game environment and manages game interactions.
    """

    def __init__(self):
        # Load settings from config
        self.num_pacman = config["num_pacman"]
        self.num_ghosts = config["num_ghosts"]

        # Initialize the field (game state graph)
        self.field = Field()

        # Initialize agents
        self.pacman = Agent(self.field, is_pacman=True)
        self.ghosts = Agent(self.field, is_pacman=False)

        self.reward_normalizer = RewardNormalizer()

        # Game state variables
        self.reset()

    # def reset(self):
    #     """
    #     Resets the environment to its initial state.
    #     """
    #     self.game_tick = 0
    #     self.game_over = False
    #     self.game_won = False
    #     self.score = 0

    #     # Reset agent positions
    #     self.pacman.set_action(
    #         self.pacman.field.pacman_spawn_pos[:self.num_pacman])
    #     self.ghosts.set_action(
    #         self.ghosts.field.ghosts_spawn_pos[:self.num_ghosts])

    #     # Update the field to reflect initial positions
    #     self.field.update_field(self.pacman.action_vec, self.ghosts.action_vec)

    #     return self.get_state()
    def reset(self):
        """
        Resets the environment to its initial state with all agents in different nodes on the grid.
        """
        self.game_tick = 0
        self.game_over = False
        self.game_won = False
        self.score = 0

        # Total number of nodes in the graph
        num_nodes = self.field.graph.x.shape[0]
        used_positions = set()

        # Randomly assign positions for Pac-Man agents
        pacman_positions = []
        for _ in range(self.num_pacman):
            pos = random.randint(0, num_nodes - 1)
            while pos in used_positions:
                pos = random.randint(0, num_nodes - 1)
            pacman_positions.append(pos)
            used_positions.add(pos)

        # Randomly assign positions for Ghost agents
        ghost_positions = []
        for _ in range(self.num_ghosts):
            pos = random.randint(0, num_nodes - 1)
            while pos in used_positions:
                pos = random.randint(0, num_nodes - 1)
            ghost_positions.append(pos)
            used_positions.add(pos)

        # Update agent positions
        self.pacman.set_action(torch.tensor(pacman_positions))
        self.ghosts.set_action(torch.tensor(ghost_positions))

        # Update the field to reflect initial positions
        self.field.update_field(self.pacman.action_vec, self.ghosts.action_vec)

        return self.get_state()
    
    def get_state(self):
        """
        Returns the current state of the environment as a graph.
        """
        return self.field.graph

    def restore_state(self, state: Data):
        """
        Restores the environment to a specific graph state.
        """
        self.field.graph = copy.deepcopy(state)  # Restore the graph

    def get_pacman_action_set(self, data: Data = None):
        """
        Returns all valid actions for Pac-Man.
        """
        return self.pacman.get_action_set(data)

    def get_ghost_action_set(self, data: Data = None):
        """
        Returns all valid actions for Ghosts.
        """
        return self.ghosts.get_action_set(data)

    def calculate_distance(self, pos1, pos2):
        """
        Returns the precomputed shortest distance between two nodes.

        Args:
            pos1 (int): Position of the first node.
            pos2 (int): Position of the second node.

        Returns:
            int: Precomputed shortest distance between the two nodes.
        """
        return self.field.distance_matrix[pos1, pos2]

    def calculate_reward(self, previous_pacman_positions, previous_ghost_positions, current_pacman_positions, current_ghost_positions):
        """
        Calculates the reward from the ghosts' perspective.
        Each ghost is rewarded for moving closer to Pac-Man and penalized for moving away.

        Args:
            previous_pacman_positions (list): Previous positions of Pac-Man.
            previous_ghost_positions (list): Previous positions of Ghosts.
            current_pacman_positions (list): Current positions of Pac-Man.
            current_ghost_positions (list): Current positions of Ghosts.

        Returns:
            float: Calculated reward for the ghosts' current actions.
        """
        reward = 0

        # Penalty for each time step (to encourage faster gameplay)
        reward += config["time_step_score"]

        # Assuming one Pac-Man for simplicity
        previous_pacman_position = previous_pacman_positions[0]
        current_pacman_position = current_pacman_positions[0]

        # Loop through each ghost
        for ghost_idx, current_ghost_pos in enumerate(current_ghost_positions):
            # Distance before and after the move
            distance_before = self.calculate_distance(previous_ghost_positions[ghost_idx], previous_pacman_position)
            distance_after = self.calculate_distance(current_ghost_pos, current_pacman_position)

            # Reward or penalize the ghost based on movement
            if distance_after < distance_before:  # Ghost moved closer
                reward += config["distance_reward_scale"]
            elif distance_after > distance_before:  # Ghost moved farther
                reward -= config["distance_reward_scale"]

        # Collision: If any ghost catches Pac-Man
        if current_pacman_position in current_ghost_positions:  # Collision detected
            reward += config["win_score"]  # Big reward for capturing Pac-Man
            self.game_over = True

        # Normalize reward
        normalized_reward = self.reward_normalizer.normalize(reward)
        return normalized_reward



    def step(self, pacman_action_vec, ghost_action_vec):
        """
        Executes one step in the environment by updating agent positions.

        Args:
            pacman_action_vec (Tensor): Action vector for Pac-Man.
            ghost_action_vec (Tensor): Action vector for Ghosts.

        Returns:
            state (Data): Updated game state graph.
            reward (float): Reward for this step.
            done (bool): Whether the game is over.
            score (float): Current game score.
        """
        # Store previous positions
        previous_pacman_positions = self.pacman.action_vec.clone().tolist()
        previous_ghost_positions = self.ghosts.action_vec.clone().tolist()

        # Update agent positions
        self.pacman.set_action(pacman_action_vec)
        self.ghosts.set_action(ghost_action_vec)
        self.field.update_field(self.pacman.action_vec, self.ghosts.action_vec)

        # Calculate reward using previous and new positions
        reward = self.calculate_reward(
            previous_pacman_positions, previous_ghost_positions,
            self.pacman.action_vec.tolist(), self.ghosts.action_vec.tolist()
        )

        # Check game-over conditions
        if self.game_tick >= config["max_steps"] or self.score <= config["loss_score"]:
            self.game_over = True

        # Update score
        self.update_score(reward)

        return self.get_state(), reward, self.game_over or self.game_won, self.score


    def update_score(self, delta):
        """
        Updates the game score.

        Args:
            delta (int): Change in score.
        """
        self.score += delta

    def dump(self):
        """
        Prints the current game state for debugging purposes.
        """
        print(f"Game Tick: {self.game_tick}")
        print(f"Score: {self.score}")
        print(f"Game Over: {self.game_over}")
        print(f"Game Won: {self.game_won}")

        for i in range(self.num_pacman):
            print(f"Pacman #{i} Position: {self.pacman.get_pos(i)}")

        for i in range(self.num_ghosts):
            print(f"Ghost #{i} Position: {self.ghosts.get_pos(i)}")


    def create_animation(self, graph_states):
        """
        Creates an animation of the graph states.
        """
        pos = {
            0: (0, 1), 1: (2, 1), 2: (4, 1), 3: (4, 3), 4: (6, 3),
            5: (8, 3), 6: (10, 3), 7: (12, 3), 8: (12, 1), 9: (14, 1),
            10: (16, 1), 11: (12, -1), 12: (12, -3), 13: (10, -3),
            14: (8, -3), 15: (6, -3), 16: (4, -3), 17: (4, -1), 18: (6, 1),
            19: (8, 1), 20: (10, 1), 21: (10, -1), 22: (8, -1), 23: (6, -1),
        }

        fig, ax = plt.subplots(figsize=(12, 8))

        def update(frame):
            ax.clear()
            state = graph_states[frame]

            # Create NetworkX graph
            graph = nx.Graph()
            edge_index = state["graph"].edge_index.cpu().numpy()
            for edge in edge_index.T:
                graph.add_edge(edge[0], edge[1])

            # Draw the graph with fixed positions
            nx.draw(graph, pos, ax=ax, with_labels=True, node_size=500, node_color="lightblue", font_weight="bold")

            # Highlight Pac-Man positions
            nx.draw_networkx_nodes(graph, pos, nodelist=state["pacman_positions"], node_color="yellow", ax=ax)

            # Highlight Ghost positions
            nx.draw_networkx_nodes(graph, pos, nodelist=state["ghost_positions"], node_color="red", ax=ax)

            # Add a title
            gameNumber = state["episode"]
            ax.set_title(f"Graph State: Frame {frame + 1}, Game # {gameNumber}")

        # Create the animation
        anim = FuncAnimation(fig, update, frames=len(graph_states), interval=500, repeat=False)
        return anim


def main():
    # Code to test the environment
    # env = Environment()
    # env.dump()
    # pacman_AS = env.get_pacman_action_set()
    # ghost_AS = env.get_ghost_action_set()
    # print(pacman_AS)
    # print(ghost_AS)
    # pacman_action_vec = pacman_AS[0]
    # ghost_action_vec = ghost_AS[1]
    # print(f'P ACT = {pacman_action_vec}')
    # print(f'G ACT = {ghost_action_vec}')
    # env.step(pacman_action_vec, ghost_action_vec)
    # env.dump()

    env = Environment()
    num_tests = 10
    for test in range(num_tests):
        # Reset the environment
        env.reset()

        # Get Pac-Man and Ghost positions
        pacman_positions = [env.pacman.get_pos(i) for i in range(env.num_pacman)]
        ghost_positions = [env.ghosts.get_pos(i) for i in range(env.num_ghosts)]

        print("positions: ", pacman_positions + ghost_positions)

        # Combine positions into one list
        all_positions = pacman_positions + ghost_positions

        # Check for duplicates
        if len(all_positions) != len(set(all_positions)):
            print(f"Test {test + 1}/{num_tests}: Overlap detected!")
            print(f"Pac-Man Positions: {pacman_positions}")
            print(f"Ghost Positions: {ghost_positions}")
        else:
            print(f"Test {test + 1}/{num_tests}: No overlap detected.")


if __name__ == "__main__":
    main()
