import torch
from torch_geometric.data import Data
from config import config


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

        # Initialize positions
        for i in range(self.num_pacman):
            x[self.pacman_spawn_pos[i], self.pacman_idx_start + i] = 1
        for i in range(self.num_ghosts):
            x[self.ghosts_spawn_pos[i], self.ghosts_idx_start + i] = 1

        self.graph = Data(
            x=x, edge_index=undirected_edge_index.t().contiguous())
        self.graph.validate(raise_on_error=True)

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

    def get_action_set(self):
        """
        Retrieves all possible actions for the agent(s), which is the neighbors of the current node(s).
        """
        edge_index = self.graph.edge_index
        x = self.graph.x
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

        # Game state variables
        self.reset()

    def reset(self):
        """
        Resets the environment to its initial state.
        """
        self.game_tick = 0
        self.game_over = False
        self.game_won = False
        self.score = 0

        # Reset agent positions
        self.pacman.set_action(
            self.pacman.field.pacman_spawn_pos[:self.num_pacman])
        self.ghosts.set_action(
            self.ghosts.field.ghosts_spawn_pos[:self.num_ghosts])

        # Update the field to reflect initial positions
        self.field.update_field(self.pacman.action_vec, self.ghosts.action_vec)

        return self.get_state()

    def get_state(self):
        """
        Returns the current state of the environment as a graph.
        """
        return self.field.graph

    def get_pacman_action_set(self):
        """
        Returns all valid actions for Pac-Man.
        """
        return self.pacman.get_action_set()

    def get_ghost_action_set(self):
        """
        Returns all valid actions for Ghosts.
        """
        return self.ghosts.get_action_set()

    def step(self, pacman_action_vec, ghost_action_vec):
        """
        Executes one step in the environment by updating agent positions.

        Args:
            pacman_action_vec (Tensor): Action vector for Pac-Man.
            ghost_action_vec (Tensor): Action vector for Ghosts.

        Returns:
            state (Data): Updated game state graph.
            reward (int): Reward for the step.
            done (bool): Whether the game is over.
            score (int): Current game score.
        """
        # Increment game tick
        self.game_tick += 1

        # Update agent positions
        self.pacman.set_action(pacman_action_vec)
        self.ghosts.set_action(ghost_action_vec)

        # Update the field with new positions
        self.field.update_field(self.pacman.action_vec, self.ghosts.action_vec)

        # Check for collisions (Pac-Man caught by any Ghost)
        # TODO: why can't these be pytorch tensor operations?
        pacman_positions = set(self.pacman.action_vec.tolist())
        ghost_positions = set(self.ghosts.action_vec.tolist())

        if pacman_positions & ghost_positions:
            self.game_won = True
            reward = config["reward"]
        else:
            reward = config["punishment"]

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
        print(self.field.graph)

        for i in range(self.num_pacman):
            print(f"Pacman #{i} Position: {self.pacman.get_pos(i)}")

        for i in range(self.num_ghosts):
            print(f"Ghost #{i} Position: {self.ghosts.get_pos(i)}")


def main():
    # Code to test the environment
    env = Environment()
    env.dump()
    pacman_AS = env.get_pacman_action_set()
    ghost_AS = env.get_ghost_action_set()
    print(pacman_AS)
    print(ghost_AS)
    pacman_action_vec = pacman_AS[0]
    ghost_action_vec = ghost_AS[1]
    print(f'P ACT = {pacman_action_vec}')
    print(f'G ACT = {ghost_action_vec}')
    env.step(pacman_action_vec, ghost_action_vec)
    env.dump()


if __name__ == "__main__":
    main()
