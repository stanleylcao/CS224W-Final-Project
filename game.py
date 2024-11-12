import torch
from torch_geometric.data import Data


class Field:
    """
    This class is dedicated to representing the state of the game
    """

    def __init__(self, num_pacman=1, num_ghosts=2):
        """
        Initializes the board for pacman
        """
        self.num_pacman = num_pacman
        self.pacman_idx_start = 0
        # hardcoded position based on game mechanics
        self.pacman_spawn_pos = [5]

        self.num_ghosts = num_ghosts
        self.ghosts_idx_start = 1
        # hardcoded positions; extra positions if have more ghosts
        self.ghosts_spawn_pos = [18, 20, 23, 21, 22]

        edge_index = torch.tensor([[0, 1],
                                   [0, 10],
                                   [1, 2],
                                   [2, 3],
                                   [2, 17],
                                   [3, 4],
                                   [4, 5],
                                   [5, 6],
                                   [5, 19],
                                   [6, 7],
                                   [7, 8],
                                   [8, 9],
                                   [8, 11],
                                   [9, 10],
                                   [11, 12],
                                   [12, 13],
                                   [13, 14],
                                   [14, 15],
                                   [15, 16],
                                   [16, 17],
                                   [18, 19],
                                   [18, 23],
                                   [19, 20],
                                   [19, 22],
                                   [20, 21],
                                   [21, 22],
                                   [22, 23],], dtype=torch.long)

        # add other direction to represent undirected graph
        flipped_edges = edge_index.flip(dims=[1])
        undirected_edge_index = torch.cat([edge_index, flipped_edges], dim=0)

        x = torch.zeros(size=(torch.max(edge_index) + 1,
                        self.num_pacman + self.num_ghosts), dtype=torch.float)

        # Set pacman spawn on graph attributes
        for i in range(self.num_pacman):
            x[self.pacman_spawn_pos[i], self.pacman_idx_start + i] = 1

        # Set ghosts spawn on graph attributes
        for i in range(self.num_ghosts):
            x[self.ghosts_spawn_pos[i], self.ghosts_idx_start + i] = 1

        self.graph = Data(
            x=x, edge_index=undirected_edge_index.t().contiguous())
        self.graph.validate(raise_on_error=True)

    def update_field(self, pacman, ghosts):
        """
        Updates the graph attributes based on Pacman and Ghost objects. Fetches
        the move from those objects.
        """
        if pacman.action_vec is None or ghosts.action_vec is None:
            return
        # assert len(pacman.action_vec) == self.num_pacman
        assert len(ghosts.action_vec) == self.num_ghosts
        self.graph.x[:, :] = 0  # clear board
        self.graph.x[pacman.action_vec, 0] = 1  # set new pacman position
        # TODO: generalize
        self.graph.x[ghosts.action_vec, torch.arange(  # set new ghost positions
            self.num_ghosts) + self.ghosts_idx_start] = 1


class Pacman:
    """
    Class to encapsulate Pacman attributes
    """

    def __init__(self, field=None):
        self.field = field
        self.graph = field.graph
        self.action_vec = None

    def get_action_set(self):
        """
        Retrieves all possible actions for Pacman, which is just the neighbors
        of the node
        """
        edge_index = self.graph.edge_index
        x = self.graph.x
        pacman_neighbors = []
        for i in range(self.field.num_pacman):
            pos = (x[:, self.field.pacman_idx_start + i] == 1).nonzero().item()
            mask = (edge_index[0, :] == pos)
            neighbors = edge_index[1, mask]
            pacman_neighbors.append(neighbors)
        action_set = torch.cartesian_prod(*pacman_neighbors)
        return action_set

    def set_action(self, action_vec):
        """
        Sets the action, which will be read by the Field object to update the
        game state 
        """
        self.action_vec = action_vec


class Ghosts:
    """
    Class to encapsulate Ghost attributes
    """

    def __init__(self, field=None):
        self.field = field
        self.graph = field.graph
        self.action_vec = None

    def get_action_set(self):
        """
        Retrieves all possible actions for the Ghosts, which is the Cartesian
        product of all neighbors of nodes where a ghost is present at
        """
        edge_index = self.graph.edge_index
        x = self.graph.x
        ghost_neighbors = []
        for i in range(self.field.num_ghosts):
            pos = (x[:, self.field.ghosts_idx_start + i] == 1).nonzero().item()
            mask = (edge_index[0, :] == pos)
            neighbors = edge_index[1, mask]
            ghost_neighbors.append(neighbors)
        action_set = torch.cartesian_prod(*ghost_neighbors)
        return action_set

    def set_action(self, action_vec):
        """
        Sets the action, which will be read by the Field object to update the
        game state
        """
        self.action_vec = action_vec


class Environment:
    """
    Class that encapsulates the environment
    """
    # Hyperparameters
    ACTION_SPACE = [0, 1, 2]
    ACTION_SPACE_SIZE = len(ACTION_SPACE)
    ACTION_SHAPE = (ACTION_SPACE_SIZE,)
    PUNISHMENT = -1
    REWARD = 1
    score = 0
    MAX_VAL = 2

    LOSS_SCORE = -5
    WIN_SCORE = 5

    def __init__(self, num_pacman=1, num_ghosts=2):
        self.num_pacman = num_pacman
        self.num_ghosts = num_ghosts
        self.reset()

    def get_state(self):
        return self.field.graph

    def reset(self):
        self.game_tick = 0
        self.game_over = False
        self.game_won = False
        self.field = Field(num_pacman=self.num_pacman,
                           num_ghosts=self.num_ghosts)
        self.pacman = Pacman(self.field)
        self.ghosts = Ghosts(self.field)

        self.score = 0
        self.field.update_field(self.pacman, self.ghosts)

        return self.get_state()

    def get_pacman_action_set(self):
        return self.pacman.get_action_set()

    def get_ghost_action_set(self):
        return self.ghosts.get_action_set()

    def step(self, pacman_action_vec, ghost_action_vec):
        """
        This runs every step of the game. The DQN can pass an action vector from
        both pacman and the ghosts and in return, gets the next game state,
        reward, etc.
        """
        self.game_tick += 1

        self.pacman.set_action(pacman_action_vec)
        self.ghosts.set_action(ghost_action_vec)

        reward = 0

        self.field.update_field(self.pacman, self.ghosts)

        if self.score <= self.LOSS_SCORE:
            self.game_over = True

        if self.score >= self.WIN_SCORE:
            self.game_won = True

        return self.get_state(), reward, self.game_over or self.game_won, self.score

    def update_score(self, delta):
        self.score += delta

    def dump(self):
        """
        Debugging function to print out the state of the game.
        """
        print(self.field.graph)
        x = self.field.graph.x
        for i in range(self.num_pacman):
            print(
                f'Pacman #{i} POS = \t{(x[:, self.field.pacman_idx_start + i] == 1).nonzero().item()}')
        for i in range(self.num_ghosts):
            print(
                f'Ghost #{i} POS = \t{(x[:, self.field.ghosts_idx_start + i] == 1).nonzero().item()}')


def main():
    # Code to test the environment
    env = Environment()
    env.dump()
    pacman_AS = env.get_pacman_action_set()
    ghost_AS = env.get_ghost_action_set()
    pacman_action_vec = pacman_AS[0]
    ghost_action_vec = ghost_AS[1]
    print(f'P ACT = {pacman_action_vec}')
    print(f'G ACT = {ghost_action_vec}')
    env.step(pacman_action_vec, ghost_action_vec)
    env.dump()


if __name__ == "__main__":
    main()
