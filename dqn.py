"""
dqn.py
Implements the Deep Q-Learning Network (DQN) to optimize ghost actions.
- Maintains a target model for stability in Q-learning and a Prioritized Replay Buffer for sampling and replaying important experiences.
- Predicts Q-values based on ghost positions and their neighbors in the graph.
- Selects actions using an epsilon-greedy policy: random action for exploration, highest Q-value for exploitation.
- Updates the model using the Temporal Difference (TD) error and trains with a loss function (MSELoss).
"""
import numpy as np
import copy
from replay_buffer import PrioritizedReplayBuffer
import torch
from torch_geometric.data import Data
from game import Environment
from config import config
import torch.optim as optim
import torch.nn.functional as F


class DQN:
    """
    Implements the Deep Q-Learning algorithm using a Graph Neural Network (GNN) model.
    """
    def __init__(self, state_shape, action_size, model, learning_rate_max=0.001, learning_rate_decay=0.995, gamma=0.75,
                 memory_size=2000, batch_size=32, exploration_max=1.0, exploration_min=0.01, exploration_decay=0.995):
        """
        Initializes the DQN class.

        Args:
            state_shape (tuple): Shape of the graph-based state representation.
            action_size (int): Number of possible actions.
            model (torch.nn.Module): Neural network model for Q-value predictions.
            learning_rate_max (float): Maximum learning rate.
            learning_rate_decay (float): Decay factor for learning rate.
            gamma (float): Discount factor for future rewards.
            memory_size (int): Maximum size of the prioritized replay buffer.
            batch_size (int): Number of samples for training in each replay step.
            exploration_max (float): Initial exploration rate for epsilon-greedy policy.
            exploration_min (float): Minimum exploration rate.
            exploration_decay (float): Decay factor for exploration rate.
        """
        self.learning_rate_max = learning_rate_max
        self.learning_rate = learning_rate_max
        self.learning_rate_decay = learning_rate_decay
        self.gamma = gamma
        self.memory_size = memory_size
        self.memory = PrioritizedReplayBuffer(capacity=2000)
        self.batch_size = batch_size
        self.exploration_rate = exploration_max
        self.exploration_max = exploration_max
        self.exploration_min = exploration_min
        self.exploration_decay = exploration_decay

        self.model = model
        # Clone the model for the target network
        self.target_model = copy.deepcopy(self.model)
        self.update_target_model()

        self.optim = torch.optim.Adam(
            model.parameters(), lr=self.learning_rate, weight_decay=1e-5)
        self.scheduler = optim.lr_scheduler.ExponentialLR(
            self.optim, gamma=self.learning_rate_decay)

    def update_target_model(self):
        """
        Copies the weights of the policy network into the target network for stable Q-value estimation.
        """
        self.target_model.load_state_dict(self.model.state_dict())

    def remember(self, state, action, reward, next_state, done):
        """
        Stores an experience tuple in the replay buffer.

        Args:
            state (Data): Current state of the environment as a graph.
            action (torch.Tensor): Action taken in the current state.
            reward (float): Reward received for the action.
            next_state (Data): Resulting state after the action.
            done (bool): Flag indicating whether the episode is over.
        """
        self.memory.push((copy.deepcopy(state), action, reward,
                         copy.deepcopy(next_state), done))

    def random_act(self, env: Environment):
        """
        Selects a random action from the available action set.

        Args:
            env (Environment): Game environment.

        Returns:
            torch.Tensor: Random action selected.
        """
        ghost_actions = env.get_ghost_action_set()
        random_action_idx = np.random.randint(len(ghost_actions))
        random_action = ghost_actions[random_action_idx]
        return random_action

    def act(self, env: Environment, epsilon=None):
        """
        Selects an action using epsilon-greedy exploration.

        Args:
            env (Environment): Game environment.
            epsilon (float, optional): Exploration rate. Uses self.exploration_rate if not provided.

        Returns:
            torch.Tensor: Selected action.
        """
        data = env.get_state()

        node_features = data.x
        edge_index = data.edge_index

        if epsilon is None:
            epsilon = self.exploration_rate  # Use current exploration rate if not provided

        # Exploration: Random action
        if np.random.rand() < epsilon:
            ghost_actions = env.get_ghost_action_set()
            random_action_idx = np.random.randint(len(ghost_actions))
            random_action = ghost_actions[random_action_idx]
            return random_action

        # Exploitation: Select action with highest Q-value
        with torch.no_grad():
            # Forward pass through the model
            self.model.eval()  # set model to eval mode
            node_embs = self.model(node_features, edge_index)  # (V, out_d)
            # print("node_embs: ", node_embs)
            ghost_actions = env.get_ghost_action_set()

            q_vals = self.get_qvals(data, node_embs)
            # Return the action with the highest Q-value
            best_act_idx = torch.argmax(q_vals).item()
            return ghost_actions[best_act_idx]

    def get_qvals(self, state: Data, node_embs):
        """
        Computes Q-values for all possible actions.

        Args:
            state (Data): Current state of the environment.
            node_embs (torch.Tensor): Node embeddings from the model.

        Returns:
            torch.Tensor: Q-values for all possible actions.
        """
        neighbors = []
        for i in range(config['num_ghosts']):  # Extract neighbors for each ghost
            pos = (state.x[:, config['ghosts_idx_start'] + i]
                   == 1).nonzero().item()
            mask = state.edge_index[0] == pos
            neighbors.append(state.edge_index[1, mask])
        ghost_actions = torch.cartesian_prod(*neighbors)

        q_vals = 0
        for i in range(config['num_ghosts']):
            cur_pos = (state.x[:, config['ghosts_idx_start'] + i]
                       == 1).nonzero().item()
            cur_pos_emb = node_embs[cur_pos]  # (out_d,)
            possible_next_pos = ghost_actions[:, i]
            # (num_neighbors, out_d)
            neighbor_embs = node_embs[possible_next_pos]

            # Calculate q_vals
            vals = torch.sum(cur_pos_emb * neighbor_embs,
                             dim=1)  # (num_neighbors,)
            q_vals += vals
        return q_vals

    def pacman_act(self, env):
        """
        Selects a random valid action for Pac-Man.

        Args:
            env (Environment): Game environment.

        Returns:
            torch.Tensor: Action vector for Pac-Man.
        """
        pacman_action_set = env.get_pacman_action_set()
        pacman_action = pacman_action_set[torch.randint(
            len(pacman_action_set), (1,)).item()]
        return pacman_action

    def ghost_heuristic_action(self, env):
        """
        Selects actions for ghosts using a heuristic policy based on minimizing distance to Pac-Man.

        Args:
            env (Environment): Game environment.

        Returns:
            torch.Tensor: Action vectors for all ghosts based on the heuristic policy.
        """
        state = env.get_state()
        edge_index = state.edge_index
        pacman_position = env.pacman.get_pos(0)
        ghost_positions = [env.ghosts.get_pos(i) for i in range(env.num_ghosts)]

        actions = []
        for ghost_idx, ghost_pos in enumerate(ghost_positions):
            action_set = env.get_ghost_action_set()
            if len(action_set) == 0:  # Handle empty action set
                actions.append(ghost_pos)  # Stay in the same position
                continue

            best_action = None
            min_distance = float('inf')

            for action in action_set[:, ghost_idx]:
                new_pos = action.item()
                distance = torch.count_nonzero(edge_index[1] == new_pos) - torch.count_nonzero(
                    edge_index[0] == pacman_position
                )
                if distance < min_distance:
                    min_distance = distance
                    best_action = new_pos

            actions.append(best_action)

        return torch.tensor(actions)

    def train_model(self, env, state: Data, q_values):
        """
        Trains the model using a single batch of data.

        Args:
            env (Environment): Game environment.
            state (Data): Current state of the environment as a graph.
            q_values (torch.Tensor): Q-values to use as targets.

        Returns:
            float: Loss from the training step.
        """
        loss = None
        for i in range(20):
            self.optim.zero_grad()
            model_embs = self.model(state.x, state.edge_index)
            pred_q_vals = self.get_qvals(state, model_embs)
            # Huber loss
            loss = F.smooth_l1_loss(pred_q_vals, q_values)
            loss.backward()
            # Clip gradients to avoid exploding gradients
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), max_norm=0.5)
            self.optim.step()

        return loss.item()  # should return last loss

    def replay(self, env: Environment, episode=0):
        """
        Samples a batch of experiences from the replay buffer and trains the model.

        Args:
            env (Environment): Game environment.
            episode (int): Current training episode for logging purposes.

        Returns:
            list: Losses for each training step in the replay.
        """
        if self.memory.length() < self.batch_size:
            return None

        experiences, indices, weights = self.memory.sample(self.batch_size)

        losses = []
        for i, (state, action, reward, next_state, done) in enumerate(experiences):
            with torch.no_grad():
                if done:
                    target = reward
                else:
                    # Use the main network to select the best action
                    main_model_embs = self.model(
                        next_state.x, next_state.edge_index)
                    next_q_values = self.get_qvals(next_state, main_model_embs)
                    best_action = torch.argmax(next_q_values).item()

                    # Use the target network to evaluate the value of the best action
                    target_model_embs = self.target_model(
                        next_state.x, next_state.edge_index)
                    target_q_values = self.get_qvals(
                        next_state, target_model_embs)
                    target = reward + self.gamma * target_q_values[best_action]

                # Compute Q-values for the current state
                model_embs = self.model(state.x, state.edge_index)
                q_values = self.get_qvals(state, model_embs)

                # Map action to a valid index
                ghost_actions = env.get_ghost_action_set(data=state)
                matching_rows = (ghost_actions == action).all(
                    dim=1).nonzero(as_tuple=True)
                if matching_rows[0].numel() == 0:
                    continue

                action_idx = matching_rows[0].item()
                q_values_current_action = q_values[action_idx]

                # Compute TD error and loss
                td_error = target - q_values_current_action
                self.memory.update_priorities(
                    [indices[i]], [np.abs(td_error.item())])
                q_values[action_idx] = target

            # Train the model using Huber loss
            loss = self.train_model(env, state, q_values)
            losses.append(loss)

        # Step the scheduler to decay the learning rate
        self.scheduler.step()

        return losses