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
    def __init__(self, state_shape, action_size, model, learning_rate_max=0.001, learning_rate_decay=0.995, gamma=0.75,
                 memory_size=2000, batch_size=32, exploration_max=1.0, exploration_min=0.01, exploration_decay=0.995):
        # self.state_shape = state_shape
        # self.state_tensor_shape = (-1,) + state_shape
        # self.action_size = action_size
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

        # self.loss_fn = torch.nn.MSELoss()
        self.optim = torch.optim.Adam(
            model.parameters(), lr=self.learning_rate, weight_decay=1e-5)
        self.scheduler = optim.lr_scheduler.ExponentialLR(
            self.optim, gamma=self.learning_rate_decay)

    def update_target_model(self):
        self.target_model.load_state_dict(self.model.state_dict())

    def remember(self, state, action, reward, next_state, done):
        self.memory.push((copy.deepcopy(state), action, reward,
                         copy.deepcopy(next_state), done))

    def random_act(self, env: Environment):
        ghost_actions = env.get_ghost_action_set()
        random_action_idx = np.random.randint(len(ghost_actions))
        random_action = ghost_actions[random_action_idx]
        return random_action

    def act(self, env: Environment, epsilon=None):
        """
        Selects an ghost action based on the current policy or explores randomly.

        Args:
            data: torch_geometric data object that contains graph information
            epsilon (float, optional): Exploration rate. If None, uses self.exploration_rate.
        Returns:
            int: Selected action.
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
        # The following logic is technically in the agent class, but we don't
        # have access to that when calculating qvals. Thus, it's repeated here,
        # which is probably not the best design
        neighbors = []
        for i in range(config['num_ghosts']):  # TODO: vectorize this?
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
            # TODO: this is the dot product. Realistically, this all should
            # go in the model class, since we should be able to switch out
            # dot product aggregation with weighted dot product
            vals = torch.sum(cur_pos_emb * neighbor_embs,
                             dim=1)  # (num_neighbors,)
            q_vals += vals
        return q_vals

    # TODO: this should be replaced with some naive policy (e.g., move away from
    # ghosts)
    def pacman_act(self, env):
        """
        Retrieves all possible next actions and selects one of them at random.
        """
        pacman_action_set = env.get_pacman_action_set()
        pacman_action = pacman_action_set[torch.randint(
            len(pacman_action_set), (1,)).item()]
        return pacman_action

    def pacman_heuristic_action(self, env):
        # BUG
        state = env.get_state()
        edge_index = state.edge_index
        pacman_position = env.pacman.get_pos(0)
        ghost_positions = [env.ghosts.get_pos(
            i) for i in range(env.num_ghosts)]

        action_set = env.get_pacman_action_set()
        if len(action_set) == 0:  # Handle empty action set
            return torch.tensor([pacman_position])  # Stay in the same position

        best_action = None
        max_distance = -1

        for action in action_set:
            new_pos = action.item()
            distances = [
                torch.count_nonzero(
                    edge_index[1] == ghost_pos) - torch.count_nonzero(edge_index[0] == pacman_position)
                for ghost_pos in ghost_positions
            ]
            nearest_distance = min(distances)
            if nearest_distance > max_distance:
                max_distance = nearest_distance
                best_action = action

        return best_action

    def ghost_heuristic_action(self, env):
        state = env.get_state()
        edge_index = state.edge_index
        pacman_position = env.pacman.get_pos(0)
        ghost_positions = [env.ghosts.get_pos(
            i) for i in range(env.num_ghosts)]

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
        # 20 is tunable parameter? Hard to judge whats happening in the real code
        # Also, we can't really train on batches because the action set is not a
        # constant size. (e.g., imaging ghosts are on the corner nodes, so they
        # only have two neighbors)
        loss = None
        for i in range(1):
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
        if self.memory.length() < self.batch_size:
            # print(f"Replay skipped: Insufficient memory ({self.memory.length()}/{self.batch_size})")
            return None

        # print(f"Starting replay for episode {episode}")
        experiences, indices, weights = self.memory.sample(self.batch_size)
        # print(f"Sampled {len(experiences)} experiences from memory")

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
                # print(q_values) # brb

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
        # print(f"Learning rate after step: {self.scheduler.get_last_lr()}")

        return losses

    def load(self, name):
        # self.model = tf.keras.models.load_model(name)
        # self.target_model = tf.keras.models.load_model(name)
        # TODO: replace with pytorch load functions
        pass

    def save(self, name):
        self.model.save(name)
