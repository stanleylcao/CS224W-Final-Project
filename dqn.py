import numpy as np
import copy
import random
from pprint import pprint
from replay_buffer import PrioritizedReplayBuffer
import torch

from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

from game import Environment

from config import config


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

        self.loss_fn = torch.nn.MSELoss()
        self.optim = torch.optim.Adam(
            model.parameters(), lr=self.learning_rate)

    def update_target_model(self):
        self.target_model.load_state_dict(self.model.state_dict())

    def remember(self, state, action, reward, next_state, done):
            self.memory.push((copy.deepcopy(state), action, reward, copy.deepcopy(next_state), done))

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

        # Commented this out for now for debugging
        # if epsilon is None:
        #     epsilon = self.exploration_rate  # Default to exploration rate

        # Ensure epsilon is a scalar (convert tensor to float if necessary)
        # if isinstance(epsilon, torch.Tensor):
        #     epsilon = epsilon.item()

        # Exploration: Random action
        # if np.random.rand() < epsilon:
        #     return random.randrange(self.action_size)

        # Exploitation: Select action with highest Q-value
        with torch.no_grad():
            # Forward pass through the model
            self.model.eval()  # set model to eval mode
            node_embs = self.model(node_features, edge_index)  # (V, out_d)
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
        pacman_action_set = env.get_pacman_action_set()
        pacman_action = pacman_action_set[torch.randint(
            len(pacman_action_set), (1,)).item()]
        return pacman_action

    def train_model(self, env, state: Data, q_values):
        # 20 is tunable parameter? Hard to judge whats happening in the real code
        # Also, we can't really train on batches because the action set is not a
        # constant size. (e.g., imaging ghosts are on the corner nodes, so they
        # only have two neighbors)
        loss = None
        for i in range(20):
            self.optim.zero_grad()
            model_embs = self.model(state.x, state.edge_index)
            pred_q_vals = self.get_qvals(state, model_embs)
            loss = self.loss_fn(pred_q_vals, q_values)
            loss.backward()
            self.optim.step()
        return loss.item()  # should return last loss

    def replay(self, env: Environment, episode=0):
        if self.memory.length() < self.batch_size:
            return None
        experiences, indices, weights = self.memory.sample(self.batch_size)
        # unpacked_experiences = list(zip(*experiences))

        losses = []
        for i, (state, action, reward, next_state, done) in enumerate(experiences):
            with torch.no_grad():
                # Compute Q-values for the next state
                target_model_embs = self.target_model(next_state.x, next_state.edge_index)
                target_q_values = self.get_qvals(next_state, target_model_embs)
                max_target_q_values = torch.max(target_q_values)
                target = reward + (1 - done) * self.gamma * max_target_q_values

                # Compute Q-values for the current state
                model_embs = self.model(state.x, state.edge_index)
                q_values = self.get_qvals(state, model_embs)

                # Map action to a valid index
                ghost_actions = env.get_ghost_action_set(data = state)

                #print(f"Action: {action.tolist()}, Type: {type(action)}, Shape: {action.shape}")
                #print(f"Ghost Actions: {ghost_actions.tolist()}, Type: {type(ghost_actions)}, Shape: {ghost_actions.shape}")

                # TODO: this is a temporary fix -- need to figure out why some games initialize on a terminal state
                matching_rows = (ghost_actions == action).all(dim=1).nonzero(as_tuple=True)
                if matching_rows[0].numel() == 0:
                    # print(f"Skipping invalid action {action.tolist()} not found in {ghost_actions.tolist()}")
                    continue

                action_idx = (ghost_actions == action).all(dim=1).nonzero(as_tuple=True)[0].item()
                # print(f"Action Index: {action_idx}")  # Debugging

                q_values_current_action = q_values[action_idx]

                # Compute TD error
                td_error = target - q_values_current_action
                self.memory.update_priorities([indices[i]], [np.abs(td_error.item())])

                # Update Q-values for learning
                q_values[action_idx] = target

            loss = self.train_model(env, state, q_values)
            losses.append(loss)

        # states = tf.convert_to_tensor(states)
        # states = tf.reshape(states, self.state_tensor_shape)
        # actions = tf.convert_to_tensor(actions, dtype=tf.int32)
        # rewards = tf.convert_to_tensor(rewards, dtype=tf.float32)
        # next_states = tf.convert_to_tensor(next_states)
        # next_states = tf.reshape(next_states, self.state_tensor_shape)
        # dones = tf.convert_to_tensor(dones, dtype=tf.float32)

        # target_q_values = self.target_model.predict(next_states, verbose=0)
        # q_values = self.model.predict(states, verbose=0)

        # # Compute target values using the Bellman equation
        # max_target_q_values = np.max(target_q_values, axis=1)
        # targets = rewards + (1 - dones) * self.gamma * max_target_q_values

        # # Compute TD errors
        # batch_indices = np.arange(self.batch_size)
        # q_values_current_action = q_values[batch_indices, actions]
        # td_errors = targets - q_values_current_action
        # self.memory.update_priorities(indices, np.abs(td_errors))

        # For learning: Adjust Q values of taken actions to match the computed targets
        # q_values[batch_indices, actions] = targets

        # loss = self.model.train_on_batch(
        #     states, q_values, sample_weight=weights)

        self.exploration_rate = self.exploration_max*self.exploration_decay**episode
        self.exploration_rate = max(
            self.exploration_min, self.exploration_rate)
        self.learning_rate = self.learning_rate_max * self.learning_rate_decay ** episode
        # Decay learning rate
        for param_group in self.optim.param_groups:
            param_group['lr'] = self.learning_rate

        return losses

    def load(self, name):
        # self.model = tf.keras.models.load_model(name)
        # self.target_model = tf.keras.models.load_model(name)
        # TODO: replace with pytorch load functions
        pass

    def save(self, name):
        self.model.save(name)
