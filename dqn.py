import numpy as np
import copy
import random
from replay_buffer import PrioritizedReplayBuffer
import torch

from torch_geometric.data import Data

from game import Environment


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

    def update_target_model(self):
        self.target_model.load_state_dict(self.model.state_dict())

    def remember(self, state, action, reward, next_state, done):
        self.memory.push((state, action, reward, next_state, done))

    def act(self, env: Environment, epsilon=None):
        """
        Selects an action based on the current policy or explores randomly.

        Args:
            data: torch_geometric data object that contains graph information
            epsilon (float, optional): Exploration rate. If None, uses self.exploration_rate.
        Returns:
            int: Selected action.
        """
        data = env.get_state()

        node_features = data.x
        edge_index = data.edge_index

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

            q_vals = 0
            for i in range(env.num_ghosts):  # TODO: vectorize this?
                cur_pos = env.ghosts.get_pos(i)
                cur_pos_emb = node_embs[cur_pos,]  # (out_d,)
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
            # Return the action with the highest Q-value
            return torch.argmax(q_vals).item()

    def replay(self, episode=0):

        if self.memory.length() < self.batch_size:
            return None

        experiences, indices, weights = self.memory.sample(self.batch_size)
        unpacked_experiences = list(zip(*experiences))
        states, actions, rewards, next_states, dones = [
            list(arr) for arr in unpacked_experiences]

        # Convert to tensors
        states = tf.convert_to_tensor(states)
        states = tf.reshape(states, self.state_tensor_shape)
        actions = tf.convert_to_tensor(actions, dtype=tf.int32)
        rewards = tf.convert_to_tensor(rewards, dtype=tf.float32)
        next_states = tf.convert_to_tensor(next_states)
        next_states = tf.reshape(next_states, self.state_tensor_shape)
        dones = tf.convert_to_tensor(dones, dtype=tf.float32)

        # Compute Q values and next Q values
        target_q_values = self.target_model.predict(next_states, verbose=0)
        q_values = self.model.predict(states, verbose=0)

        # Compute target values using the Bellman equation
        max_target_q_values = np.max(target_q_values, axis=1)
        targets = rewards + (1 - dones) * self.gamma * max_target_q_values

        # Compute TD errors
        batch_indices = np.arange(self.batch_size)
        q_values_current_action = q_values[batch_indices, actions]
        td_errors = targets - q_values_current_action
        self.memory.update_priorities(indices, np.abs(td_errors))

        # For learning: Adjust Q values of taken actions to match the computed targets
        q_values[batch_indices, actions] = targets

        loss = self.model.train_on_batch(
            states, q_values, sample_weight=weights)

        self.exploration_rate = self.exploration_max*self.exploration_decay**episode
        self.exploration_rate = max(
            self.exploration_min, self.exploration_rate)
        self.learning_rate = self.learning_rate_max*self.learning_rate_decay**episode
        tf.keras.backend.set_value(
            self.model.optimizer.learning_rate, self.learning_rate)

        return loss

    def load(self, name):
        self.model = tf.keras.models.load_model(name)
        self.target_model = tf.keras.models.load_model(name)

    def save(self, name):
        self.model.save(name)
