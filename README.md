# Project Progress and Next Steps
---

## 1. Logic Flow in `main.py`

We wrote a `main.py` with the overall logic and how all the pieces fit together:
- Initializing the environment (`game.py`).
- Loading the appropriate GNN model (GCN, GAT, or GraphSAGE).
- Setting up the DQN agent (`dqn.py`).
- A training loop to interact with the environment and train the DQN.
- Placeholder logic for saving and evaluating the model.

This should be the entry point for the project and our understanding of how the environment, GNN models, and DQN interact.

---

## 2. Refactoring with `config.py`

We created a `config.py` file to centralize all the constants and hyperparameters, so any changes can be made in one place. We were thinking this would make the code easier to manage lol. (TODO: `dqn.py` still needs to be updated to use `config.py` for hyperparameters)

---

## 3. Requirements File

We added a `requirements.txt` file for the virtual environments. 

---

## 4. Implemented GNN Models

We implemented the 3 GNN models in the `models/` folder from the project proposal:

1. GCN
2. GraphSage
3. GAT

Each model is implemented as a PyTorch class with a consistent input-output interface so it's easier to integrate and switch between models in `main.py`.

---

## 5. Updates to the DQN Class

We made the following updates to the `DQN` class to start integrating it with the main.py file and to integrate with the GNN models (this isn't completed yet):

1. **GNN Integration:**
   - Modified the `DQN` class to take in the pre-initialized GNN model as an argument (`model`) from main.py

2. **Action Selection:**
   - Updated the `act()` method to use the GNN model for predictions:
     - The `act()` function now takes the node features and edge indices as input, passes them through the GNN model, and selects an action based on the output.

4. **Prioritized Replay Buffer:**
   - Copied the `PrioritizedReplayBuffer` class code from the Medium article into a new `replay_buffer.py` file.

---


## Next Steps

### 1. Refactor `dqn.py`:
- Replace all hardcoded hyperparameters with values from `config.py`.
- Update the DQN class code so it integrates with the GNN models and works within the current logic flow of `main.py`.

### 2. Debug and Test:
- Run `main.py` and verify that:
  - The environment initializes correctly.
  - The training loop runs without errors.
  - The DQN starts to learn meaningful policies (observe training metrics like loss or rewards).

### 3. Add Evaluation Logic:
- Implement an evaluation loop in `main.py` to test the trained model:
  - Log evaluation metrics for comparison between different GNN models.
- **Visualization:**
  - Add tools to visualize the graph, node features, or agent actions during gameplay.
- **Logging:**
  - Set up a logging system to track training progress, loss, rewards, and evaluation metrics.
- **(Extension) Additional Models:**
  - Experiment with other GNN architectures or hyperparameter tuning for better performance.



