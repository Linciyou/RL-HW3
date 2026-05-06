## Why

This homework implements Deep Q-Network (DQN) and its variants as part of a reinforcement learning course. The goal is to build a working DQN agent that learns to navigate a grid-world environment, then progressively improve it with state-of-the-art techniques (Double DQN, Dueling DQN) and modern training frameworks (PyTorch Lightning). This work is required for the course HW3, worth 100% of the assignment grade.

## What Changes

- **New**: `hw3_1_naive.py` — Naive DQN agent with Experience Replay Buffer for `static` mode grid world
- **New**: `hw3_2_variants.py` — Double DQN and Dueling DQN agents for `player` mode grid world
- **New**: `hw3_3_lightning.py` — PyTorch Lightning DQN with advanced training techniques for `random` mode
- **New**: `Gridworld.py` / `GridBoard.py` — Grid world environment (sourced from DeepReinforcementLearningInAction)
- **New**: `README.md` — Project documentation and understanding report (HW3-1 requirement)
- **New**: Training result plots and comparison charts

## Capabilities

### New Capabilities

- `naive-dqn`: Basic DQN with Experience Replay Buffer running on `static` grid world mode. Includes a neural network Q-function approximator trained via the Bellman equation.
- `double-dqn`: Double DQN variant that decouples action selection from Q-value estimation to reduce overestimation bias. Runs on `player` mode.
- `dueling-dqn`: Dueling DQN architecture that separates state-value (V) and advantage (A) streams for better generalization. Runs on `player` mode.
- `lightning-dqn`: Refactored DQN using PyTorch Lightning with gradient clipping, learning rate scheduling, and other training stabilization techniques. Runs on `random` mode.

### Modified Capabilities

<!-- None — this is a new project with no existing specs. -->

## Impact

- **Dependencies**: PyTorch, PyTorch Lightning, NumPy, Matplotlib
- **Code**: All new files; uses `Gridworld.py` and `GridBoard.py` from the DeepReinforcementLearningInAction repository
- **Environment**: Requires Python 3.8+, CUDA optional but recommended
