## Context

This is a reinforcement learning homework assignment implementing DQN and its variants. The base environment is a 4×4 grid world from the `DeepReinforcementLearningInAction` book repository. The agent must learn to navigate from a starting position to a goal while avoiding pits and walls. Three progressively harder modes are used: `static` (everything fixed), `player` (player starts randomly), and `random` (all objects randomized).

**Current state**: Empty project with only the base grid environment files cloned.

**Constraints**: Must use PyTorch as the base framework; HW3-3 converts to PyTorch Lightning.

## Goals / Non-Goals

**Goals:**
- Implement a functioning Naive DQN with Experience Replay on `static` mode
- Implement Double DQN and Dueling DQN on `player` mode and compare their performance
- Convert the best DQN model to PyTorch Lightning with training stabilization techniques for `random` mode
- Write a comprehensive README that serves as the HW3-1 understanding report
- Produce training plots comparing all methods

**Non-Goals:**
- Hyperparameter search / automated tuning
- Multi-agent environments
- Environments other than the grid world
- Deployment or productionization of agents

## Decisions

### Decision 1: Separate Python Scripts per Homework Part
**Choice**: Three separate `.py` files (`hw3_1_naive.py`, `hw3_2_variants.py`, `hw3_3_lightning.py`).
**Rationale**: Each homework part has a distinct mode and distinct algorithm. Separation keeps each file self-contained and easy to run and grade independently. Alternatives considered: single notebook (too large, hard to reuse code), single script with CLI flags (harder to understand).

### Decision 2: Shared `Gridworld.py` / `GridBoard.py` (no modification)
**Choice**: Use the original files from the DeepReinforcementLearningInAction repo unchanged.
**Rationale**: The assignment says to use the provided starter code as a baseline. Modifying the environment risks introducing bugs.

### Decision 3: Double DQN — separate online/target networks with online-for-action, target-for-value
**Choice**: Standard DDQN formulation: `a* = argmax_a Q_online(s', a)`, then `y = r + γ * Q_target(s', a*)`.
**Rationale**: This is the canonical Double DQN from Van Hasselt et al. 2015 and directly addresses overestimation bias.

### Decision 4: Dueling DQN — shared conv body, separate V and A streams, combined via `Q = V + (A - mean(A))`
**Choice**: Advantage normalization via mean subtraction (not max).
**Rationale**: Mean subtraction is more stable than max subtraction; standard implementation from Wang et al. 2015.

### Decision 5: PyTorch Lightning for HW3-3
**Choice**: Use `LightningModule` wrapping the DQN, with `Trainer` for gradient clipping and LR scheduling.
**Rationale**: Assignment explicitly requires conversion to Lightning (or Keras). Lightning provides clean separation of model logic and training loop; gradient clipping is a one-liner in the `Trainer`.

### Decision 6: Training Stabilization Techniques in HW3-3
**Chosen techniques**:
1. **Gradient clipping** (`gradient_clip_val=1.0`) — prevents exploding gradients
2. **Learning rate scheduling** (cosine annealing) — improves convergence
3. **Target network soft updates** (τ=0.005) — smoother target updates than hard copy every N steps

## Risks / Trade-offs

- **Grid world convergence**: The `random` mode environment is significantly harder; DQN may need more episodes to converge — Mitigation: tune `n_episodes` and `epsilon_decay`.
- **PyTorch Lightning version compatibility**: Lightning API has changed across versions — Mitigation: pin version in requirements, use `lightning>=2.0` API.
- **Experience Replay buffer size**: Too small → overfit on recent transitions; too large → slow learning — Mitigation: use buffer of 1000–5000 with batch size 64.
