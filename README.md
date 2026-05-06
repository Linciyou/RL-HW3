# Deep Reinforcement Learning - HW3

This repository contains the implementation of Deep Q-Networks (DQN) and its variants for navigating a Gridworld environment, based on the `DeepReinforcementLearningInAction` codebase.

## HW3-1: Naive DQN and Experience Replay

### Understanding Report
**1. Deep Q-Network (DQN) Overview**
A DQN is a reinforcement learning agent that uses a Neural Network to approximate the optimal Q-value function $Q^*(s, a)$. Instead of maintaining a large table of Q-values for every state-action pair, the neural network takes the state as input and outputs the estimated Q-values for all possible actions. The network is trained by minimizing the Mean Squared Error (MSE) between the predicted Q-values and the target Q-values computed using the Bellman equation.

**2. The Bellman Equation**
The Bellman equation defines the relationship between the value of a state and the values of its successor states. In Q-learning, the target Q-value is given by:
$Y_t = R_{t+1} + \gamma \max_a Q(S_{t+1}, a)$
The agent uses the current policy to take an action, observes the reward $R_{t+1}$ and next state $S_{t+1}$, and then updates its network to bring its prediction closer to $Y_t$.

**3. Experience Replay Buffer**
In traditional Q-learning, updates are performed sequentially as the agent interacts with the environment. This leads to highly correlated training samples and non-stationary targets, which destabilizes neural network training.
An Experience Replay Buffer addresses this by:
- Storing past transitions $(s_t, a_t, r_{t+1}, s_{t+1}, d)$ in a fixed-size queue.
- Sampling random mini-batches from the buffer during training.
This breaks the temporal correlation between consecutive samples and allows the network to learn more robust features.

**Implementation (`hw3_1_naive.py`)**: 
The agent successfully learns to navigate the `static` mode Gridworld. Training loss spikes initially as the agent explores, but the reward plot shows convergence to a stable positive average reward as it masters the fixed start and goal positions.

---

## HW3-2: Double DQN and Dueling DQN

### Variant Comparison
**1. Double DQN (DDQN)**
Basic DQN suffers from *overestimation bias* because it uses the same network to both select the best action and estimate its value: $\max_a Q(s', a)$.
Double DQN mitigates this by decoupling action selection from value estimation. It uses an **online network** to select the action, and a **target network** to evaluate it:
$Y_t = R_{t+1} + \gamma Q_{target}(s', \arg\max_a Q_{online}(s', a))$
*Result*: DDQN prevents inflated Q-values, leading to more stable policies, especially in stochastic environments like the `player` mode where the start position is randomized.

**2. Dueling DQN**
Dueling DQN changes the network *architecture* rather than the loss function. It splits the network into two streams:
- **Value stream $V(s)$**: Estimates how good the state is, regardless of the action taken.
- **Advantage stream $A(s, a)$**: Estimates the relative advantage of each action in that state.
They are combined at the final layer: $Q(s,a) = V(s) + (A(s,a) - \text{mean}(A(s,a)))$.
*Result*: This architecture learns state values much faster, because updating $V(s)$ updates the baseline for all actions simultaneously. It performs exceptionally well in states where the choice of action doesn't significantly affect the outcome.

**Comparison in `player` Mode**: Both variants outperform the Naive DQN in generalizing to random starting positions. Dueling DQN typically shows faster initial learning because of its efficient state-value updates, while DDQN achieves a very stable asymptotic performance by avoiding overestimation.

---

## HW3-3: PyTorch Lightning DQN

**Implementation (`hw3_3_lightning.py`)**:
The model has been refactored using `LightningModule` to handle the `random` mode (where all objects are randomly placed). 
To stabilize training in this highly dynamic environment, the following techniques were integrated:
- **Gradient Clipping**: `clip_grad_norm_` (max_norm=1.0) prevents exploding gradients during sudden large Q-value updates.
- **Learning Rate Scheduling**: `CosineAnnealingLR` gradually reduces the learning rate to ensure fine-tuning near convergence.
- **Soft Target Updates**: The target network is updated continuously using Polyak averaging ($\tau = 0.005$) instead of hard-copying every $N$ steps, smoothing the learning trajectory.

---

## HW3-4 (Bonus): Rainbow DQN — Random Mode GridWorld

**Implementation (`hw3_4_rainbow.py`)**

### Why Rainbow for Random Mode?

The `random` mode places **all four objects** (Player, Goal, Pit, Wall) at completely random positions every episode.
This means the agent can never memorize a fixed path — it must learn **abstract spatial relationships** (e.g., "move towards the `+`, away from the `-`") from pure pixels.
Standard DQN and even Double/Dueling variants struggle to converge reliably in this setting.
Rainbow DQN combines multiple orthogonal improvements to address each failure mode.

### The 5 Integrated Components

| # | Component | Problem Solved | Key Idea |
|---|---|---|---|
| 1 | **Double DQN** | Overestimation bias | Online net selects action; target net evaluates it |
| 2 | **Dueling Network** | Slow value learning | Separate $V(s)$ and $A(s,a)$ streams; $Q = V + (A - \bar{A})$ |
| 3 | **Prioritized Replay (PER)** | Wasted sample efficiency | Sample transitions with higher TD-error more often; correct bias with IS weights |
| 4 | **Multi-step Returns (n=3)** | Slow credit assignment | Bootstrap from $R_t^{(n)} = r_t + \gamma r_{t+1} + \gamma^2 r_{t+2}$; richer gradient signal |
| 5 | **Noisy Nets** | ε-greedy explores uniformly | Learnable noise $\sigma$ per weight; network decides **when** to explore |

> **Note on Distributional RL (C51):** Omitted intentionally. C51 requires predicting a full return distribution over a discrete support, adding categorical cross-entropy loss and a projection step. On a 4×4 GridWorld with only 3 distinct reward values (−10, −1, +10) the distributional benefit is minimal, while code complexity increases significantly.

### Architecture

```
Input (64-dim) → Linear(128) → ReLU
                                    ├─ NoisyLinear(128) → ReLU → NoisyLinear(1)   = V(s)
                                    └─ NoisyLinear(128) → ReLU → NoisyLinear(4)   = A(s,a)
                             Q(s,a) = V(s) + A(s,a) − mean_a'[A(s,a')]
```

### Training Hyperparameters

| Parameter | Value | Rationale |
|---|---|---|
| Episodes | 3000 | Longer needed for fully random env |
| γ (discount) | 0.99 | Encourage long-horizon planning |
| n-step | 3 | Balance bias vs. variance |
| PER α | 0.6 | Partial prioritization |
| PER β annealing | 0.4 → 1.0 | Correct IS bias progressively |
| Target update | every 200 steps | Hard copy for stability |
| σ_init (NoisyNet) | 0.5 | Initial exploration level |
| Grad clip | 10.0 | Prevent exploding gradients |
