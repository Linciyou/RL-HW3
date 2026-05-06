"""
HW3-4 (Bonus): Rainbow DQN for Random Mode GridWorld
=====================================================
Rainbow DQN integrates the following improvements over vanilla DQN:
  1. Double DQN          - Decouples action selection & evaluation to reduce overestimation.
  2. Dueling Network     - Separates V(s) and A(s,a) streams for better state-value learning.
  3. Prioritized Exp.    - Samples important transitions more frequently (PER).
     Replay (PER)
  4. Multi-step Returns  - Uses n-step TD targets for richer gradient signal.
  5. Noisy Nets          - Adds parametric noise to linear layers for implicit exploration,
                           replacing ε-greedy entirely.

Note: Distributional RL (C51) is omitted intentionally — it adds significant complexity
      (categorical projection, KL-divergence loss) with diminishing benefit on a 4×4 grid.
      The 5 components above already form a very strong "practical Rainbow" agent.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from collections import deque, namedtuple
import random
import math
import copy
from Gridworld import Gridworld

# ─────────────────────────────────────────────
# Component 1: Noisy Linear Layer
# Replaces standard nn.Linear in the network.
# Each weight/bias has a learnable noise scale (σ) and a fixed random noise (ε).
# The network learns WHEN to explore by adjusting σ — no ε-greedy needed.
# ─────────────────────────────────────────────
class NoisyLinear(nn.Module):
    """
    Factorised Noisy Linear layer (Fortunato et al., 2017).
    
    w = μ_w + σ_w ⊙ ε_w,   b = μ_b + σ_b ⊙ ε_b
    where ε is sampled from a factorised normal distribution.
    """
    def __init__(self, in_features: int, out_features: int, sigma_init: float = 0.5):
        super().__init__()
        self.in_features  = in_features
        self.out_features = out_features
        self.sigma_init   = sigma_init

        # Learnable parameters: mean (μ) and std-scale (σ) for weights and biases
        self.weight_mu    = nn.Parameter(torch.empty(out_features, in_features))
        self.weight_sigma = nn.Parameter(torch.empty(out_features, in_features))
        self.bias_mu      = nn.Parameter(torch.empty(out_features))
        self.bias_sigma   = nn.Parameter(torch.empty(out_features))

        # Fixed noise buffers (not learnable)
        self.register_buffer('weight_epsilon', torch.empty(out_features, in_features))
        self.register_buffer('bias_epsilon',   torch.empty(out_features))

        self.reset_parameters()
        self.sample_noise()

    def reset_parameters(self):
        # μ initialized uniformly like a standard linear layer
        bound = 1.0 / math.sqrt(self.in_features)
        self.weight_mu.data.uniform_(-bound, bound)
        self.bias_mu.data.uniform_(-bound, bound)
        # σ initialized to σ_init / sqrt(p) (factorised NoisyNets paper)
        self.weight_sigma.data.fill_(self.sigma_init / math.sqrt(self.in_features))
        self.bias_sigma.data.fill_(self.sigma_init / math.sqrt(self.out_features))

    @staticmethod
    def _scale_noise(size: int) -> torch.Tensor:
        """Factorised noise: f(x) = sgn(x) * sqrt(|x|)"""
        x = torch.randn(size)
        return x.sign().mul(x.abs().sqrt())

    def sample_noise(self):
        """Resample noise buffers — called once per update step."""
        eps_in  = self._scale_noise(self.in_features)
        eps_out = self._scale_noise(self.out_features)
        # Outer product for factorised noise
        self.weight_epsilon.copy_(eps_out.outer(eps_in))
        self.bias_epsilon.copy_(eps_out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.training:
            weight = self.weight_mu + self.weight_sigma * self.weight_epsilon
            bias   = self.bias_mu   + self.bias_sigma   * self.bias_epsilon
        else:
            # At test time use mean weights only (deterministic greedy)
            weight = self.weight_mu
            bias   = self.bias_mu
        return F.linear(x, weight, bias)


# ─────────────────────────────────────────────
# Component 2: Dueling + Noisy Network
# Architecture:
#   Input → shared feature extractor
#         ├→ Value stream V(s)        [NoisyLinear]
#         └→ Advantage stream A(s,a)  [NoisyLinear]
#   Q(s,a) = V(s) + A(s,a) − mean(A(s,a'))
# ─────────────────────────────────────────────
class RainbowNet(nn.Module):
    """
    Dueling Network with NoisyLinear heads.
    - Shared feature backbone uses standard Linear (input is small/deterministic).
    - Both Value and Advantage streams use NoisyLinear for implicit exploration.
    """
    def __init__(self, input_dim: int = 64, hidden_dim: int = 128, action_dim: int = 4,
                 sigma_init: float = 0.5):
        super().__init__()

        # Shared feature extractor (standard linear — input is already small & fixed)
        self.feature = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
        )

        # Value stream: s → scalar V(s)
        self.value_hidden = NoisyLinear(hidden_dim, hidden_dim, sigma_init)
        self.value_out    = NoisyLinear(hidden_dim, 1,          sigma_init)

        # Advantage stream: s → vector A(s, ·)
        self.adv_hidden   = NoisyLinear(hidden_dim, hidden_dim, sigma_init)
        self.adv_out      = NoisyLinear(hidden_dim, action_dim, sigma_init)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat  = self.feature(x)

        val   = F.relu(self.value_hidden(feat))
        val   = self.value_out(val)                        # (B, 1)

        adv   = F.relu(self.adv_hidden(feat))
        adv   = self.adv_out(adv)                          # (B, A)

        # Dueling combination with mean-centering (Wang et al., 2016)
        q = val + adv - adv.mean(dim=1, keepdim=True)     # (B, A)
        return q

    def sample_noise(self):
        """Resample noise in all NoisyLinear layers."""
        for m in self.modules():
            if isinstance(m, NoisyLinear):
                m.sample_noise()


# ─────────────────────────────────────────────
# Component 3: Prioritized Experience Replay (PER)
# Key idea: transitions with higher TD-error are sampled more often.
# Uses a SumTree for O(log N) priority updates and sampling.
# ─────────────────────────────────────────────
class SumTree:
    """
    Binary SumTree for efficient O(log N) priority sampling.
    Leaf nodes store individual priorities; internal nodes store sums.
    """
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.tree     = np.zeros(2 * capacity - 1, dtype=np.float64)
        self.data     = np.empty(capacity, dtype=object)
        self.write    = 0   # circular write pointer
        self.n_entries = 0

    def _propagate(self, idx: int, change: float):
        parent = (idx - 1) // 2
        self.tree[parent] += change
        if parent != 0:
            self._propagate(parent, change)

    def _retrieve(self, idx: int, s: float) -> int:
        left  = 2 * idx + 1
        right = left + 1
        if left >= len(self.tree):
            return idx
        if s <= self.tree[left]:
            return self._retrieve(left, s)
        else:
            return self._retrieve(right, s - self.tree[left])

    @property
    def total(self) -> float:
        return self.tree[0]

    def add(self, priority: float, data):
        idx = self.write + self.capacity - 1
        self.data[self.write] = data
        self.update(idx, priority)
        self.write = (self.write + 1) % self.capacity
        self.n_entries = min(self.n_entries + 1, self.capacity)

    def update(self, idx: int, priority: float):
        change = priority - self.tree[idx]
        self.tree[idx] = priority
        self._propagate(idx, change)

    def get(self, s: float):
        idx      = self._retrieve(0, s)
        data_idx = idx - self.capacity + 1
        return idx, self.tree[idx], self.data[data_idx]


Transition = namedtuple('Transition', ['state', 'action', 'reward', 'next_state', 'done'])


class PrioritizedReplayBuffer:
    """
    Prioritized Experience Replay buffer (Schaul et al., 2015).

    Sampling probability:  P(i) = p_i^α / Σ p_j^α
    Importance-sampling weight: w_i = (N · P(i))^{-β} / max_j w_j
    α controls how much prioritization is used (0 = uniform).
    β anneals from β_start → 1.0 to correct for the bias.
    """
    def __init__(self, capacity: int = 5000, alpha: float = 0.6,
                 beta_start: float = 0.4, beta_frames: int = 100_000):
        self.capacity    = capacity
        self.alpha       = alpha
        self.beta_start  = beta_start
        self.beta_frames = beta_frames
        self.frame       = 1          # counts total add() calls for β annealing
        self.tree        = SumTree(capacity)
        self.max_priority = 1.0      # new transitions start with max priority

    @property
    def beta(self) -> float:
        """Linearly anneal β from beta_start to 1.0."""
        return min(1.0, self.beta_start + self.frame * (1.0 - self.beta_start) / self.beta_frames)

    def push(self, state, action, reward, next_state, done):
        transition = Transition(state, action, reward, next_state, done)
        self.tree.add(self.max_priority ** self.alpha, transition)
        self.frame += 1

    def sample(self, batch_size: int):
        batch_indices = []
        batch_data    = []
        is_weights    = []

        segment = self.tree.total / batch_size
        for i in range(batch_size):
            s         = random.uniform(segment * i, segment * (i + 1))
            idx, p, t = self.tree.get(s)
            batch_indices.append(idx)
            batch_data.append(t)

        # Compute importance-sampling weights
        total  = self.tree.total
        n      = self.tree.n_entries
        beta   = self.beta
        probs  = np.array([self.tree.tree[i] for i in batch_indices]) / total
        # Clamp probabilities to avoid division by zero
        probs  = np.maximum(probs, 1e-8)
        weights = (n * probs) ** (-beta)
        weights /= weights.max()          # normalize so max weight = 1

        states      = np.stack([t.state      for t in batch_data])
        actions     = np.array([t.action     for t in batch_data])
        rewards     = np.array([t.reward     for t in batch_data], dtype=np.float32)
        next_states = np.stack([t.next_state for t in batch_data])
        dones       = np.array([t.done       for t in batch_data], dtype=bool)

        return (states, actions, rewards, next_states, dones,
                np.array(batch_indices), weights.astype(np.float32))

    def update_priorities(self, indices, td_errors):
        """Update priorities after computing new TD errors."""
        for idx, err in zip(indices, td_errors):
            priority = (abs(err) + 1e-6) ** self.alpha
            self.tree.update(idx, priority)
            self.max_priority = max(self.max_priority, priority)

    def __len__(self):
        return self.tree.n_entries


# ─────────────────────────────────────────────
# Component 4: Multi-step Return Buffer
# Collects n consecutive transitions and returns a compressed
# (s_t, a_t, R_t^{(n)}, s_{t+n}, done) transition where:
#   R_t^{(n)} = r_t + γ·r_{t+1} + ... + γ^{n-1}·r_{t+n-1}
# This provides a richer gradient signal and faster credit assignment.
# ─────────────────────────────────────────────
class MultiStepBuffer:
    """
    Rolling buffer that accumulates n-step transitions before
    pushing a compressed multi-step transition to the PER buffer.
    """
    def __init__(self, n_step: int, gamma: float, per_buffer: PrioritizedReplayBuffer):
        self.n_step     = n_step
        self.gamma      = gamma
        self.per_buffer = per_buffer
        self.buffer     = deque(maxlen=n_step)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

        if len(self.buffer) == self.n_step:
            # Compute n-step discounted return
            R = 0.0
            for i, (s, a, r, ns, d) in enumerate(self.buffer):
                R += (self.gamma ** i) * r

            s0  = self.buffer[0][0]   # state at t
            a0  = self.buffer[0][1]   # action at t
            sn  = next_state          # state at t+n
            dn  = done                # done flag at t+n
            self.per_buffer.push(s0, a0, R, sn, dn)

        if done:
            self.buffer.clear()


# ─────────────────────────────────────────────
# Helper
# ─────────────────────────────────────────────
def preprocess_state(state: np.ndarray) -> np.ndarray:
    """Flatten 4×4×4 board tensor to a 64-dim float32 vector."""
    return state.flatten().astype(np.float32)


# ─────────────────────────────────────────────
# Rainbow DQN Training Loop
# ─────────────────────────────────────────────
def train_rainbow(
    epochs:       int   = 3000,
    hidden_dim:   int   = 128,
    lr:           float = 5e-4,
    gamma:        float = 0.99,
    batch_size:   int   = 64,
    n_step:       int   = 3,
    target_update:int   = 200,   # hard-update every N steps (global)
    buffer_cap:   int   = 10_000,
    alpha:        float = 0.6,
    beta_start:   float = 0.4,
    beta_frames:  int   = 50_000,
    sigma_init:   float = 0.5,
    max_episode_steps: int = 60,
) -> list:

    action_dim = 4
    input_dim  = 64   # 4×4 grid × 4 channels

    # Networks (Double DQN: online + target)
    online_net = RainbowNet(input_dim, hidden_dim, action_dim, sigma_init)
    target_net = copy.deepcopy(online_net)
    target_net.eval()

    optimizer = torch.optim.Adam(online_net.parameters(), lr=lr)

    # Buffers
    per_buffer  = PrioritizedReplayBuffer(buffer_cap, alpha, beta_start, beta_frames)
    multi_step  = MultiStepBuffer(n_step, gamma, per_buffer)

    rewards        = []
    global_step    = 0
    gamma_n        = gamma ** n_step   # discount for n-step bootstrapped target

    print("=" * 60)
    print("  Rainbow DQN — Random Mode GridWorld")
    print(f"  Components: Double DQN | Dueling Net | PER | "
          f"Multi-step(n={n_step}) | NoisyNets")
    print("=" * 60)

    for episode in range(epochs):
        game  = Gridworld(size=4, mode='random')
        state = preprocess_state(game.board.render_np())

        episode_reward = 0
        step_count     = 0
        done           = False

        while not done:
            step_count  += 1
            global_step += 1

            # ── Action selection via NoisyNet (no ε-greedy needed) ──
            online_net.train()          # keep noise active during acting
            online_net.sample_noise()   # resample noise each step
            with torch.no_grad():
                q_vals = online_net(torch.FloatTensor(state).unsqueeze(0))
                action = q_vals.argmax(dim=1).item()

            # ── Environment step ──
            game.makeMove(action)
            next_state     = preprocess_state(game.board.render_np())
            reward         = game.reward()
            episode_reward += reward

            terminal = (reward != -1) or (step_count >= max_episode_steps)
            done     = terminal

            # ── Store n-step transition ──
            multi_step.push(state, action, reward, next_state, done)
            state = next_state

            # ── Learning step ──
            if len(per_buffer) >= batch_size:
                (s_b, a_b, r_b, ns_b, d_b,
                 indices, is_weights) = per_buffer.sample(batch_size)

                s_b  = torch.FloatTensor(s_b)
                a_b  = torch.LongTensor(a_b)
                r_b  = torch.FloatTensor(r_b)
                ns_b = torch.FloatTensor(ns_b)
                d_b  = torch.BoolTensor(d_b)
                w_b  = torch.FloatTensor(is_weights)

                # ── Double DQN target ──
                online_net.train()
                online_net.sample_noise()
                q_pred = online_net(s_b).gather(1, a_b.unsqueeze(1)).squeeze(1)

                with torch.no_grad():
                    # Action chosen by online net, evaluated by target net
                    target_net.sample_noise()
                    next_actions = online_net(ns_b).argmax(dim=1, keepdim=True)
                    q_next       = target_net(ns_b).gather(1, next_actions).squeeze(1)
                    q_target     = r_b + gamma_n * q_next * (~d_b)

                td_errors = (q_pred - q_target).detach().abs().cpu().numpy()

                # ── Weighted MSE loss (IS weights correct PER bias) ──
                loss = (w_b * F.mse_loss(q_pred, q_target, reduction='none')).mean()

                optimizer.zero_grad()
                loss.backward()
                # Gradient clipping for stability
                torch.nn.utils.clip_grad_norm_(online_net.parameters(), max_norm=10.0)
                optimizer.step()

                # ── Update PER priorities ──
                per_buffer.update_priorities(indices, td_errors)

            # ── Hard target network update ──
            if global_step % target_update == 0:
                target_net.load_state_dict(online_net.state_dict())

        rewards.append(episode_reward)

        if (episode + 1) % 200 == 0:
            avg = np.mean(rewards[-100:])
            print(f"  Episode {episode+1:4d} | Buffer: {len(per_buffer):5d} | "
                  f"Avg Reward (last 100): {avg:6.2f}")

    return rewards


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────
def main():
    rewards = train_rainbow(epochs=3000)

    # ── Plot ──
    window     = 100
    moving_avg = np.convolve(rewards, np.ones(window) / window, mode='valid')

    plt.figure(figsize=(11, 5))

    # Raw rewards (faint)
    plt.subplot(1, 2, 1)
    plt.plot(rewards, alpha=0.25, color='steelblue', label='Episode Reward')
    plt.plot(np.arange(window - 1, len(rewards)),
             moving_avg, color='steelblue', linewidth=2, label=f'MA-{window}')
    plt.title('Rainbow DQN — Random Mode GridWorld')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.legend()
    plt.grid(alpha=0.3)

    # Moving average only (cleaner view)
    plt.subplot(1, 2, 2)
    plt.plot(moving_avg, color='darkorange', linewidth=2)
    plt.title(f'Smoothed Reward (window={window})')
    plt.xlabel(f'Episode')
    plt.ylabel('Reward')
    plt.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig('plots/hw3_4_rainbow.png', dpi=150)
    print("\nSaved plot to plots/hw3_4_rainbow.png")

    final_avg = np.mean(rewards[-100:])
    print(f"Final Avg Reward (last 100 episodes): {final_avg:.2f}")


if __name__ == "__main__":
    main()
