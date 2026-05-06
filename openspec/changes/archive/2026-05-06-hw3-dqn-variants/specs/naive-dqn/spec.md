## ADDED Requirements

### Requirement: Naive DQN agent for static mode
The system SHALL implement a Deep Q-Network (DQN) agent that learns to navigate a 4×4 grid world in `static` mode using a fully connected neural network as the Q-function approximator and an Experience Replay Buffer.

#### Scenario: Agent trains without errors
- **WHEN** `hw3_1_naive.py` is run
- **THEN** the agent completes training for a configurable number of episodes without errors

#### Scenario: Experience Replay Buffer stores and samples transitions
- **WHEN** a transition (state, action, reward, next_state, done) is added to the buffer
- **THEN** the buffer stores it and can randomly sample a mini-batch of size 64 for training

#### Scenario: Agent learns to reach the goal
- **WHEN** training is complete (>= 1000 episodes)
- **THEN** the agent's average reward over the last 100 episodes SHALL be positive (> 0)

#### Scenario: Training loss decreases over time
- **WHEN** training runs for 1000 episodes
- **THEN** a plot of training loss and reward is saved to `plots/hw3_1_training.png`

### Requirement: Understanding report in README
The README.md SHALL contain an understanding report for HW3-1 explaining the DQN algorithm, the role of Experience Replay, and the Bellman equation.

#### Scenario: README contains DQN explanation
- **WHEN** the README is read
- **THEN** it SHALL include sections on: DQN algorithm overview, Experience Replay Buffer, Bellman equation, and observations from running the code
