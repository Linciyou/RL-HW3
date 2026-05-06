# double-dqn Specification

## Purpose
TBD - created by archiving change hw3-dqn-variants. Update Purpose after archive.
## Requirements
### Requirement: Double DQN agent for player mode
The system SHALL implement a Double DQN agent where action selection uses the online network and Q-value estimation uses the target network, running on `player` mode.

#### Scenario: DDQN trains and improves
- **WHEN** `hw3_2_variants.py` runs the Double DQN agent for >= 2000 episodes in `player` mode
- **THEN** the average reward over the last 100 episodes SHALL be greater than the Naive DQN baseline

#### Scenario: Target network is used for value estimation only
- **WHEN** computing the DDQN target: `y = r + γ * Q_target(s', argmax_a Q_online(s', a))`
- **THEN** the online network selects the best action and the target network evaluates it

#### Scenario: Comparison plot is generated
- **WHEN** both Double DQN and Dueling DQN training complete
- **THEN** a comparison plot of rewards over episodes is saved to `plots/hw3_2_comparison.png`

