# dueling-dqn Specification

## Purpose
TBD - created by archiving change hw3-dqn-variants. Update Purpose after archive.
## Requirements
### Requirement: Dueling DQN agent for player mode
The system SHALL implement a Dueling DQN agent with a network architecture that separates state-value V(s) and advantage A(s,a) streams, combined as Q(s,a) = V(s) + (A(s,a) - mean_a(A(s,a))), running on `player` mode.

#### Scenario: Dueling architecture is correctly structured
- **WHEN** the DuelingDQN model is instantiated
- **THEN** it SHALL have a shared feature layer, a separate value stream outputting shape (batch, 1), and an advantage stream outputting shape (batch, n_actions)

#### Scenario: Dueling DQN trains and improves
- **WHEN** `hw3_2_variants.py` runs the Dueling DQN agent for >= 2000 episodes in `player` mode
- **THEN** the average reward over the last 100 episodes SHALL be positive

#### Scenario: Comparison with Double DQN is discussed in README
- **WHEN** the README section on HW3-2 is read
- **THEN** it SHALL contain a comparison of Double DQN vs Dueling DQN discussing their respective advantages and results

