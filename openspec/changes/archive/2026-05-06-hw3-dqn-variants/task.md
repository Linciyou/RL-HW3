# Task: Reinforcement Learning HW3 - DQN and its variants

## Objective
Implement Deep Q-Networks (DQN) and its variants based on the DeepReinforcementLearningInAction repository.

## Requirements

1. **HW3-1: Naive DQN for static mode [30%]**
   - Run the provided code: naive DQN or Experience buffer replay.
   - Clarify understanding of the code.
   - Submit an understanding report including Basic DQN implementation for an easy environment and Experience Replay Buffer.
   - The grid environment must be configured to `mode='static'` (Player at (0,3), Goal at (0,0), Pit at (0,1), Wall at (1,1)).

2. **HW3-2: Enhanced DQN Variants for player mode [40%]**
   - Implement **Double DQN**.
   - Implement **Dueling DQN**.
   - Compare them and focus on how they improve upon the basic DQN approach.
   - The grid environment must be configured to `mode='player'` (Random player position, other objects static).

3. **HW3-3: Enhance DQN for random mode WITH Training Tips [30%]**
   - Convert the DQN model from PyTorch to **PyTorch Lightning** (or Keras).
   - Integrate training techniques to stabilize/improve learning (e.g., gradient clipping, learning rate scheduling).
   - The grid environment must be configured to `mode='random'` (All objects random).

## Workflow
1. Locate the grid world and DQN code from `DeepReinforcementLearningInAction/Chapter 3`.
2. Re-implement and split into three scripts or notebooks (`hw3_1_naive.py`, `hw3_2_variants.py`, `hw3_3_lightning.py`).
3. Draft the understanding report for HW3-1.
