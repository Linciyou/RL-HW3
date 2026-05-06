## 1. Setup Environment

- [x] 1.1 Verify and setup `Gridworld.py` and `GridBoard.py` dependencies (NumPy, Matplotlib, PyTorch)
- [x] 1.2 Setup `plots` directory for saving training result graphs

## 2. HW3-1: Naive DQN implementation

- [x] 2.1 Implement `hw3_1_naive.py`: Setup Q-network model, epsilon-greedy policy, and Experience Replay Buffer
- [x] 2.2 Implement training loop for Naive DQN in `static` mode
- [x] 2.3 Add logging and plot generation for training loss and rewards (`plots/hw3_1_training.png`)
- [x] 2.4 Verify agent learns to reach goal (avg reward > 0 over last 100 eps)

## 3. HW3-2: Double DQN and Dueling DQN

- [x] 3.1 Implement `hw3_2_variants.py`: Setup Double DQN structure (target and online networks) with `player` mode
- [x] 3.2 Implement Dueling DQN architecture (Value and Advantage streams) in `hw3_2_variants.py`
- [x] 3.3 Add training loops for both variants, tracking their performances
- [x] 3.4 Generate comparison plots and save to `plots/hw3_2_comparison.png`

## 4. HW3-3: PyTorch Lightning DQN

- [x] 4.1 Implement `hw3_3_lightning.py`: Define `DQNLightning` class inheriting from `lightning.LightningModule`
- [x] 4.2 Configure optimizer, learning rate scheduler, and target network soft updates
- [x] 4.3 Configure `Trainer` with `gradient_clip_val=1.0` and train on `random` mode
- [x] 4.4 Save training reward plots to `plots/hw3_3_training.png`

## 5. Documentation and Report

- [x] 5.1 Write HW3-1 understanding report inside `README.md`
- [x] 5.2 Add HW3-2 variant comparison to `README.md`
- [x] 5.3 Ensure all requirements match the specifications and design documents
