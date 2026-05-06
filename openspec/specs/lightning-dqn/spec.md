# lightning-dqn Specification

## Purpose
TBD - created by archiving change hw3-dqn-variants. Update Purpose after archive.
## Requirements
### Requirement: PyTorch Lightning DQN for random mode
The system SHALL implement a DQN agent using PyTorch Lightning's `LightningModule` that trains on `random` mode, with gradient clipping and learning rate scheduling integrated via the `Trainer`.

#### Scenario: Lightning module structure is correct
- **WHEN** the `DQNLightning` class is instantiated
- **THEN** it SHALL subclass `lightning.LightningModule`, implement `training_step`, `configure_optimizers`, and `train_dataloader`

#### Scenario: Gradient clipping is active during training
- **WHEN** the Lightning `Trainer` is configured
- **THEN** `gradient_clip_val=1.0` SHALL be set on the Trainer

#### Scenario: Learning rate scheduling is applied
- **WHEN** `configure_optimizers` is called
- **THEN** it SHALL return both an optimizer and a learning rate scheduler (e.g., CosineAnnealingLR)

#### Scenario: Agent trains in random mode
- **WHEN** `hw3_3_lightning.py` is run
- **THEN** the agent completes training on `random` mode without errors and saves a reward plot to `plots/hw3_3_training.png`

#### Scenario: Training stabilization is evident
- **WHEN** training reward is plotted
- **THEN** the smoothed reward curve SHALL show an upward trend over >= 2000 episodes

