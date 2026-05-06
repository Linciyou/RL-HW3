import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from collections import deque
import random
from Gridworld import Gridworld

# 1. Experience Replay Buffer
class ReplayBuffer:
    def __init__(self, capacity=5000):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
        
    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = map(np.stack, zip(*batch))
        return state, action, reward, next_state, done
    
    def __len__(self):
        return len(self.buffer)

# 2. Q-Network
class QNetwork(nn.Module):
    def __init__(self, input_dim=64, hidden_dim=150, output_dim=4):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, x):
        x = self.relu1(self.fc1(x))
        x = self.relu2(self.fc2(x))
        return self.fc3(x)

def preprocess_state(state):
    state_flat = state.flatten()
    return torch.FloatTensor(state_flat).unsqueeze(0)

def main():
    action_dim = 4
    
    # 64 parameters is 4x4 grid * 4 layers
    model = QNetwork(64, 150, action_dim)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = torch.nn.MSELoss()
    
    replay_buffer = ReplayBuffer(capacity=1000)
    batch_size = 64
    gamma = 0.9
    epsilon = 1.0
    epsilon_min = 0.1
    epsilon_decay = 0.995
    
    epochs = 1500
    
    losses = []
    rewards = []
    
    for i in range(epochs):
        game = Gridworld(size=4, mode='static')
        state_ = game.board.render_np()
        state = preprocess_state(state_)
        status = 1
        
        episode_reward = 0
        step_count = 0
        
        while status == 1:
            step_count += 1
            # Epsilon-greedy
            qval = model(state)
            if random.random() < epsilon:
                action = np.random.randint(0, action_dim)
            else:
                action = torch.argmax(qval).item()
                
            game.makeMove(action)
            next_state_ = game.board.render_np()
            next_state = preprocess_state(next_state_)
            reward = game.reward()
            episode_reward += reward
            
            if reward != -1 or step_count > 50:
                status = 0
                done = True
            else:
                done = False
                
            replay_buffer.push(state.numpy()[0], action, reward, next_state.numpy()[0], done)
            state = next_state
            
            if len(replay_buffer) > batch_size:
                s_batch, a_batch, r_batch, ns_batch, done_batch = replay_buffer.sample(batch_size)
                s_batch = torch.FloatTensor(s_batch)
                a_batch = torch.LongTensor(a_batch)
                r_batch = torch.FloatTensor(r_batch)
                ns_batch = torch.FloatTensor(ns_batch)
                done_batch = torch.BoolTensor(done_batch)
                
                qval_batch = model(s_batch)
                qval_batch = qval_batch.gather(1, a_batch.unsqueeze(1)).squeeze(1)
                
                with torch.no_grad():
                    nqval_batch = model(ns_batch)
                    max_nqval = nqval_batch.max(1)[0]
                    target_qval = r_batch + gamma * max_nqval * (~done_batch)
                    
                loss = loss_fn(qval_batch, target_qval)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                losses.append(loss.item())
                
        rewards.append(episode_reward)
        if epsilon > epsilon_min:
            epsilon *= epsilon_decay
            
        if (i+1) % 100 == 0:
            avg_reward = np.mean(rewards[-100:])
            print(f"Episode: {i+1}, Epsilon: {epsilon:.3f}, Avg Reward (last 100): {avg_reward:.2f}")

    final_avg_reward = np.mean(rewards[-100:])
    print(f"Final Avg Reward (last 100 episodes): {final_avg_reward:.2f}")

    # Plot
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(losses)
    plt.title('Training Loss (Naive DQN)')
    plt.xlabel('Steps')
    plt.ylabel('Loss')

    plt.subplot(1, 2, 2)
    window = 50
    moving_avg = np.convolve(rewards, np.ones(window)/window, mode='valid')
    plt.plot(moving_avg)
    plt.title('Rewards Moving Average (Naive DQN)')
    plt.xlabel('Episodes')
    plt.ylabel('Reward')
    
    plt.tight_layout()
    plt.savefig('plots/hw3_1_training.png')
    print("Saved plot to plots/hw3_1_training.png")

if __name__ == "__main__":
    main()
