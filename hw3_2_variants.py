import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from collections import deque
import random
import copy
from Gridworld import Gridworld

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

# Standard Q-Network for Double DQN
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

# Dueling Q-Network
class DuelingQNetwork(nn.Module):
    def __init__(self, input_dim=64, hidden_dim=150, output_dim=4):
        super(DuelingQNetwork, self).__init__()
        # Shared feature layer
        self.feature = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU()
        )
        # Value stream
        self.value_stream = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        # Advantage stream
        self.advantage_stream = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
        
    def forward(self, x):
        features = self.feature(x)
        values = self.value_stream(features)
        advantages = self.advantage_stream(features)
        # Q(s, a) = V(s) + (A(s, a) - mean(A(s, a)))
        qvals = values + (advantages - advantages.mean(dim=1, keepdim=True))
        return qvals

def preprocess_state(state):
    state_flat = state.flatten()
    return torch.FloatTensor(state_flat).unsqueeze(0)

def train_agent(agent_type='double', epochs=2000):
    action_dim = 4
    input_dim = 64
    
    if agent_type == 'double':
        online_net = QNetwork(input_dim, 150, action_dim)
        target_net = copy.deepcopy(online_net)
    elif agent_type == 'dueling':
        online_net = DuelingQNetwork(input_dim, 150, action_dim)
        # Using basic DQN updates for Dueling network or target net
        target_net = copy.deepcopy(online_net)
        
    optimizer = torch.optim.Adam(online_net.parameters(), lr=1e-3)
    loss_fn = torch.nn.MSELoss()
    
    replay_buffer = ReplayBuffer(capacity=2000)
    batch_size = 64
    gamma = 0.9
    epsilon = 1.0
    epsilon_min = 0.05
    epsilon_decay = 0.998
    target_update_freq = 50
    
    rewards = []
    
    for i in range(epochs):
        game = Gridworld(size=4, mode='player')
        state_ = game.board.render_np()
        state = preprocess_state(state_)
        status = 1
        
        episode_reward = 0
        step_count = 0
        
        while status == 1:
            step_count += 1
            qval = online_net(state)
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
                
                qval_batch = online_net(s_batch).gather(1, a_batch.unsqueeze(1)).squeeze(1)
                
                with torch.no_grad():
                    if agent_type == 'double':
                        # Double DQN: action selected by online net, evaluated by target net
                        next_action = online_net(ns_batch).argmax(1).unsqueeze(1)
                        target_qval = r_batch + gamma * target_net(ns_batch).gather(1, next_action).squeeze(1) * (~done_batch)
                    else:
                        # Dueling DQN: standard target network update (can also be combined with Double)
                        max_nqval = target_net(ns_batch).max(1)[0]
                        target_qval = r_batch + gamma * max_nqval * (~done_batch)
                    
                loss = loss_fn(qval_batch, target_qval)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
        rewards.append(episode_reward)
        if epsilon > epsilon_min:
            epsilon *= epsilon_decay
            
        if (i+1) % target_update_freq == 0:
            target_net.load_state_dict(online_net.state_dict())
            
        if (i+1) % 200 == 0:
            avg_reward = np.mean(rewards[-100:])
            print(f"[{agent_type.upper()}] Episode: {i+1}, Epsilon: {epsilon:.3f}, Avg Reward (last 100): {avg_reward:.2f}")
            
    return rewards

def main():
    print("Training Double DQN...")
    double_rewards = train_agent('double', epochs=2000)
    
    print("\nTraining Dueling DQN...")
    dueling_rewards = train_agent('dueling', epochs=2000)
    
    # Plotting
    window = 100
    double_ma = np.convolve(double_rewards, np.ones(window)/window, mode='valid')
    dueling_ma = np.convolve(dueling_rewards, np.ones(window)/window, mode='valid')
    
    plt.figure(figsize=(10, 6))
    plt.plot(double_ma, label='Double DQN')
    plt.plot(dueling_ma, label='Dueling DQN')
    plt.title('Double vs Dueling DQN (Player Mode)')
    plt.xlabel(f'Episodes (Moving Average window={window})')
    plt.ylabel('Reward')
    plt.legend()
    plt.grid()
    plt.savefig('plots/hw3_2_comparison.png')
    print("Saved comparison plot to plots/hw3_2_comparison.png")

if __name__ == "__main__":
    main()
