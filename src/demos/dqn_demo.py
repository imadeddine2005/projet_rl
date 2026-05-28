"""
Deep Q-Network (DQN) Demo on CartPole-v1
==========================================
This script demonstrates DQN studied in:
  - Lab2 (1).pdf  (DQN with Experience Replay, Target Network)
  - DRLSLides (6).pdf (Value-Based vs Policy-Based comparison)
  - Lecture3.pdf  (Function Approximation, Deep Q-Networks)

Key DQN innovations over plain Q-Learning (from the courses):
  1. Experience Replay Buffer  — breaks correlation between consecutive samples
  2. Target Network            — stabilises training (separate frozen network)
  3. Neural Network Q-function — handles continuous/large state spaces

This serves as the INTERMEDIATE BASELINE between Q-Learning and PPO in the report.
"""

import os
import random
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from collections import deque

import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim

# ----------------------------------------------------------------
# HYPERPARAMETERS (from Lab 2 and DRLSLides)
# ----------------------------------------------------------------
ENV_NAME       = 'CartPole-v1'
TOTAL_EPISODES = 500
MAX_STEPS      = 500
ALPHA          = 1e-3      # Learning rate
GAMMA          = 0.99      # Discount factor γ (Bellman)
EPSILON        = 1.0       # Initial ε for ε-greedy
EPSILON_MIN    = 0.01
EPSILON_DECAY  = 0.995
BATCH_SIZE     = 64        # Mini-batch size from replay buffer
MEMORY_SIZE    = 10000     # Replay buffer capacity
TARGET_UPDATE  = 10        # Update target network every N episodes

# ----------------------------------------------------------------
# NEURAL NETWORK: Q-FUNCTION APPROXIMATOR
# (replaces the Q-table from Q-Learning)
# ----------------------------------------------------------------
class DQNNetwork(nn.Module):
    """
    Simple MLP that approximates Q(s, a) for all actions simultaneously.
    Input : state (observation vector)
    Output: Q-values for each action
    """
    def __init__(self, state_size, action_size):
        super(DQNNetwork, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(state_size, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, action_size)
        )

    def forward(self, x):
        return self.network(x)

# ----------------------------------------------------------------
# EXPERIENCE REPLAY BUFFER
# (Key DQN contribution — studied in Lab 2)
# Stores (state, action, reward, next_state, done) tuples
# and samples random mini-batches to break temporal correlation
# ----------------------------------------------------------------
class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (np.array(states), np.array(actions),
                np.array(rewards, dtype=np.float32),
                np.array(next_states), np.array(dones, dtype=np.float32))

    def __len__(self):
        return len(self.buffer)

# ----------------------------------------------------------------
# DQN AGENT
# ----------------------------------------------------------------
class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size  = state_size
        self.action_size = action_size
        self.epsilon     = EPSILON
        self.device      = torch.device('cpu')

        # Online network (trained every step)
        self.q_network = DQNNetwork(state_size, action_size).to(self.device)

        # Target network (frozen copy, updated periodically)
        # This is the key stabilisation trick from DQN (Mnih et al., 2015)
        self.target_network = DQNNetwork(state_size, action_size).to(self.device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.target_network.eval()

        self.optimizer = optim.Adam(self.q_network.parameters(), lr=ALPHA)
        self.memory    = ReplayBuffer(MEMORY_SIZE)
        self.criterion = nn.MSELoss()

    def act(self, state):
        """Epsilon-greedy action selection (studied in Lecture3.pdf)."""
        if np.random.rand() < self.epsilon:
            return random.randrange(self.action_size)  # Explore
        state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_values = self.q_network(state_t)
        return q_values.argmax().item()                # Exploit

    def remember(self, state, action, reward, next_state, done):
        """Store transition in replay buffer."""
        self.memory.push(state, action, reward, next_state, done)

    def replay(self):
        """
        Sample a mini-batch and update Q-network using the Bellman equation:
        Target = r + γ * max_a' Q_target(s', a')   [if not done]
        Target = r                                   [if done]
        Loss   = MSE(Q_online(s,a), Target)
        """
        if len(self.memory) < BATCH_SIZE:
            return

        states, actions, rewards, next_states, dones = self.memory.sample(BATCH_SIZE)

        states_t      = torch.FloatTensor(states).to(self.device)
        actions_t     = torch.LongTensor(actions).to(self.device)
        rewards_t     = torch.FloatTensor(rewards).to(self.device)
        next_states_t = torch.FloatTensor(next_states).to(self.device)
        dones_t       = torch.FloatTensor(dones).to(self.device)

        # Current Q-values from online network
        current_q = self.q_network(states_t).gather(1, actions_t.unsqueeze(1)).squeeze(1)

        # Target Q-values from frozen target network (Bellman)
        with torch.no_grad():
            max_next_q = self.target_network(next_states_t).max(1)[0]
            target_q   = rewards_t + GAMMA * max_next_q * (1 - dones_t)

        loss = self.criterion(current_q, target_q)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def update_target_network(self):
        """Periodically copy online network weights to target network."""
        self.target_network.load_state_dict(self.q_network.state_dict())

    def decay_epsilon(self):
        """Reduce exploration rate over time."""
        self.epsilon = max(EPSILON_MIN, self.epsilon * EPSILON_DECAY)


# ----------------------------------------------------------------
# MAIN TRAINING LOOP
# ----------------------------------------------------------------
def train():
    print("=" * 55)
    print("  DQN Demo: CartPole-v1")
    print("=" * 55)

    env        = gym.make(ENV_NAME)
    state_size  = env.observation_space.shape[0]
    action_size = env.action_space.n
    agent      = DQNAgent(state_size, action_size)

    print(f"State size : {state_size}")
    print(f"Action size: {action_size}")
    print(f"Device     : {agent.device}")
    print(f"Hyperparams: alpha={ALPHA}, gamma={GAMMA}, batch={BATCH_SIZE}")
    print()

    rewards_history = []
    scores_window   = deque(maxlen=100)

    for episode in range(1, TOTAL_EPISODES + 1):
        state, _ = env.reset()
        total_reward = 0

        for _ in range(MAX_STEPS):
            action               = agent.act(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done                 = terminated or truncated

            agent.remember(state, action, reward, next_state, done)
            agent.replay()      # Learn from experience replay

            state        = next_state
            total_reward += reward

            if done:
                break

        agent.decay_epsilon()
        rewards_history.append(total_reward)
        scores_window.append(total_reward)

        # Update target network periodically
        if episode % TARGET_UPDATE == 0:
            agent.update_target_network()

        if episode % 50 == 0:
            avg = np.mean(scores_window)
            print(f"  Episode {episode:4d}/{TOTAL_EPISODES} | "
                  f"Avg Score (last 100): {avg:6.1f} | "
                  f"Epsilon: {agent.epsilon:.3f}")

        # CartPole is considered solved at avg score >= 475
        if np.mean(scores_window) >= 475.0:
            print(f"\n  ✅ CartPole SOLVED at episode {episode}! "
                  f"Avg score: {np.mean(scores_window):.1f}")
            break

    env.close()

    # ----------------------------------------------------------------
    # GENERATE LEARNING CURVE PLOT
    # ----------------------------------------------------------------
    print("\nGenerating DQN learning curve plot...")

    window   = 50
    smoothed = np.convolve(rewards_history,
                           np.ones(window) / window, mode='valid')

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(rewards_history, alpha=0.3, color='darkorange', label='Raw score')
    ax.plot(range(window - 1, len(rewards_history)),
            smoothed, color='darkorange', linewidth=2,
            label=f'Smoothed (window={window})')
    ax.axhline(y=475, color='green', linestyle='--', linewidth=1.5,
               label='Solved threshold (475)')
    ax.set_xlabel('Episode')
    ax.set_ylabel('Total Reward')
    ax.set_title('DQN Learning Curve: CartPole-v1\n'
                 '(Experience Replay + Target Network)', fontweight='bold')
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()

    save_dir  = os.path.join(os.path.dirname(__file__), '..', '..', 'results')
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, 'dqn_cartpole.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Plot saved to: {os.path.abspath(save_path)}")
    print("\nDQN Demo Complete!")
    return rewards_history


if __name__ == "__main__":
    train()
