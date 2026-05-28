"""
Q-Learning Demo on FrozenLake-v1
=================================
This script demonstrates the fundamental Q-Learning algorithm studied in:
  - Lecture3.pdf (Q-Learning algorithm)
  - gym-Env-DQN-lab1.pdf (Lab 1: FrozenLake implementation)
  - Bellman Equation.pdf (Bellman update rule)

The Bellman update rule used here:
  Q(s,a) <- Q(s,a) + alpha * [r + gamma * max Q(s',a') - Q(s,a)]

This serves as the BASELINE comparison for our PPO agent in the report.
"""

import numpy as np
import gymnasium as gym
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for saving plots
import matplotlib.pyplot as plt
import os

# ----------------------------------------------------------------
# HYPERPARAMETERS (from Lecture3.pdf and Lab 1)
# ----------------------------------------------------------------
TOTAL_EPISODES = 10000     # Total training episodes
MAX_STEPS      = 100       # Max steps per episode
ALPHA          = 0.85      # Learning rate  (α)
GAMMA          = 0.95      # Discount factor (γ) — from Bellman equation
EPSILON        = 1.0       # Initial exploration rate (ε-greedy)
EPSILON_MIN    = 0.01      # Minimum epsilon
EPSILON_DECAY  = 0.995     # Decay rate per episode

# ----------------------------------------------------------------
# ENVIRONMENT SETUP
# ----------------------------------------------------------------
print("=" * 50)
print("  Q-Learning Demo: FrozenLake-v1")
print("=" * 50)

env = gym.make('FrozenLake-v1', is_slippery=True)
n_states  = env.observation_space.n   # 16 states (4x4 grid)
n_actions = env.action_space.n        # 4 actions: left, down, right, up

print(f"States : {n_states}")
print(f"Actions: {n_actions}")
print(f"Hyperparameters: alpha={ALPHA}, gamma={GAMMA}, epsilon={EPSILON}")
print()

# ----------------------------------------------------------------
# INITIALIZE Q-TABLE (all zeros — Bellman start condition)
# ----------------------------------------------------------------
Q = np.zeros((n_states, n_actions))
print(f"Q-Table initialized with shape: {Q.shape}")

# ----------------------------------------------------------------
# EPSILON-GREEDY ACTION SELECTION
# (studied in Lecture3.pdf — Exploration vs Exploitation)
# ----------------------------------------------------------------
def choose_action(state, epsilon):
    """
    Epsilon-greedy policy:
    - With probability epsilon: explore randomly
    - With probability (1-epsilon): exploit best known action
    """
    if np.random.uniform(0, 1) < epsilon:
        return env.action_space.sample()   # Explore
    else:
        return np.argmax(Q[state, :])      # Exploit

# ----------------------------------------------------------------
# BELLMAN UPDATE — Q-LEARNING
# Q(s,a) <- Q(s,a) + alpha * [r + gamma * max_a'(Q(s',a')) - Q(s,a)]
# (from Bellman Equation.pdf and Lecture3.pdf)
# ----------------------------------------------------------------
def bellman_update(state, action, reward, next_state):
    """Applies the Bellman optimality equation to update the Q-table."""
    td_target = reward + GAMMA * np.max(Q[next_state, :])
    td_error  = td_target - Q[state, action]
    Q[state, action] = Q[state, action] + ALPHA * td_error

# ----------------------------------------------------------------
# TRAINING LOOP
# ----------------------------------------------------------------
rewards_per_episode = []
epsilon = EPSILON

print("\nStarting Q-Learning training...")
for episode in range(TOTAL_EPISODES):
    state, _ = env.reset()
    total_reward = 0

    for step in range(MAX_STEPS):
        action     = choose_action(state, epsilon)
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

        bellman_update(state, action, reward, next_state)

        state        = next_state
        total_reward += reward

        if done:
            break

    # Decay epsilon (reduce exploration over time)
    epsilon = max(EPSILON_MIN, epsilon * EPSILON_DECAY)
    rewards_per_episode.append(total_reward)

    if (episode + 1) % 1000 == 0:
        avg = np.mean(rewards_per_episode[-1000:])
        print(f"  Episode {episode+1:5d}/{TOTAL_EPISODES} | "
              f"Avg Reward (last 1000): {avg:.3f} | Epsilon: {epsilon:.3f}")

env.close()

# ----------------------------------------------------------------
# EVALUATION
# ----------------------------------------------------------------
print("\n" + "=" * 50)
print("  EVALUATION (100 episodes, greedy policy)")
print("=" * 50)

eval_env    = gym.make('FrozenLake-v1', is_slippery=True)
total_wins  = 0
EVAL_EPISODES = 100

for _ in range(EVAL_EPISODES):
    state, _ = eval_env.reset()
    for _ in range(MAX_STEPS):
        action = np.argmax(Q[state, :])   # Greedy (no exploration)
        state, reward, terminated, truncated, _ = eval_env.step(action)
        if terminated or truncated:
            total_wins += reward
            break

eval_env.close()
win_rate = (total_wins / EVAL_EPISODES) * 100
print(f"Win Rate after Q-Learning: {win_rate:.1f}%")
print(f"(Random agent baseline: ~1.5%)")

# ----------------------------------------------------------------
# GENERATE LEARNING CURVE PLOT
# ----------------------------------------------------------------
print("\nGenerating learning curve plot...")

window = 200
smoothed = np.convolve(rewards_per_episode,
                       np.ones(window) / window, mode='valid')

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle('Q-Learning on FrozenLake-v1 (Bellman Equation)', fontsize=14, fontweight='bold')

# Plot 1: Raw + Smoothed rewards
axes[0].plot(rewards_per_episode, alpha=0.2, color='steelblue', label='Raw reward')
axes[0].plot(range(window - 1, TOTAL_EPISODES),
             smoothed, color='steelblue', linewidth=2, label=f'Smoothed (window={window})')
axes[0].set_xlabel('Episode')
axes[0].set_ylabel('Reward')
axes[0].set_title('Learning Curve: Reward per Episode')
axes[0].legend()
axes[0].grid(True, linestyle='--', alpha=0.5)

# Plot 2: Q-Table heatmap (4x4 grid — best action value per state)
best_q = np.max(Q, axis=1).reshape(4, 4)
im = axes[1].imshow(best_q, cmap='hot', interpolation='nearest')
plt.colorbar(im, ax=axes[1])
axes[1].set_title('Q-Table Heatmap: max Q(s,a) per state')
axes[1].set_xlabel('Grid Column')
axes[1].set_ylabel('Grid Row')
for i in range(4):
    for j in range(4):
        axes[1].text(j, i, f'{best_q[i, j]:.2f}',
                     ha='center', va='center', color='cyan', fontsize=8)

plt.tight_layout()

save_dir  = os.path.join(os.path.dirname(__file__), '..', '..', 'results')
os.makedirs(save_dir, exist_ok=True)
save_path = os.path.join(save_dir, 'q_learning_frozenlake.png')
plt.savefig(save_path, dpi=300, bbox_inches='tight')
print(f"Plot saved to: {os.path.abspath(save_path)}")

print("\nQ-Learning Demo Complete!")
print(f"Final Q-Table:\n{Q.round(3)}")
