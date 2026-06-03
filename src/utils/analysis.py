import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import supersuit as ss
from stable_baselines3 import PPO

# We need to import the env setup from the parent directory
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from env_setup import create_soccer_env

# Ensure it finds the model regardless of where the script is run from
DEFAULT_MODEL = os.path.join(os.path.dirname(__file__), "..", "..", "models", "ppo_soccer_2v2_colab")

def run_analysis(model_path=DEFAULT_MODEL, num_episodes=10):
    """
    Runs the trained model for a set number of episodes and calculates statistics
    such as average rewards and episode lengths for Football Analysis.
    It also generates professional graphs for the academic report.
    """
    print(f"Loading model from {model_path} for analysis...")
    try:
        model = PPO.load(model_path)
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    # Create the environment without rendering for faster analysis
    env = create_soccer_env(render_mode=None)
    env = ss.pettingzoo_env_to_vec_env_v1(env)
    env = ss.concat_vec_envs_v1(env, num_vec_envs=1, num_cpus=1, base_class='stable_baselines3')
    
    episode_rewards = []
    episode_lengths = []
    
    print(f"Running {num_episodes} episodes for statistical analysis...")
    for episode in range(num_episodes):
        obs = env.reset()
        total_reward = 0
        steps = 0
        
        while steps < 1500: # Limit steps 
            action, _ = model.predict(obs, deterministic=True)
            obs, rewards, dones, infos = env.step(action)
            total_reward += np.sum(rewards)
            steps += 1
            
            if dones.any():
                break
                
        episode_rewards.append(total_reward)
        episode_lengths.append(steps)
        print(f"Episode {episode + 1}/{num_episodes} | Reward: {total_reward:.2f} | Steps: {steps}")

    print("\n" + "="*40)
    print("      FOOTBALL ANALYSIS RESULTS")
    print("="*40)
    print(f"Total Matches Analyzed  : {num_episodes}")
    print(f"Average Team Reward     : {np.mean(episode_rewards):.2f} ± {np.std(episode_rewards):.2f}")
    print(f"Average Match Length    : {np.mean(episode_lengths):.2f} steps")
    print(f"Maximum Reward Achieved : {np.max(episode_rewards):.2f}")
    print("="*40)
    
    env.close()
    
    # ---------------------------------------------------------
    # GENERATE PROFESSIONAL GRAPHS
    # ---------------------------------------------------------
    print("Generating performance graphs...")
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Graph 1: Rewards over episodes
    episodes_x = np.arange(1, num_episodes + 1)
    ax1.plot(episodes_x, episode_rewards, marker='o', linestyle='-', color='#1f77b4', linewidth=2)
    ax1.set_title('Team Reward per Match (Evaluation)', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Match Number', fontsize=12)
    ax1.set_ylabel('Total Accumulated Reward', fontsize=12)
    ax1.grid(True, linestyle='--', alpha=0.7)
    
    # Graph 2: Match Lengths
    ax2.bar(episodes_x, episode_lengths, color='#ff7f0e', alpha=0.8)
    ax2.set_title('Match Duration (Steps)', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Match Number', fontsize=12)
    ax2.set_ylabel('Number of Steps', fontsize=12)
    ax2.grid(True, linestyle='--', alpha=0.7, axis='y')
    
    plt.tight_layout()
    
    # Save the figure (both PNG and PDF)
    results_dir = os.path.join(os.path.dirname(__file__), "..", "..", "results")
    os.makedirs(results_dir, exist_ok=True)
    
    plt.savefig(os.path.join(results_dir, "football_analysis_results.png"), dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(results_dir, "football_analysis_results.pdf"), bbox_inches='tight')
    
    # Also save in LaTeX figures/ directory if it exists
    latex_fig_dir = os.path.join(os.path.dirname(__file__), "..", "..", "rapport", "IA-Template-Report-main_1", "figures")
    if os.path.exists(latex_fig_dir):
        plt.savefig(os.path.join(latex_fig_dir, "football_analysis_results.png"), dpi=300, bbox_inches='tight')
        plt.savefig(os.path.join(latex_fig_dir, "football_analysis_results.pdf"), bbox_inches='tight')
        
    plt.close()
    print("Graphs saved successfully to results/ and LaTeX figures/ as PNG and PDF.")

if __name__ == "__main__":
    run_analysis(num_episodes=10)
