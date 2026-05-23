import os
import supersuit as ss
from stable_baselines3 import PPO
from stable_baselines3.ppo import MlpPolicy
from env_setup import create_soccer_env

def train_agents(total_timesteps=100000, save_path="../models/ppo_soccer_2v2"):
    """
    Trains the agents using Proximal Policy Optimization (PPO).
    We use parameter sharing (all agents learn from the same policy network).
    """
    # Create the base PettingZoo parallel environment
    env = create_soccer_env(render_mode=None)
    
    # PettingZoo environments need to be converted to vector environments for Stable Baselines 3.
    # ss.pettingzoo_env_to_vec_env_v1 converts the parallel env into a VecEnv.
    env = ss.pettingzoo_env_to_vec_env_v1(env)
    
    # We concatenate the environments to share the policy among agents.
    # num_vec_envs=1 means we use 1 environment instance (you can increase this in Colab if using many CPUs).
    env = ss.concat_vec_envs_v1(env, num_vec_envs=1, num_cpus=1, base_class='stable_baselines3')
    
    print("Environment wrapped for Stable Baselines 3. Starting training...")
    
    # Ensure the models directory exists
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    # Initialize PPO model
    # MuJoCo soccer returns dictionary observations, so we must use MultiInputPolicy
    model = PPO(
        "MultiInputPolicy",
        env,
        verbose=1,
        learning_rate=3e-4,
        batch_size=256,
        tensorboard_log="../logs/ppo_soccer_tensorboard/"
    )
    
    # Train the model
    print(f"Training for {total_timesteps} timesteps...")
    model.learn(total_timesteps=total_timesteps)
    
    # Save the model
    model.save(save_path)
    print(f"Model saved successfully to {save_path}.zip")
    
if __name__ == "__main__":
    # For local testing, we run a very short training loop (e.g., 2048 timesteps)
    # We do this because running on CPU is slow.
    # In Colab (with GPU), you will increase this to 1,000,000+ timesteps.
    print("--- Local Testing Mode ---")
    # Using 2,000,000 steps for better training on the local GTX 1650 GPU
    train_agents(total_timesteps=2000000, save_path="../models/ppo_soccer_2v2_local")
