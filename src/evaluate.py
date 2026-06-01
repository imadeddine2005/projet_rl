import os
import time
import supersuit as ss
from stable_baselines3 import PPO
from env_setup import create_soccer_env

# Ensure it finds the model regardless of where the script is run from
DEFAULT_MODEL = os.path.join(os.path.dirname(__file__), "..", "models", "ppo_soccer_2v2_colab")

def evaluate_agents(model_path=DEFAULT_MODEL):
    """
    Loads a trained model and evaluates it in the environment with rendering enabled.
    """
    print(f"Loading model from {model_path}...")
    try:
        model = PPO.load(model_path)
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    # Create the environment with rendering enabled
    print("Initializing environment for evaluation...")
    env = create_soccer_env(render_mode="human")
    
    # We must wrap the env identically to how it was wrapped during training
    env = ss.pettingzoo_env_to_vec_env_v1(env)
    env = ss.concat_vec_envs_v1(env, num_vec_envs=1, num_cpus=1, base_class='stable_baselines3')
    
    obs = env.reset()
    
    print("Starting evaluation. Watch the MuJoCo window!")
    for i in range(2000):
        # Predict the action using the trained model
        action, _states = model.predict(obs, deterministic=True)
        
        # Step the environment
        obs, rewards, dones, infos = env.step(action)
        
        # Add a small delay so we can see the physics smoothly
        time.sleep(0.02)
        
        # If the episode is done (e.g., someone scored or time ran out)
        if dones.any():
            print("Episode finished. Resetting...")
            obs = env.reset()

    env.close()
    print("Evaluation finished.")

if __name__ == "__main__":
    evaluate_agents()
