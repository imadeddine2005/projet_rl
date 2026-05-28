import shimmy
import numpy as np
from shimmy import DmControlMultiAgentCompatibilityV0
from pettingzoo.utils.wrappers.base_parallel import BaseParallelWrapper

class SoccerRewardShapingWrapper(BaseParallelWrapper):
    """
    Custom wrapper to provide aggressive shaped rewards.
    This forces the agents to run towards the ball by applying a distance penalty.
    """
    def __init__(self, env):
        super().__init__(env)
        
    def __reduce__(self):
        # Allow cloudpickle to serialize this wrapper safely without EzPickle issues
        return (SoccerRewardShapingWrapper, (self.env,))

    def step(self, actions):
        obs, rewards, terminations, truncations, infos = self.env.step(actions)
        
        for agent in self.possible_agents:
            if not terminations[agent] and not truncations[agent]:
                # Extract statistics
                vel_to_ball = obs[agent]['stats_vel_to_ball'][0]
                vel_ball_to_goal = obs[agent]['stats_vel_ball_to_goal'][0]
                
                # Calculate distance to ball (using X and Y coordinates)
                ball_pos = obs[agent]['ball_ego_position']
                dist_to_ball = np.linalg.norm(ball_pos[:2])
                
                # Aggressive Reward Shaping
                # 1. Distance penalty: hurts the agent if it stays far from the ball
                dist_penalty = -0.01 * dist_to_ball
                
                # 2. Velocity reward: huge boost for moving towards the ball
                vel_reward = 0.1 * vel_to_ball
                
                # 3. Goal reward: huge boost for ball moving to opponent goal
                goal_reward = 0.5 * vel_ball_to_goal
                
                rewards[agent] += dist_penalty + vel_reward + goal_reward
                
        return obs, rewards, terminations, truncations, infos

def create_soccer_env(render_mode=None):
    """
    Creates and returns a MuJoCo Soccer 2v2 multi-agent environment using PettingZoo API.
    """
    print("Initializing MuJoCo Soccer 2-vs-2 Environment...")
    
    # We use the DmControlMultiAgentCompatibilityV0 wrapper provided by Shimmy
    env = DmControlMultiAgentCompatibilityV0(
        team_size=2,
        render_mode=render_mode
    )
    
    # Fix for pickling the inner environment (required by SuperSuit/Stable-Baselines3)
    env._ezpickle_args = ()
    env._ezpickle_kwargs = {"team_size": 2, "render_mode": render_mode}
    
    # Apply our custom reward shaping wrapper
    env = SoccerRewardShapingWrapper(env)
    
    return env

if __name__ == "__main__":
    # Test the environment creation
    env = create_soccer_env(render_mode=None)
    env.reset()
    print("Environment successfully created and reset!")
    print(f"Agents in the environment: {env.agents}")
    
    actions = {agent: env.action_space(agent).sample() for agent in env.agents}
    observations, rewards, terminations, truncations, infos = env.step(actions)
    print(f"Rewards after 1 step (should include shaped reward if any): {rewards}")
    env.close()
