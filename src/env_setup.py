import shimmy
from shimmy import DmControlMultiAgentCompatibilityV0
from pettingzoo.utils.wrappers.base_parallel import BaseParallelWrapper

class SoccerRewardShapingWrapper(BaseParallelWrapper):
    """
    Custom wrapper to provide shaped rewards based on internal MuJoCo statistics.
    This encourages the agents to move towards the ball and push it towards the goal.
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
                # Extract velocity to ball and velocity of ball to goal from observation
                vel_to_ball = obs[agent]['stats_vel_to_ball'][0]
                vel_ball_to_goal = obs[agent]['stats_vel_ball_to_goal'][0]
                
                # Add tiny shaped rewards to encourage proactive behavior
                rewards[agent] += (0.001 * vel_to_ball) + (0.005 * vel_ball_to_goal)
                
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
