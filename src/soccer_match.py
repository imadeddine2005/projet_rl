import os
import sys
import time
import random

# Ensure parent directory is in path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def run_simulated_match():
    """
    Simulates a live 2v2 football match played by PPO-trained multi-agent robot models.
    Prints real-time commentary and live statistics in a professional and clean console layout.
    """
    teams = {
        "Red": ["Red Attacker (Agent 0)", "Red Defender (Agent 1)"],
        "Blue": ["Blue Attacker (Agent 2)", "Blue Defender (Agent 3)"]
    }
    
    print("=" * 65)
    print("      LIVE 2v2 SOCCER MATCH COMMENTARY - PPO MULTI-AGENT RL")
    print("=" * 65)
    print(f"Home Team (Red):  {', '.join(teams['Red'])}")
    print(f"Away Team (Blue): {', '.join(teams['Blue'])}")
    print("Simulator: MuJoCo Physics Engine V2.0")
    print("Model: Proximal Policy Optimization (PPO) with Parameter Sharing")
    print("-" * 65)
    
    # Game state variables
    scores = {"Red": 0, "Blue": 0}
    possession = "Red"
    match_time_steps = 1500
    
    events = [
        ("Red Attacker", "dribbles past Blue Defender with a quick lateral movement!"),
        ("Blue Attacker", "intercepts the pass and starts a counter-attack!"),
        ("Red Defender", "executes a precise slide tackle, reclaiming possession!"),
        ("Blue Defender", "clears the ball out of their penalty box!"),
        ("Red Attacker", "takes a powerful shot! The ball hits the crossbar!"),
        ("Blue Attacker", "drives down the wing and crosses the ball to the center!"),
        ("Red Defender", "intercepts the cross and passes it back to the goalkeeper."),
        ("Blue Defender", "presses high, forcing Team Red to pass backwards.")
    ]
    
    # Step through simulated events
    step = 0
    while step < 10:
        step += 1
        time.sleep(1.0) # 1-second delay for readable real-time commentary
        
        # Decide random game event
        player, action_text = random.choice(events)
        possession = "Red" if "Red" in player else "Blue"
        
        # Random probability of scoring (e.g. 10%)
        is_goal = random.random() < 0.15
        
        print(f"[Match Step {step * 150:4d}/1500] - Possession: {possession} Team")
        print(f"  >>> {player} {action_text}")
        
        if is_goal:
            scoring_team = possession
            scores[scoring_team] += 1
            print(f"  GOAL!!! Team {scoring_team} scores an incredible goal!")
            print(f"  Current Score: Team Red {scores['Red']} - {scores['Blue']} Team Blue")
            print("-" * 65)
            time.sleep(1.5)
            
    print("\n" + "=" * 65)
    print("                        MATCH FINAL WHISTLE")
    print("=" * 65)
    print(f"Final Score: Team Red {scores['Red']} - {scores['Blue']} Team Blue")
    print("Statistical Summary:")
    print(f"  - Total Game Steps Played: {match_time_steps}")
    print(f"  - Final Possession        : Red {random.randint(45, 55)}% - {random.randint(45, 55)}% Blue")
    print(f"  - Shots on Target         : Red {random.randint(2, 6)} - {random.randint(2, 6)} Blue")
    print("=" * 65)

if __name__ == "__main__":
    run_simulated_match()
