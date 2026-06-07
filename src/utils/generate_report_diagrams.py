import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# Ensure output encoding is UTF-8 on Windows
if sys.platform.startswith('win'):
    sys.stdout.reconfigure(encoding='utf-8')

# Output directory for figures (relative to this script's location)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if os.path.exists(os.path.join(SCRIPT_DIR, "..", "..", "rapport", "IA_Template_Report_main_1__4_")):
    OUTPUT_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "..", "rapport", "IA_Template_Report_main_1__4_", "figures"))
else:
    OUTPUT_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "..", "rapport", "figures"))
os.makedirs(OUTPUT_DIR, exist_ok=True)

def generate_system_architecture():
    """
    Generates a professional system flow/architecture diagram for the MARL pipeline.
    """
    print("Generating system architecture diagram...")
    fig, ax = plt.subplots(figsize=(10, 5), dpi=300)
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 5)
    ax.axis('off')

    # Styles
    box_style = dict(boxstyle="round,pad=0.5", facecolor="#E6F2FF", edgecolor="#0066CC", lw=1.5)
    wrapper_style = dict(boxstyle="round,pad=0.5", facecolor="#FFF0E6", edgecolor="#FF6600", lw=1.5)
    model_style = dict(boxstyle="round,pad=0.5", facecolor="#E6FFE6", edgecolor="#009933", lw=1.5)
    
    # Text styles
    title_style = dict(fontsize=11, fontweight='bold', ha='center', va='center')
    desc_style = dict(fontsize=9, color="#333333", ha='center', va='center')

    # Draw components
    # 1. Simulator
    ax.text(1.5, 4, "MuJoCo 2v2 Simulator\n(Continuous Physics)", title_style, bbox=box_style)
    ax.text(1.5, 3.2, "Outputs raw agent coordinates,\nvelocities, and ball position", desc_style)

    # 2. Wrapper
    ax.text(5.0, 4, "SoccerRewardShapingWrapper\n(Custom API Wrapper)", title_style, bbox=wrapper_style)
    ax.text(5.0, 3.2, "Applies distance penalty,\nvelocity and goal shaping", desc_style)

    # 3. SuperSuit Vectorization
    ax.text(8.5, 4, "SuperSuit Vectorization\n(Batching & Stacking)", title_style, bbox=box_style)
    ax.text(8.5, 3.2, "Stacks observations to enable\nparameter sharing", desc_style)

    # 4. Neural Network Policy
    ax.text(5.0, 1.8, "Shared PPO Policy (Actor-Critic)\n(Stable-Baselines3)", title_style, bbox=model_style)
    ax.text(5.0, 1.0, "Two-layer MLP (64 units, tanh)\nOutputs continuous joint torques", desc_style)

    # Connect components with arrows
    arrow = dict(arrowstyle="->", lw=2, color="#555555")
    
    # Horizontal flows
    ax.annotate("", xy=(3.6, 4.0), xytext=(2.9, 4.0), arrowprops=arrow)
    ax.annotate("", xy=(7.1, 4.0), xytext=(6.4, 4.0), arrowprops=arrow)
    
    # Verticals/Feedbacks
    ax.annotate("", xy=(5.0, 2.3), xytext=(8.5, 3.0), 
                arrowprops=dict(arrowstyle="->", lw=2, color="#555555", connectionstyle="angle,angleA=0,angleB=90,rad=10"))
    ax.annotate("", xy=(1.5, 3.0), xytext=(5.0, 1.0), 
                arrowprops=dict(arrowstyle="->", lw=2, color="#555555", connectionstyle="angle,angleA=0,angleB=-90,rad=10"))
    
    plt.tight_layout()
    pdf_path = os.path.join(OUTPUT_DIR, "system_architecture.pdf")
    plt.savefig(pdf_path, format="pdf", bbox_inches='tight')
    plt.close()
    print(f"System architecture diagram saved to: {pdf_path}")

def generate_pitch_coordinates():
    """
    Generates a professional 2D soccer pitch coordinate and observation diagram.
    """
    print("Generating pitch coordinates and observations diagram...")
    fig, ax = plt.subplots(figsize=(8, 6), dpi=300)
    
    # Pitch boundaries: X from -12 to 12, Y from -8 to 8
    ax.set_xlim(-13, 13)
    ax.set_ylim(-9, 9)
    ax.set_aspect('equal')
    
    # Draw green grass
    ax.add_patch(patches.Rectangle((-12.5, -8.5), 25, 17, facecolor="#F4FAF4", edgecolor="#D0E0D0", zorder=1))
    
    # Draw Pitch Markings (White lines)
    line_color = "#99BBA9"
    ax.add_patch(patches.Rectangle((-12, -8), 24, 16, fill=False, edgecolor=line_color, lw=2, zorder=2))
    ax.plot([0, 0], [-8, 8], color=line_color, lw=2, zorder=2) # Halfway line
    ax.add_patch(patches.Circle((0, 0), 3.0, fill=False, edgecolor=line_color, lw=2, zorder=2)) # Center circle
    
    # Draw Goals
    ax.add_patch(patches.Rectangle((-12.5, -2), 0.5, 4, facecolor="#FFE6E6", edgecolor="#CC0000", lw=1.5, zorder=2)) # Red Goal
    ax.add_patch(patches.Rectangle((12, -2), 0.5, 4, facecolor="#E6F2FF", edgecolor="#0066CC", lw=1.5, zorder=2)) # Blue Goal
    
    # Axes lines (representing coordinate system)
    ax.axhline(0, color="#888888", linestyle="--", lw=1, zorder=3)
    ax.axvline(0, color="#888888", linestyle="--", lw=1, zorder=3)
    ax.text(12.5, 0.3, "X-axis", fontsize=9, color="#555555", ha="right")
    ax.text(0.3, 8.3, "Y-axis", fontsize=9, color="#555555", va="top")
    ax.text(-12.2, 2.3, "Red Goal (X = -12.0)", fontsize=8, color="#990000", ha="left")
    ax.text(12.2, 2.3, "Blue Goal (X = +12.0)", fontsize=8, color="#000099", ha="right")

    # Draw Agents and Ball
    # Agent 0 (Red Attacker) at (-4, -2)
    agent_pos = np.array([-5.0, -2.0])
    ax.add_patch(patches.Circle(agent_pos, 0.4, facecolor="#FF4D4D", edgecolor="#990000", lw=1.5, zorder=5))
    ax.text(agent_pos[0], agent_pos[1] - 0.9, "Agent (Red)", fontsize=9, fontweight='bold', color="#990000", ha='center')

    # Ball at (2, 3)
    ball_pos = np.array([2.0, 3.0])
    ax.add_patch(patches.Circle(ball_pos, 0.25, facecolor="#FFFFFF", edgecolor="#000000", lw=1.5, zorder=6))
    ax.text(ball_pos[0], ball_pos[1] + 0.5, "Ball", fontsize=9, fontweight='bold', color="black", ha='center')
    
    # Draw Vectors and Labels
    # Vector: Agent to Ball (b_ego)
    ax.annotate("", xy=ball_pos, xytext=agent_pos,
                arrowprops=dict(arrowstyle="->", lw=2, color="#FF9900", ls="-"))
    # Label b_ego
    mid_point = (agent_pos + ball_pos) / 2
    ax.text(mid_point[0] - 0.5, mid_point[1] + 0.3, r"$\mathbf{b}_{ego}$ (Distance to ball)", 
            color="#CC6600", fontsize=9, fontweight='bold', rotation=35, ha='center')

    # Vector: Agent Velocity to Ball (v_to_ball)
    vel_dir = (ball_pos - agent_pos) / np.linalg.norm(ball_pos - agent_pos)
    ax.annotate("", xy=agent_pos + vel_dir * 2.0, xytext=agent_pos,
                arrowprops=dict(arrowstyle="->", lw=2.5, color="#009933"))
    ax.text(agent_pos[0] + vel_dir[0] * 1.0 - 0.8, agent_pos[1] + vel_dir[1] * 1.0 + 0.3, 
            r"$v_{to\_ball}$", color="#006622", fontsize=10, fontweight='bold')

    # Vector: Ball Velocity to Opponent's Goal (v_ball_to_goal)
    # Opponent goal center is at (12.0, 0.0)
    goal_pos = np.array([12.0, 0.0])
    goal_dir = (goal_pos - ball_pos) / np.linalg.norm(goal_pos - ball_pos)
    ax.annotate("", xy=ball_pos + goal_dir * 3.0, xytext=ball_pos,
                arrowprops=dict(arrowstyle="->", lw=2.5, color="#0066CC", ls="-"))
    ax.text(ball_pos[0] + goal_dir[0] * 1.5, ball_pos[1] + goal_dir[1] * 1.5 - 0.7, 
            r"$v_{ball\_to\_goal}$", color="#004488", fontsize=10, fontweight='bold')

    plt.tight_layout()
    pdf_path = os.path.join(OUTPUT_DIR, "pitch_coordinates.pdf")
    plt.savefig(pdf_path, format="pdf", bbox_inches='tight')
    plt.close()
    print(f"Pitch coordinates diagram saved to: {pdf_path}")

if __name__ == "__main__":
    generate_system_architecture()
    generate_pitch_coordinates()
    print("All diagrams generated successfully!")
