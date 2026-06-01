# Multi-Agent Deep Reinforcement Learning for Cooperative Soccer (MuJoCo)

This repository contains the source code and experimental implementations for the Deep Reinforcement Learning course project at **ENSAM Casablanca** (Filière: *Intelligence Artificielle et Génie Informatique - IAGI 2*).

## Project Overview

The goal of this project is to study, implement, and analyze multi-agent cooperative control using **Proximal Policy Optimization (PPO)** in a continuous 3D physics-based simulation environment: **MuJoCo Soccer (2v2)**. 

### Core Features
* **Multi-Agent Deep Reinforcement Learning**: Cooperative multi-agent control using policy parameter sharing to optimize team coordination.
* **Custom Reward Shaping Wrapper**: Implements aggressive distance-to-ball penalties, ball velocity rewards towards the opponent's goal, and team coordinate tracking to accelerate convergence.
* **Football Analytics Dashboard**: Tracks ball possession percentages, team spread distances, and generates 2D pitch spatial density heatmaps.
* **Live Match Commentary Demo**: Interactive match simulator displaying live commentary of game events in real-time.

---

## Authors
* **Imad Eddine Oukrati**
* **Mohamed Echchyoughi**

**Supervisor**: Prof. Hirchoua Badr

---

## Repository Structure

```text
projet_rl/
│
├── src/
│   ├── env_setup.py           # Environment creation & custom reward shaping wrapper
│   ├── train.py               # PPO multi-agent training script
│   ├── evaluate.py            # Script to load and visually evaluate trained models
│   ├── soccer_match.py        # Live console match simulator with real-time commentary
│   │
│   └── utils/
│       ├── analysis.py        # Statistics and metrics collector for model evaluations
│       ├── soccer_analytics.py # Advanced football analytics & 2D pitch heatmap visualizer
│       └── plot_training.py   # TensorBoard log visualizer & training curve plotter
│
├── models/                    # Saved models and checkpoints (.zip)
├── results/                   # Evaluation plots, training curves, and spatial heatmaps
├── rapport/                   # LaTeX source code of the academic report
├── requirements.txt           # Required Python packages
└── Colab_Training_Notebook.ipynb # Google Colab T4 GPU execution notebook
```

---

## Installation & Setup

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/<your-username>/projet_rl.git
   cd projet_rl
   ```

2. **Install Dependencies**:
   It is recommended to use a Python virtual environment (Python 3.8+):
   ```bash
   pip install -r requirements.txt
   ```
   *Note: MuJoCo and stable-baselines3 require modern compiler tools on some operating systems.*

---

## Running the Code

### 1. Advanced Football Analytics & Spatial Heatmaps
To run the evaluation simulation, track team spatial density, and save the 2D pitch heatmap to `results/soccer_pitch_heatmap.png`, run:
```bash
python src/utils/soccer_analytics.py
```

### 2. Live Match Commentary Demo
To watch a live 2v2 match with real-time text commentary of game events (dribbles, tackles, shots, and goals) directly in the console, run:
```bash
python src/soccer_match.py
```

### 3. PPO Soccer Training & 3D Evaluation
To start local multi-agent training on MuJoCo Soccer:
```bash
python src/train.py
```

To evaluate the pre-trained PPO model with 3D rendering (requires `render_mode="human"` support):
```bash
python src/evaluate.py
```

---

## Academic Report
The complete LaTeX source code of the academic report is located inside the `rapport/` directory. The report documents:
* Detailed literature review on Markov Decision Processes (MDP), policy gradient stability, and Multi-Agent RL.
* Rigorous mathematical formulation of PPO and policy parameter sharing.
* Reward shaping mathematical formulations and distance metrics.
* Complete quantitative performance evaluation and spatial coverage analysis.
