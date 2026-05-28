"""
Training Curves Plotter (TensorBoard Log Reader)
=================================================
This script reads the TensorBoard logs generated during PPO training
and produces professional matplotlib graphs for the academic report.

The curves it plots:
  - ep_rew_mean  : Average episode reward over time (main learning indicator)
  - ep_len_mean  : Average episode length over time
  - loss/value   : Value function loss (convergence indicator)
  - loss/policy  : Policy gradient loss

Run this AFTER training. Logs are saved in: src/logs/
"""

import os
import sys
import glob
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

# ----------------------------------------------------------------
# TensorBoard log parser (uses tensorboard's event file reader)
# ----------------------------------------------------------------
def read_tensorboard_logs(log_dir):
    """
    Reads TensorBoard event files and returns a dict of {tag: [(step, value), ...]}
    """
    try:
        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    except ImportError:
        print("ERROR: tensorboard not installed. Run: pip install tensorboard")
        sys.exit(1)

    event_files = glob.glob(os.path.join(log_dir, '**', 'events.out.tfevents.*'),
                            recursive=True)
    if not event_files:
        print(f"No TensorBoard event files found in: {log_dir}")
        print("Make sure you have trained the model first (train.py).")
        sys.exit(1)

    print(f"Found {len(event_files)} event file(s).")
    all_data = {}

    for ef in event_files:
        ea = EventAccumulator(ef)
        ea.Reload()
        available_tags = ea.Tags().get('scalars', [])
        print(f"  Tags found: {available_tags}")

        for tag in available_tags:
            events = ea.Scalars(tag)
            steps  = [e.step  for e in events]
            values = [e.value for e in events]
            if tag not in all_data:
                all_data[tag] = (steps, values)

    return all_data


def smooth(values, weight=0.85):
    """Exponential moving average smoothing (like TensorBoard does)."""
    smoothed = []
    last = values[0]
    for v in values:
        smoothed_val = last * weight + v * (1 - weight)
        smoothed.append(smoothed_val)
        last = smoothed_val
    return smoothed


def plot_training_curves(log_dir=None, save_dir=None):
    """
    Main function: reads logs and generates a professional 4-panel figure.
    """
    # Default paths relative to this script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    if log_dir is None:
        log_dir  = os.path.join(script_dir, '..', 'logs')
    if save_dir is None:
        save_dir = os.path.join(script_dir, '..', '..', 'results')

    log_dir  = os.path.abspath(log_dir)
    save_dir = os.path.abspath(save_dir)
    os.makedirs(save_dir, exist_ok=True)

    print("=" * 55)
    print("  PPO Training Curves Plotter")
    print("=" * 55)
    print(f"Reading logs from: {log_dir}")

    data = read_tensorboard_logs(log_dir)

    if not data:
        print("No scalar data found in logs.")
        sys.exit(1)

    # ----------------------------------------------------------------
    # FIGURE: 4-panel professional plot
    # ----------------------------------------------------------------
    fig = plt.figure(figsize=(16, 10))
    fig.suptitle('PPO Training Curves — MuJoCo Soccer 2v2 (Multi-Agent RL)',
                 fontsize=16, fontweight='bold', y=0.98)
    gs = GridSpec(2, 2, figure=fig, hspace=0.4, wspace=0.35)

    COLOR_RAW    = '#4C72B0'
    COLOR_SMOOTH = '#DD8452'

    panels = [
        ('rollout/ep_rew_mean', 'Average Episode Reward',
         'Training Timesteps', 'Mean Reward', gs[0, 0]),
        ('rollout/ep_len_mean', 'Average Episode Length',
         'Training Timesteps', 'Mean Length (steps)', gs[0, 1]),
        ('train/value_loss',    'Value Function Loss',
         'Training Timesteps', 'Loss', gs[1, 0]),
        ('train/policy_gradient_loss', 'Policy Gradient Loss',
         'Training Timesteps', 'Loss', gs[1, 1]),
    ]

    for tag, title, xlabel, ylabel, grid_pos in panels:
        ax = fig.add_subplot(grid_pos)

        # Try the exact tag, then fuzzy match
        matched_tag = None
        if tag in data:
            matched_tag = tag
        else:
            # Try partial match
            for k in data:
                if tag.split('/')[-1] in k:
                    matched_tag = k
                    break

        if matched_tag is None:
            ax.text(0.5, 0.5, f'Tag not found:\n{tag}',
                    ha='center', va='center', transform=ax.transAxes,
                    fontsize=10, color='gray')
            ax.set_title(title, fontweight='bold', fontsize=11)
            continue

        steps, values = data[matched_tag]
        smoothed_vals = smooth(values)

        ax.plot(steps, values, alpha=0.25, color=COLOR_RAW, linewidth=0.8,
                label='Raw')
        ax.plot(steps, smoothed_vals, color=COLOR_SMOOTH, linewidth=2.0,
                label='Smoothed (EMA)')

        ax.set_title(title, fontweight='bold', fontsize=11)
        ax.set_xlabel(xlabel, fontsize=9)
        ax.set_ylabel(ylabel, fontsize=9)
        ax.legend(fontsize=8)
        ax.grid(True, linestyle='--', alpha=0.4)
        ax.xaxis.set_major_formatter(plt.FuncFormatter(
            lambda x, _: f'{x/1e6:.1f}M' if x >= 1e6 else f'{int(x/1e3)}K'))

    save_path = os.path.join(save_dir, 'ppo_training_curves.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\nTraining curves saved to: {save_path}")

    # ----------------------------------------------------------------
    # PRINT SUMMARY STATISTICS
    # ----------------------------------------------------------------
    print("\n" + "=" * 55)
    print("  TRAINING SUMMARY STATISTICS")
    print("=" * 55)
    for tag, (steps, values) in data.items():
        if values:
            print(f"  {tag}:")
            print(f"    Final value : {values[-1]:.4f}")
            print(f"    Max value   : {max(values):.4f}")
            print(f"    Min value   : {min(values):.4f}")
            print(f"    Total steps : {steps[-1]:,}")
    print("=" * 55)

    return save_path


if __name__ == "__main__":
    plot_training_curves()
