import json
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
from common.data import repo_path

def plot_standard_trajectories():
    res_dir = repo_path("results/task3_grpo")
    log_file = res_dir / "grpo_train_log_standard.jsonl"
    
    if not log_file.exists():
        print(f"Skipping standard trajectories plot (file not found: {log_file})")
        return
        
    df = pd.read_json(log_file, lines=True)
    
    plt.rcParams.update({'font.size': 8})
    fig, axes = plt.subplots(2, 4, figsize=(14, 6))
    fig.subplots_adjust(hspace=0.4, wspace=0.3)
    axes = axes.flatten()
    
    metrics = [
        ("reward", "Mean Reward"),
        ("kl", "KL Divergence"),
        ("group_reward_std", "Within-Group Reward Std"),
        ("uninformative_fraction", "Uninformative Fraction"),
        ("policy_loss", "Policy Loss"),
        ("grad_norm", "Gradient Norm"),
        ("entropy", "Entropy"),
        ("length", "Mean Response Length")
    ]
    
    for ax, (col, title) in zip(axes, metrics):
        if col in df.columns:
            ax.plot(df['update'], df[col], marker='o', markersize=3, linewidth=1.5, color='#1f77b4')
            ax.set_title(title)
            ax.set_xlabel("Update Step")
            ax.grid(True, linestyle='--', alpha=0.6)
            
    out_dir = res_dir / "plots"
    out_dir.mkdir(exist_ok=True)
    plt.savefig(out_dir / "standard_trajectories_grid.pdf", bbox_inches='tight', dpi=200)
    plt.savefig(out_dir / "standard_trajectories_grid.png", bbox_inches='tight', dpi=200)
    plt.close()


def plot_normalization_study():
    res_dir = repo_path("results/task3_grpo")
    
    # Load evaluation metrics
    grpo_file = res_dir / "norm_grpo_metrics.json"
    dr_grpo_file = res_dir / "norm_dr_grpo_metrics.json"
    
    if not (grpo_file.exists() and dr_grpo_file.exists()):
        print("Skipping normalization study plot (files not found)")
        return
        
    with open(grpo_file) as f:
        grpo_metrics = json.load(f)
    with open(dr_grpo_file) as f:
        dr_grpo_metrics = json.load(f)
        
    metrics = [
        ("reward", "Reward"),
        ("kl", "KL Divergence"),
        ("length", "Length"),
        ("entropy", "Entropy")
    ]
    
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    fig.subplots_adjust(hspace=0.3, wspace=0.3)
    axes = axes.flatten()
    
    x = [0, 1]
    labels = ['Canonical GRPO', 'Dr. GRPO']
    colors = ['#1f77b4', '#ff7f0e']
    
    for ax, (metric_key, title) in zip(axes, metrics):
        vals = [grpo_metrics[metric_key], dr_grpo_metrics[metric_key]]
        ax.bar(x, vals, color=colors, width=0.5)
        ax.set_title(title)
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.grid(True, linestyle='--', alpha=0.3, axis='y')
        
        # Add value labels on top of the bars for clarity
        for i, v in enumerate(vals):
            ax.text(i, v + (max(vals)*0.02), f"{v:.4f}" if metric_key != "length" else f"{v:.1f}", 
                    ha='center', va='bottom', fontweight='bold')
    
    fig.suptitle('Length-Normalization Study (Evaluation Metrics)', fontsize=14)
    
    out_dir = res_dir / "plots"
    out_dir.mkdir(exist_ok=True)
    plt.savefig(out_dir / "normalization_comparison.pdf", bbox_inches='tight')
    plt.savefig(out_dir / "normalization_comparison.png", bbox_inches='tight')
    plt.close()


def plot_group_size_study():
    res_dir = repo_path("results/task3_grpo")
    stats_file = res_dir / "group_size_analysis.json"
    
    if not stats_file.exists():
        print("Skipping group size plot (file not found)")
        return
        
    with open(stats_file) as f:
        stats = json.load(f)
        
    k_vals = [2, 4, 8]
    inform_fracs = [stats[f"k_{k}"]["informative_fraction"] for k in k_vals]
    reward_stds = [stats[f"k_{k}"]["mean_reward_std"] for k in k_vals]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    
    ax1.plot(k_vals, inform_fracs, marker='s', color='purple', linewidth=2)
    ax1.set_title("Informative Group Fraction vs K")
    ax1.set_xlabel("Group Size K")
    ax1.set_ylabel("Fraction (Min != Max)")
    ax1.set_xticks(k_vals)
    ax1.grid(True, linestyle='--', alpha=0.6)
    
    ax2.plot(k_vals, reward_stds, marker='^', color='orange', linewidth=2)
    ax2.set_title("Mean Reward Std vs K")
    ax2.set_xlabel("Group Size K")
    ax2.set_ylabel("Std Dev")
    ax2.set_xticks(k_vals)
    ax2.grid(True, linestyle='--', alpha=0.6)
    
    out_dir = res_dir / "plots"
    out_dir.mkdir(exist_ok=True)
    plt.savefig(out_dir / "group_size_analysis.pdf", bbox_inches='tight')
    plt.savefig(out_dir / "group_size_analysis.png", bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    print("Plotting standard trajectories...")
    plot_standard_trajectories()
    print("Plotting normalization study...")
    plot_normalization_study()
    print("Plotting group size study...")
    plot_group_size_study()
    print("Done!")
