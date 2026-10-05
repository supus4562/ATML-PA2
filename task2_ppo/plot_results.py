import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import argparse

# NeurIPS formatting settings
plt.rcParams.update({
    "font.size": 9,
    "axes.titlesize": 10,
    "axes.labelsize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "figure.dpi": 200,
    "lines.linewidth": 1.5,
    "lines.markersize": 4
})

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", default="results/task2_ppo")
    args = ap.parse_args()
    
    res_dir = Path(args.results_dir)
    plots_dir = res_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Standard Continuation Trajectories (Compact 2x4 Grid)
    std_log = res_dir / "ppo_train_log_standard.jsonl"
    if std_log.exists():
        updates = []
        data = {m: [] for m in ["reward", "kl", "policy_loss", "value_loss", "entropy", "clip_fraction", "grad_norm_policy", "response_length"]}
        with open(std_log, "r") as f:
            for line in f:
                d = json.loads(line)
                updates.append(d["update"])
                for m in data:
                    data[m].append(d[m])
                    
        fig, axes = plt.subplots(2, 4, figsize=(10, 4.5))
        axes = axes.flatten()
        for i, (m, vals) in enumerate(data.items()):
            axes[i].plot(updates, vals, marker='o', color='teal')
            axes[i].set_title(m.replace('_', ' ').title())
            axes[i].grid(True, linestyle='--', alpha=0.6)
            if i >= 4:
                axes[i].set_xlabel("Update")
        
        plt.tight_layout(pad=0.5)
        plt.savefig(plots_dir / "standard_trajectories_grid.pdf", bbox_inches='tight')
        plt.savefig(plots_dir / "standard_trajectories_grid.png", bbox_inches='tight')
        plt.close()

    # 2. Ablation Studies (Compact 1x3 Grid)
    eps_vals = [0.05, 0.20, 0.50]
    eps_rewards = []
    for eps in eps_vals:
        p = res_dir / f"clip_{eps}_metrics.json"
        if p.exists():
            with open(p) as f:
                eps_rewards.append(json.load(f)["reward"])
                
    beta_vals = [0.0, 0.10, 0.20]
    beta_rewards, beta_kls = [], []
    for beta in beta_vals:
        p = res_dir / f"beta_{beta}_metrics.json"
        if p.exists():
            with open(p) as f:
                d = json.load(f)
                beta_rewards.append(d["reward"])
                beta_kls.append(d["kl"])

    if eps_rewards and beta_rewards:
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(10, 2.5))
        
        # Clip Epsilon vs Reward
        ax1.plot(eps_vals, eps_rewards, marker='s', color='blue')
        ax1.set_xlabel(r"Clip $\epsilon$")
        ax1.set_ylabel("Reward")
        ax1.set_title("Reward vs Clipping")
        ax1.grid(True, linestyle='--', alpha=0.6)
        
        # Beta vs Reward
        ax2.plot(beta_vals, beta_rewards, marker='^', color='red')
        ax2.set_xlabel(r"KL $\beta$")
        ax2.set_title("Reward vs KL Penalty")
        ax2.grid(True, linestyle='--', alpha=0.6)
        
        # Beta vs KL
        ax3.plot(beta_vals, beta_kls, marker='o', color='purple')
        ax3.set_xlabel(r"KL $\beta$")
        ax3.set_ylabel("KL Divergence")
        ax3.set_title("KL Drift vs KL Penalty")
        ax3.grid(True, linestyle='--', alpha=0.6)
        
        plt.tight_layout(pad=0.5)
        plt.savefig(plots_dir / "ablation_studies_compact.pdf", bbox_inches='tight')
        plt.savefig(plots_dir / "ablation_studies_compact.png", bbox_inches='tight')
        plt.close()

    # 3. Explicit Stability Statistic for Clipping Study
    stability_stats = {}
    for eps in eps_vals:
        log_file = res_dir / f"ppo_train_log_clip_{eps}.jsonl"
        if log_file.exists():
            losses, grad_norms = [], []
            with open(log_file, "r") as f:
                for line in f:
                    d = json.loads(line)
                    losses.append(d["policy_loss"])
                    grad_norms.append(d["grad_norm_policy"])
            
            stability_stats[f"clip_{eps}"] = {
                "policy_loss_std": float(np.std(losses)) if len(losses) > 0 else 0,
                "max_grad_norm": float(np.max(grad_norms)) if len(grad_norms) > 0 else 0
            }
            
    if stability_stats:
        with open(res_dir / "clipping_stability_stats.json", "w") as f:
            json.dump(stability_stats, f, indent=2)

if __name__ == "__main__":
    main()
