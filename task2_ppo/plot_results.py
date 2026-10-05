import json
from pathlib import Path
import matplotlib.pyplot as plt
import argparse

def plot_trajectory(jsonl_path, metric, out_path, title):
    updates = []
    vals = []
    with open(jsonl_path, "r") as f:
        for line in f:
            d = json.loads(line)
            updates.append(d["update"])
            vals.append(d[metric])
            
    plt.figure()
    plt.plot(updates, vals, marker='o')
    plt.title(title)
    plt.xlabel("Update")
    plt.ylabel(metric)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", default="results/task2_ppo")
    args = ap.parse_args()
    
    res_dir = Path(args.results_dir)
    plots_dir = res_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Standard Continuation Trajectories
    std_log = res_dir / "ppo_train_log_standard.jsonl"
    if std_log.exists():
        metrics = ["reward", "kl", "policy_loss", "value_loss", "entropy", "clip_fraction", "grad_norm_policy", "response_length"]
        for m in metrics:
            plot_trajectory(std_log, m, plots_dir / f"standard_trajectory_{m}.png", f"Standard PPO: {m}")
            
    # 2. Epsilon Ablation (Clipping Study)
    eps_vals = [0.05, 0.20, 0.50]
    eps_rewards = []
    eps_kls = []
    for eps in eps_vals:
        p = res_dir / f"clip_{eps}_metrics.json"
        if p.exists():
            with open(p) as f:
                d = json.load(f)
                eps_rewards.append(d["reward"])
                eps_kls.append(d["kl"])
                
    if len(eps_rewards) == len(eps_vals):
        plt.figure()
        plt.plot(eps_vals, eps_rewards, marker='o', label="Reward")
        plt.xlabel("Clip Epsilon")
        plt.ylabel("Reward")
        plt.title("Reward vs Clip Epsilon")
        plt.grid(True)
        plt.savefig(plots_dir / "clip_study_reward.png")
        plt.close()

    # 3. Beta Ablation
    beta_vals = [0.0, 0.10, 0.20]
    beta_rewards = []
    beta_kls = []
    for beta in beta_vals:
        p = res_dir / f"beta_{beta}_metrics.json"
        if p.exists():
            with open(p) as f:
                d = json.load(f)
                beta_rewards.append(d["reward"])
                beta_kls.append(d["kl"])
                
    if len(beta_rewards) == len(beta_vals):
        plt.figure()
        plt.plot(beta_vals, beta_rewards, marker='o', color='red')
        plt.xlabel("KL Beta")
        plt.ylabel("Reward")
        plt.title("Reward vs KL Beta")
        plt.grid(True)
        plt.savefig(plots_dir / "beta_study_reward.png")
        plt.close()

        plt.figure()
        plt.plot(beta_vals, beta_kls, marker='o', color='purple')
        plt.xlabel("KL Beta")
        plt.ylabel("KL Divergence")
        plt.title("KL vs KL Beta")
        plt.grid(True)
        plt.savefig(plots_dir / "beta_study_kl.png")
        plt.close()
        
    print(f"Plots saved to {plots_dir}")

    # 4. Explicit Stability Statistic for Clipping Study
    import numpy as np
    stability_stats = {}
    for eps in eps_vals:
        log_file = res_dir / f"ppo_train_log_clip_{eps}.jsonl"
        if log_file.exists():
            losses = []
            grad_norms = []
            with open(log_file, "r") as f:
                for line in f:
                    d = json.loads(line)
                    losses.append(d["policy_loss"])
                    grad_norms.append(d["grad_norm_policy"])
            
            # A good stability statistic: standard deviation of policy loss and max gradient norm
            stability_stats[f"clip_{eps}"] = {
                "policy_loss_std": float(np.std(losses)) if len(losses) > 0 else 0,
                "max_grad_norm": float(np.max(grad_norms)) if len(grad_norms) > 0 else 0
            }
            
    if stability_stats:
        with open(res_dir / "clipping_stability_stats.json", "w") as f:
            json.dump(stability_stats, f, indent=2)
        print("Saved clipping_stability_stats.json (Required for your report!)")

if __name__ == "__main__":
    main()
