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

if __name__ == "__main__":
    main()
