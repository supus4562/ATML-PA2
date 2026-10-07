import argparse
import json
from collections import defaultdict
import numpy as np

from common.data import load_yaml, read_jsonl, repo_path


def load_k8_cache(path):
    rows = read_jsonl(path)
    by_prompt = defaultdict(list)
    for row in rows:
        by_prompt[str(row["source_index"])].append(row)
    # Instructor cache has 8 rows per prompt, one row per completion.
    bad = {pid: len(group) for pid, group in by_prompt.items() if len(group) < 8}
    if bad:
        raise ValueError(f"Expected at least K=8 cached completions per prompt; short groups: {bad}")
    for group in by_prompt.values():
        group.sort(key=lambda x: int(x.get("generation_index", 0)))
    return by_prompt


def regroup_equal_generation_budget(by_prompt, k: int):
    """Return K-sized groups while keeping total cached completions fixed."""
    groups = []
    for pid, completions in by_prompt.items():
        # Split the 8 completions into chunks of size K
        for i in range(0, len(completions), k):
            if i + k <= len(completions):
                groups.append(completions[i:i+k])
    return groups


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/grpo.yaml")
    args = ap.parse_args()
    cfg = load_yaml(args.config)
    by_prompt = load_k8_cache(cfg["group_cache"])
    print(f"Cached prompts: {len(by_prompt)}")
    print(f"Group sizes to analyze: {cfg['group_sizes']}")
    
    results = {}
    
    # Pre-calculate prompt difficulties (mean reward across all 8 completions)
    prompt_difficulties = {}
    for pid, completions in by_prompt.items():
        mean_reward = np.mean([c["reward"] for c in completions])
        prompt_difficulties[pid] = mean_reward

    for k in cfg["group_sizes"]:
        groups = regroup_equal_generation_budget(by_prompt, k)
        
        informative_count = 0
        reward_stds = []
        all_advantages = []
        
        for group in groups:
            rewards = np.array([c["reward"] for c in group])
            r_min, r_max = rewards.min(), rewards.max()
            r_std = rewards.std(ddof=0)
            reward_stds.append(r_std)
            
            if r_min != r_max:
                informative_count += 1
                adv = (rewards - rewards.mean()) / (r_std + 1e-6)
            else:
                adv = np.zeros_like(rewards)
            
            all_advantages.extend(adv.tolist())
            
        informative_fraction = informative_count / len(groups) if groups else 0
        mean_reward_std = np.mean(reward_stds) if reward_stds else 0
        relative_signal_variance = np.var(all_advantages) if all_advantages else 0
        
        print(f"\\n--- K = {k} ---")
        print(f"Informative Groups: {informative_fraction:.1%}")
        print(f"Mean Reward Std: {mean_reward_std:.4f}")
        print(f"Relative Signal Variance: {relative_signal_variance:.4f}")
        
        results[f"k_{k}"] = {
            "informative_fraction": informative_fraction,
            "mean_reward_std": float(mean_reward_std),
            "relative_signal_variance": float(relative_signal_variance)
        }
        
    out_dir = repo_path(cfg["results_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "group_size_analysis.json", "w") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
