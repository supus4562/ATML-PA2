from __future__ import annotations

import argparse
import torch
import json

from common.data import load_yaml, repo_path, read_jsonl, prompt_messages
from common.models import load_value_model, load_reward_model, load_tokenizer, load_policy, token_values
from common.generation import  score_reward_pairs
from task2_ppo.ppo import compute_gae, shaped_rewards, ppo_policy_loss, normalize_advantages
from common.generation import response_token_logprobs, _response_mask

def load_cached_rollouts(path):
    rows = torch.load(repo_path(path), map_location="cpu", weights_only=False)
    normalized = []
    for row in rows:
        row = dict(row)
        if "old_logprobs" not in row and "old_policy_logprobs" in row:
            row["old_logprobs"] = row["old_policy_logprobs"]
        if "ref_logprobs" not in row and "reference_logprobs" in row:
            row["ref_logprobs"] = row["reference_logprobs"]
        normalized.append(row)
    return normalized

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/ppo.yaml")
    args = ap.parse_args()
    cfg = load_yaml(args.config)
    rows = load_cached_rollouts(cfg["cached_rollouts"])
    
    prompts_data = read_jsonl(cfg["paths"]["rl_prompt_train"])
    
    tokenizer = load_tokenizer(cfg["base_model"])
    policy = load_policy(cfg, adapter_path=cfg["paths"]["ppo_midpoint_policy"], trainable=False)
    value_model = load_value_model(cfg, cfg["paths"]["ppo_midpoint_value"], train_mode="head_only")
    reward_model, reward_tokenizer = load_reward_model(cfg)
    device = next(policy.parameters()).device
    
    kl_beta = float(cfg["kl_beta"])
    clip_values = cfg["clip_values"]
    
    prompts_msgs = [prompt_messages(prompts_data[r["source_index"]]) for r in rows]
    responses = [r["response"] for r in rows]
    
    rendered = [tokenizer.apply_chat_template(p, tokenize=False, add_generation_prompt=True) for p in prompts_msgs]
    enc = tokenizer(rendered, return_tensors="pt", padding=True, truncation=True, max_length=int(cfg["max_prompt_length"]))
    enc = {k: v.to(device) for k, v in enc.items()}
    prompt_width = enc["input_ids"].shape[1]
    
    resp_ids_list = [tokenizer.encode(r, add_special_tokens=False) for r in responses]
    max_resp_len = max(len(r) for r in resp_ids_list)
    
    response_ids = torch.zeros((len(rows), max_resp_len), dtype=torch.long, device=device)
    response_mask = torch.zeros((len(rows), max_resp_len), dtype=torch.float32, device=device)
    old_logp = torch.zeros((len(rows), max_resp_len), dtype=torch.float32, device=device)
    ref_logp = torch.zeros((len(rows), max_resp_len), dtype=torch.float32, device=device)
    
    for i, (r_ids, row) in enumerate(zip(resp_ids_list, rows)):
        r_len = len(r_ids)
        response_ids[i, :r_len] = torch.tensor(r_ids, dtype=torch.long, device=device)
        response_mask[i, :r_len] = 1.0
        # Check if old_logprobs is shorter than r_len (it shouldn't be, but just in case)
        old_lp = row["old_logprobs"]
        ref_lp = row["ref_logprobs"]
        lp_len = min(r_len, len(old_lp))
        old_logp[i, :lp_len] = old_lp[:lp_len].to(device)
        ref_logp[i, :lp_len] = ref_lp[:lp_len].to(device)
        # Apply eos mask logic
        response_mask[i] = _response_mask(response_ids[i].unsqueeze(0), tokenizer.eos_token_id).squeeze(0)

    sequences = torch.cat([enc["input_ids"], response_ids], dim=1)
    attention_mask = torch.cat([enc["attention_mask"], torch.ones_like(response_ids)], dim=1)
    
    with torch.no_grad():
        task_rewards = score_reward_pairs(reward_model, reward_tokenizer, prompts_msgs, responses)
        
        all_values = token_values(value_model, sequences, attention_mask)
        values = all_values[:, prompt_width - 1 : prompt_width - 1 + max_resp_len]
        
        rewards_shaped = shaped_rewards(task_rewards, old_logp, ref_logp, response_mask, kl_beta)
        adv, returns = compute_gae(rewards_shaped, values, response_mask, gamma=cfg.get("gamma", 1.0), lam=cfg.get("gae_lambda", 0.95))
        adv_norm = normalize_advantages(adv, response_mask)
        
        new_logp, _ = response_token_logprobs(policy, sequences, attention_mask, prompt_width, response_ids)
        
    print(f"\n--- Clipping Study ---")
    results = []
    for eps in clip_values:
        p_loss, ratio, clip_frac = ppo_policy_loss(new_logp, old_logp, adv_norm, response_mask, eps=eps)
        print(f"Epsilon: {eps:.2f} -> Clipped Surrogate: {-p_loss.item():.4f}, Affected Token Fraction: {clip_frac.item():.4f}")
        results.append({
            "epsilon": eps,
            "clipped_surrogate": -p_loss.item(),
            "affected_fraction": clip_frac.item()
        })
        
    out_dir = repo_path(cfg["results_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "analyze_clipping.json", "w") as f:
        json.dump(results, f, indent=2)

if __name__ == "__main__":
    main()

    # Now run short continuations for each epsilon
    from task2_ppo.continue_train import run_ppo
    for eps in clip_values:
        run_name = f"clip_{eps}"
        print(f"\nRunning short fork for epsilon: {eps}")
        run_ppo(
            config_path=args.config,
            output=f"outputs/task2_ppo/{run_name}",
            updates=cfg["fork_updates"],
            clip_epsilon=eps,
            run_name=run_name
        )
