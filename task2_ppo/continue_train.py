from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.optim import AdamW

from common.data import load_yaml, prompt_messages, read_jsonl, repo_path
from common.logging_utils import set_seed, append_jsonl
from common.models import (
    load_policy,
    load_reward_model,
    load_tokenizer,
    load_value_model,
    trainable_parameters,
    value_parameter_groups,
    reference_mode,
    token_values
)
from common.generation import (
    batch_generate,
    response_sequence_logprobs,
    score_reward_pairs,
    response_token_logprobs,

)
from task2_ppo.ppo import (
    compute_gae,
    shaped_rewards,
    ppo_policy_loss,
    value_mse_loss,
    normalize_advantages
)
import time
import math


def prepare_ppo_continuation(config_path: str):
    cfg = load_yaml(config_path)
    set_seed(int(cfg["seed"]))

    tokenizer = load_tokenizer(cfg["base_model"])
    policy = load_policy(
        cfg,
        adapter_path=cfg["paths"]["ppo_midpoint_policy"],
        trainable=True,
    )
    value_model = load_value_model(
        cfg,
        cfg["paths"]["ppo_midpoint_value"],
        train_mode=cfg.get("value_train_mode", "head_only"),
    )
    reward_model, reward_tokenizer = load_reward_model(cfg)
    prompts = read_jsonl(cfg["paths"]["rl_prompt_train"])

    policy_optimizer = AdamW(
        trainable_parameters(policy),
        lr=float(cfg["policy_learning_rate"]),
    )
    value_optimizer = AdamW(
        value_parameter_groups(
            value_model,
            lora_lr=float(cfg["value_lora_learning_rate"]),
            head_lr=float(cfg["value_head_learning_rate"]),
        ),
        weight_decay=0.0,
    )

    return {
        "cfg": cfg,
        "tokenizer": tokenizer,
        "policy": policy,
        "value_model": value_model,
        "reward_model": reward_model,
        "reward_tokenizer": reward_tokenizer,
        "prompt_rows": prompts,
        "policy_optimizer": policy_optimizer,
        "value_optimizer": value_optimizer,
    }

def run_ppo(config_path: str, output: str | None = None, updates: int | None = None, clip_epsilon: float | None = None, kl_beta: float | None = None, run_name: str = "standard"):
    bundle = prepare_ppo_continuation(config_path)
    cfg = bundle["cfg"]
    if updates is not None:
        cfg["updates"] = int(updates)
    if clip_epsilon is not None:
        cfg["clip_epsilon"] = float(clip_epsilon)
    if kl_beta is not None:
        cfg["kl_beta"] = float(kl_beta)
        
    out = repo_path(output or cfg["output"])
    out.parent.mkdir(parents=True, exist_ok=True)
    
    log_path = repo_path(cfg["results_dir"]) / f"ppo_train_log_{run_name}.jsonl"
    
    device = next(bundle["policy"].parameters()).device
    
    updates_count = int(cfg["updates"])
    ppo_epochs = int(cfg["ppo_epochs"])
    prompts_per_update = int(cfg["prompts_per_update"]) # Actually batch size for rollouts
    max_prompt_len = int(cfg["max_prompt_length"])
    max_resp_len = int(cfg["max_response_length"])
    kl_beta_val = float(cfg["kl_beta"])
    clip_eps = float(cfg["clip_epsilon"])
    gamma = float(cfg.get("gamma", 1.0))
    gae_lam = float(cfg.get("gae_lambda", 0.95))
    val_coef = float(cfg.get("value_coef", 0.5))
    max_grad_norm = float(cfg.get("max_grad_norm", 1.0))
    gen_cfg = cfg.get("generation", {})
    
    policy = bundle["policy"]
    value_model = bundle["value_model"]
    tokenizer = bundle["tokenizer"]
    reward_model = bundle["reward_model"]
    reward_tokenizer = bundle["reward_tokenizer"]
    prompt_rows = bundle["prompt_rows"]
    
    print(f"Starting PPO run '{run_name}' for {updates_count} updates.")
    print(f"KL Beta: {kl_beta_val}, Clip Eps: {clip_eps}")
    
    from tqdm import tqdm
    pbar = tqdm(range(updates_count), desc=f"PPO [{run_name}]")
    
    for update_idx in pbar:
        t0 = time.time()
        
        # 1. Rollout Generation
        policy.eval()
        value_model.eval()
        
        # Sample prompts
        start_idx = (update_idx * prompts_per_update) % len(prompt_rows)
        batch_prompts = prompt_rows[start_idx : start_idx + prompts_per_update]
        
        prompts_msgs = [prompt_messages(p) for p in batch_prompts]
        
        gen_out = batch_generate(
            policy, tokenizer, prompts_msgs,
            max_prompt_length=max_prompt_len,
            max_new_tokens=max_resp_len,
            temperature=gen_cfg.get("temperature", 0.7),
            top_p=gen_cfg.get("top_p", 0.9),
            do_sample=gen_cfg.get("do_sample", True),
        )
        
        sequences = gen_out["sequences"].clone()
        attention_mask = gen_out["attention_mask"].clone()
        prompt_width = gen_out["prompt_width"]
        response_ids = gen_out["response_ids"].clone()
        response_mask = gen_out["response_mask"].clone()
        responses = gen_out["responses"]
        
        with torch.no_grad():
            task_rewards = score_reward_pairs(reward_model, reward_tokenizer, prompts_msgs, responses)
            
            # Policy logprobs (old)
            old_logp, _ = response_token_logprobs(policy, sequences, attention_mask, prompt_width, response_ids)
            
            # Reference logprobs
            with reference_mode(policy):
                ref_logp, _ = response_token_logprobs(policy, sequences, attention_mask, prompt_width, response_ids)
                
            # Values
            all_values = token_values(value_model, sequences, attention_mask)
            # Match value shape to response_ids
            # The value at t predicts the return from t onwards. 
            # We want the value of the state *before* generating response_ids[:, t].
            # That state is represented by hidden state at prompt_width - 1 + t.
            values = all_values[:, prompt_width - 1 : prompt_width - 1 + response_ids.shape[1]]
            
            # GAE and Returns
            rewards_shaped = shaped_rewards(task_rewards, old_logp, ref_logp, response_mask, kl_beta_val)
            adv, returns = compute_gae(rewards_shaped, values, response_mask, gamma=gamma, lam=gae_lam)
            adv_norm = normalize_advantages(adv, response_mask)
            
            # Diagnostics
            kl_div = ((old_logp - ref_logp) * response_mask).sum(dim=-1).mean().item()
            mean_reward = task_rewards.mean().item()
            mean_length = sum(gen_out["response_lengths"]) / len(gen_out["response_lengths"])

        # 2. PPO Optimization
        policy.train()
        value_model.train()
        
        total_policy_loss = 0.0
        total_value_loss = 0.0
        total_entropy = 0.0
        total_clip_frac = 0.0
        
        for epoch in range(ppo_epochs):
            bundle["policy_optimizer"].zero_grad()
            bundle["value_optimizer"].zero_grad()
            
            # Recompute current logprobs and values
            new_logp, new_logits = response_token_logprobs(policy, sequences, attention_mask, prompt_width, response_ids)
            
            curr_all_values = token_values(value_model, sequences, attention_mask)
            curr_values = curr_all_values[:, prompt_width - 1 : prompt_width - 1 + response_ids.shape[1]]
            
            p_loss, ratio, clip_frac = ppo_policy_loss(new_logp, old_logp, adv_norm, response_mask, eps=clip_eps)
            v_loss = value_mse_loss(curr_values, returns, response_mask)
            
            loss = p_loss + val_coef * v_loss
            loss.backward()
            
            p_grad_norm = torch.nn.utils.clip_grad_norm_(trainable_parameters(policy), max_grad_norm).item()
            v_grad_norm = torch.nn.utils.clip_grad_norm_(trainable_parameters(value_model), max_grad_norm).item()
            
            bundle["policy_optimizer"].step()
            bundle["value_optimizer"].step()
            
            total_policy_loss += p_loss.item()
            total_value_loss += v_loss.item()
            total_clip_frac += clip_frac.item()
            
            # Entropy
            probs = torch.softmax(new_logits, dim=-1)
            logprobs = torch.log_softmax(new_logits, dim=-1)
            entropy = -(probs * logprobs).sum(dim=-1)
            total_entropy += (entropy * response_mask).sum().item() / max(1.0, response_mask.sum().item())
            
        peak_vram = torch.cuda.max_memory_allocated() / (1024**3)
        
        log_data = {
            "update": update_idx,
            "reward": mean_reward,
            "kl": kl_div,
            "policy_loss": total_policy_loss / ppo_epochs,
            "value_loss": total_value_loss / ppo_epochs,
            "entropy": total_entropy / ppo_epochs,
            "clip_fraction": total_clip_frac / ppo_epochs,
            "grad_norm_policy": p_grad_norm,
            "grad_norm_value": v_grad_norm,
            "response_length": mean_length,
            "peak_vram_gb": peak_vram,
            "elapsed": time.time() - t0
        }
        append_jsonl(log_path, log_data)
        pbar.set_postfix(rew=f"{mean_reward:.2f}", kl=f"{kl_div:.2f}", p_loss=f"{log_data['policy_loss']:.3f}", v_loss=f"{log_data['value_loss']:.3f}")

    # Save models
    policy.save_pretrained(out)
    value_model.save_pretrained(str(out) + "_value")
    print(f"Checkpoint saved -> {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/ppo.yaml")
    ap.add_argument("--output")
    ap.add_argument("--updates", type=int)
    ap.add_argument("--clip-epsilon", type=float)
    ap.add_argument("--kl-beta", type=float)
    ap.add_argument("--run-name", default="standard")
    args = ap.parse_args()
    run_ppo(args.config, args.output, args.updates, args.clip_epsilon, args.kl_beta, args.run_name)


if __name__ == "__main__":
    main()
