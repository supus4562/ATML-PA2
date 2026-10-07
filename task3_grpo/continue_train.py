import argparse
import time
from collections import defaultdict

import torch
from torch.optim import AdamW
from tqdm import tqdm

from common.data import load_yaml, read_jsonl, repo_path
from common.logging_utils import append_jsonl, set_seed
from common.models import load_policy, load_reward_model, load_tokenizer, trainable_parameters, reference_mode
from common.generation import batch_generate, response_token_logprobs, score_reward_pairs
from task3_grpo.grpo import group_relative_advantages, grpo_policy_loss, mask_truncated_sequences


def prepare_grpo_continuation(config_path: str):
    cfg = load_yaml(config_path)
    set_seed(int(cfg["seed"]))
    tokenizer = load_tokenizer(cfg["base_model"])
    policy = load_policy(
        cfg,
        adapter_path=cfg["paths"]["grpo_midpoint_policy"],
        trainable=True,
    )
    reward_model, reward_tokenizer = load_reward_model(cfg)
    prompts = read_jsonl(cfg["paths"]["rl_prompt_train"])
    optimizer = AdamW(trainable_parameters(policy), lr=float(cfg["learning_rate"]))
    return {
        "cfg": cfg,
        "tokenizer": tokenizer,
        "policy": policy,
        "reward_model": reward_model,
        "reward_tokenizer": reward_tokenizer,
        "prompt_rows": prompts,
        "optimizer": optimizer,
    }


def _sanitize_logp(logp):
    return logp.nan_to_num(nan=0.0, posinf=0.0, neginf=-1e4)


def run_grpo(config_path: str, output: str | None = None, updates: int | None = None, loss_type: str = "grpo", run_name: str = "standard"):
    bundle = prepare_grpo_continuation(config_path)
    cfg = bundle["cfg"]
    if updates is not None:
        cfg["updates"] = int(updates)
    
    out_dir = repo_path(output or cfg["output"])
    out_dir.mkdir(parents=True, exist_ok=True)
    
    res_dir = repo_path(cfg["results_dir"])
    res_dir.mkdir(parents=True, exist_ok=True)
    log_file = res_dir / f"grpo_train_log_{run_name}.jsonl"
    if log_file.exists():
        log_file.unlink() # Cleanup old logs to prevent zigzag plots

    tokenizer = bundle["tokenizer"]
    policy = bundle["policy"]
    reward_model = bundle["reward_model"]
    reward_tokenizer = bundle["reward_tokenizer"]
    prompts = bundle["prompt_rows"]
    optimizer = bundle["optimizer"]
    
    K = cfg["num_generations"]
    max_comp_len = cfg["max_completion_length"]
    max_prompt_len = cfg["max_prompt_length"]
    prompts_per_update = cfg["prompts_per_update"]
    epochs = cfg["policy_epochs"]
    clip_epsilon = cfg["clip_epsilon"]
    kl_beta = cfg["kl_beta"]
    max_grad_norm = cfg["max_grad_norm"]
    mask_truncated = cfg.get("mask_truncated_completions", True)
    
    print(f"Starting GRPO run '{run_name}' for {cfg['updates']} updates (loss_type={loss_type}).")
    print(f"K={K}, Prompts/Update={prompts_per_update}, Epochs={epochs}, Beta={kl_beta}, Eps={clip_epsilon}")
    
    policy.train()
    
    pbar = tqdm(total=cfg["updates"], desc=f"GRPO [{run_name}]")
    prompt_idx = 0
    
    for update in range(cfg["updates"]):
        if prompt_idx + prompts_per_update > len(prompts):
            prompt_idx = 0  # Wrap around
            
        batch_prompts = prompts[prompt_idx : prompt_idx + prompts_per_update]
        prompt_idx += prompts_per_update
        
        # 1. Generate K completions per prompt
        rollout_texts = []
        group_ids_list = []
        
        for g_id, row in enumerate(batch_prompts):
            for _ in range(K):
                rollout_texts.append(row["prompt"])
                group_ids_list.append(g_id)
                
        group_ids = torch.tensor(group_ids_list, device=policy.device)
        
        policy.eval()
        gen_out = batch_generate(
            policy,
            tokenizer,
            rollout_texts,
            max_prompt_length=max_prompt_len,
            max_new_tokens=max_comp_len,
        )
        
        sequences = gen_out["sequences"].clone()
        attention_mask = gen_out["attention_mask"].clone()
        response_ids = gen_out["response_ids"].clone()
        response_mask = gen_out["response_mask"].clone()
        prompt_width = gen_out["prompt_width"]
        
        # Determine which sequences hit max length limit
        response_lengths = response_mask.sum(dim=-1).cpu().numpy()
        truncated = [int(l) >= max_comp_len for l in response_lengths]
        
        if mask_truncated:
            response_mask = mask_truncated_sequences(response_mask, truncated)
        
        # 2. Get rewards
        rewards = score_reward_pairs(
            reward_model,
            reward_tokenizer,
            rollout_texts,
            tokenizer.batch_decode(response_ids, skip_special_tokens=True),
        ).clone().float()
        
        # Calculate within-group standard deviation (for logging)
        group_reward_stds = []
        uninformative = 0
        for g in torch.unique(group_ids):
            g_rewards = rewards[group_ids == g]
            if len(g_rewards) > 1:
                group_reward_stds.append(g_rewards.std(unbiased=False).item())
                if g_rewards.max() == g_rewards.min():
                    uninformative += 1
            else:
                group_reward_stds.append(0.0)
                
        mean_group_std = sum(group_reward_stds) / len(group_reward_stds) if group_reward_stds else 0.0
        uninformative_fraction = uninformative / len(torch.unique(group_ids))
        
        # 3. Calculate Group Relative Advantages
        advantages = group_relative_advantages(rewards, group_ids, eps=1e-6)
        
        # 4. Compute Old Logprobs and Reference Logprobs
        with torch.no_grad():
            old_logp = response_token_logprobs(policy, sequences, attention_mask, prompt_width).clone()
            old_logp = _sanitize_logp(old_logp)
            
            with reference_mode(policy):
                ref_logp = response_token_logprobs(policy, sequences, attention_mask, prompt_width).clone()
                ref_logp = _sanitize_logp(ref_logp)
                
        # 5. Optimize policy
        policy.train()
        update_metrics = defaultdict(float)
        valid_epochs = 0
        
        for ep in range(epochs):
            new_logp = response_token_logprobs(policy, sequences, attention_mask, prompt_width)
            new_logp = _sanitize_logp(new_logp)
            
            loss, metrics = grpo_policy_loss(
                new_logp,
                old_logp,
                advantages,
                response_mask,
                ref_logp,
                clip_epsilon,
                kl_beta,
                loss_type=loss_type,
                max_completion_length=max_comp_len
            )
            
            if not torch.isfinite(loss):
                print(f"\\n[WARN] NaN/Inf loss at update {update} epoch {ep} — skipping backward")
                continue
                
            optimizer.zero_grad()
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(trainable_parameters(policy), max_grad_norm)
            optimizer.step()
            
            valid_epochs += 1
            update_metrics["policy_loss"] += metrics["policy_term"].item()
            update_metrics["kl"] += metrics["sampled_kl"].item()
            update_metrics["clip_fraction"] += metrics["clip_fraction"].item()
            update_metrics["entropy"] += metrics["sample_entropy"].item()
            update_metrics["grad_norm"] += grad_norm.item()
            
        if valid_epochs > 0:
            for k in update_metrics:
                update_metrics[k] /= valid_epochs
                
        mean_reward = rewards.mean().item()
        mean_length = response_lengths.mean().item()
        
        pbar.set_postfix({
            "kl": f"{update_metrics['kl']:.3f}",
            "p_loss": f"{update_metrics['policy_loss']:.3f}",
            "rew": f"{mean_reward:.2f}",
        })
        pbar.update(1)
        
        append_jsonl(log_file, {
            "update": update,
            "reward": mean_reward,
            "kl": update_metrics["kl"],
            "policy_loss": update_metrics["policy_loss"],
            "clip_fraction": update_metrics["clip_fraction"],
            "entropy": update_metrics["entropy"],
            "grad_norm": update_metrics["grad_norm"],
            "length": mean_length,
            "group_reward_std": mean_group_std,
            "uninformative_fraction": uninformative_fraction
        })
        
    pbar.close()
    print(f"\\nCheckpoint saved -> {out_dir}")
    policy.save_pretrained(out_dir)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/grpo.yaml")
    ap.add_argument("--output")
    ap.add_argument("--updates", type=int)
    ap.add_argument("--loss-type", choices=["grpo", "dr_grpo"], default="grpo")
    ap.add_argument("--run-name", default="standard")
    args = ap.parse_args()
    run_grpo(args.config, args.output, args.updates, args.loss_type, args.run_name)


if __name__ == "__main__":
    main()
