from __future__ import annotations

import argparse
import torch
import json
from pathlib import Path
from tqdm import tqdm

from common.data import load_yaml, read_jsonl, prompt_messages, repo_path
from common.models import load_policy, load_reward_model, load_tokenizer, reference_mode
from common.generation import batch_generate, score_reward_pairs, response_token_logprobs


def load_evaluation_bundle(config_path: str, adapter: str):
    cfg = load_yaml(config_path)
    return {
        "cfg": cfg,
        "rows": read_jsonl(cfg["paths"]["rl_prompt_eval"]),
        "tokenizer": load_tokenizer(cfg["base_model"]),
        "policy": load_policy(cfg, adapter_path=adapter, trainable=False),
        "reward": load_reward_model(cfg),
    }


def run_evaluation(config_path: str, adapter: str, run_name: str):
    bundle = load_evaluation_bundle(config_path, adapter)
    cfg = bundle["cfg"]
    policy = bundle["policy"]
    tokenizer = bundle["tokenizer"]
    reward_model, reward_tokenizer = bundle["reward"]
    eval_rows = bundle["rows"]
    
    device = next(policy.parameters()).device
    
    # Actually we just use reference_mode(policy) to get ref_logprobs
    # so we don't need a separate ref_policy!
    
    max_seq = int(cfg["max_prompt_length"])
    max_gen = int(cfg.get("eval_max_response_length", 512))
    batch_size = int(cfg.get("batch_size", 16))
    gen_cfg = cfg.get("generation", {})
    
    def _make_batches(rows, bs):
        for i in range(0, len(rows), bs):
            yield rows[i:i+bs]
            
    print(f"\n[Evaluate] '{run_name}' | {len(eval_rows)} eval prompts")
    
    all_rewards = []
    all_kls = []
    all_entropies = []
    all_lengths = []
    qualitative_examples = []
    
    for batch_rows in tqdm(list(_make_batches(eval_rows, batch_size))):
        prompts = [prompt_messages(r) for r in batch_rows]
        
        gen_out = batch_generate(
            policy, tokenizer, prompts,
            max_prompt_length=max_seq,
            max_new_tokens=max_gen,
            temperature=gen_cfg.get("temperature", 0.7),
            top_p=gen_cfg.get("top_p", 0.9),
            do_sample=gen_cfg.get("do_sample", True),
        )
        
        sequences = gen_out["sequences"]
        attention_mask = gen_out["attention_mask"]
        prompt_width = gen_out["prompt_width"]
        response_ids = gen_out["response_ids"]
        response_mask = gen_out["response_mask"]
        responses = gen_out["responses"]
        lengths = gen_out["response_lengths"]
        
        with torch.no_grad():
            task_rewards = score_reward_pairs(reward_model, reward_tokenizer, prompts, responses)
            
            old_logp, new_logits = response_token_logprobs(policy, sequences, attention_mask, prompt_width, response_ids)
            with reference_mode(policy):
                ref_logp, _ = response_token_logprobs(policy, sequences, attention_mask, prompt_width, response_ids)
                
            kl = ((old_logp - ref_logp) * response_mask).sum(dim=-1)
            
            probs = torch.softmax(new_logits, dim=-1)
            logprobs = torch.log_softmax(new_logits, dim=-1)
            entropy = -(probs * logprobs).sum(dim=-1)
            entropy_per_seq = (entropy * response_mask).sum(dim=-1) / response_mask.sum(dim=-1).clamp_min(1.0)
            
        all_rewards.extend(task_rewards.cpu().tolist())
        all_kls.extend(kl.cpu().tolist())
        all_entropies.extend(entropy_per_seq.cpu().tolist())
        all_lengths.extend(lengths)
        
        # Save first batch qualitative
        if not qualitative_examples:
            for i in range(min(5, len(responses))):
                qualitative_examples.append({
                    "prompt": prompts[i],
                    "response": responses[i],
                    "reward": task_rewards[i].item(),
                    "kl": kl[i].item(),
                    "entropy": entropy_per_seq[i].item()
                })
                
    mean_reward = sum(all_rewards) / len(all_rewards)
    mean_kl = sum(all_kls) / len(all_kls)
    mean_entropy = sum(all_entropies) / len(all_entropies)
    mean_length = sum(all_lengths) / len(all_lengths)
    
    print(f"  Reward: {mean_reward:.4f} | KL: {mean_kl:.4f} | Entropy: {mean_entropy:.4f} | Length: {mean_length:.1f}")
    
    out_dir = repo_path(cfg["results_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    
    metrics = {
        "run_name": run_name,
        "reward": mean_reward,
        "kl": mean_kl,
        "entropy": mean_entropy,
        "length": mean_length
    }
    with open(out_dir / f"{run_name}_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
        
    with open(out_dir / f"{run_name}_qualitative.jsonl", "w") as f:
        for ex in qualitative_examples:
            f.write(json.dumps(ex) + "\n")
            
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/ppo.yaml")
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--name", default="standard")
    args = ap.parse_args()
    run_evaluation(args.config, args.adapter, args.name)

if __name__ == "__main__":
    main()
