from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from common.data import load_yaml, read_jsonl, repo_path, write_jsonl
from common.generation import batch_generate
from common.models import load_policy, load_tokenizer
from task5_feedback.rlaif import PairwiseAIJudge
from task5_feedback.rlvr import exact_reward, extract_designated_final


def policy_specs(cfg):
    return {
        "sft": None,
        "rlvr": cfg["policies"]["rlvr"],
        "rlaif": cfg["policies"]["rlaif"],
    }


def dataset_path(cfg, dataset: str):
    if dataset == "gsm":
        return cfg["paths"]["gsm_eval"]
    if dataset == "transfer":
        return cfg["paths"]["math_transfer_eval"]
    raise ValueError(dataset)


def load_math_evaluation(config_path: str, dataset: str):
    cfg = load_yaml(config_path)
    rows = read_jsonl(repo_path(dataset_path(cfg, dataset)))
    tokenizer = load_tokenizer(cfg["base_model"])
    return cfg, rows, tokenizer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/feedback.yaml")
    ap.add_argument("--dataset", choices=["gsm", "transfer"], default="gsm")
    ap.add_argument("--batch-size", type=int, default=32)
    args = ap.parse_args()
    
    cfg, rows, tokenizer = load_math_evaluation(args.config, args.dataset)
    print(f"Dataset: {args.dataset}, Rows: {len(rows)}")
    
    outdir = Path(repo_path(cfg["results_dir"])) / "task5_feedback"
    outdir.mkdir(parents=True, exist_ok=True)
    
    policies = list(policy_specs(cfg).keys())
    
    # 1. Generate responses
    generated_data = {}
    for policy in policies:
        outfile = outdir / f"responses_{policy}_{args.dataset}.jsonl"
        if not outfile.exists():
            print(f"Generating for {policy}...")
            model = load_policy(cfg, adapter_path=policy_specs(cfg)[policy], trainable=False)
            records = []
            
            # Extract messages/prompts
            if "messages" in rows[0]:
                prompts = [row["messages"] for row in rows]
            else:
                # If there is no 'messages' field, fallback to creating one
                prompts = [[{"role": "user", "content": row.get("question", row.get("prompt", ""))}] for row in rows]

            for start in range(0, len(rows), args.batch_size):
                chunk_prompts = prompts[start:start + args.batch_size]
                chunk_rows = rows[start:start + args.batch_size]
                
                gen = batch_generate(
                    model,
                    tokenizer,
                    chunk_prompts,
                    max_prompt_length=256,
                    max_new_tokens=int(cfg.get("math_max_new_tokens", 512)),
                    temperature=0.0,
                    top_p=1.0,
                    do_sample=False,
                )
                
                for row, response, n_tok in zip(chunk_rows, gen["responses"], gen["response_lengths"]):
                    rec = dict(row)
                    rec["policy"] = policy
                    rec["response"] = response
                    rec["response_tokens"] = int(n_tok)
                    records.append(rec)
            
            write_jsonl(outfile, records)
            del model
            torch.cuda.empty_cache()
            generated_data[policy] = records
        else:
            generated_data[policy] = read_jsonl(outfile)
            
    # 2. Evaluate accuracy, format, length, verifier-judge agreement
    # and pairwise win rate vs SFT.
    ai_judge = PairwiseAIJudge(cfg, cache_path=outdir / "ai_judge_cache.json")
    
    summary_stats = {}
    for policy in policies:
        print(f"\nEvaluating {policy}...")
        records = generated_data[policy]
        
        exact_correct = 0
        format_compliant = 0
        lengths = []
        
        # for verifier-judge agreement
        agree_count = 0
        agree_total = 0
        
        # vs SFT
        wins_vs_sft = 0
        ties_vs_sft = 0
        losses_vs_sft = 0
        
        for i, rec in enumerate(records):
            gold_final = rec.get("gold_final", "")
            response = rec["response"]
            lengths.append(rec["response_tokens"])
            
            # format compliance
            pred = extract_designated_final(response)
            if pred is not None:
                format_compliant += 1
                
            # exact accuracy
            rew = exact_reward(response, gold_final)
            if rew == 1.0:
                exact_correct += 1
                
            # Pairwise vs SFT
            if policy != "sft":
                sft_rec = generated_data["sft"][i]
                sft_resp = sft_rec["response"]
                problem = rec.get("question", rec.get("prompt", ""))
                
                # Pairwise comparison
                pref = ai_judge.compare(problem, response, sft_resp)
                if pref == "A":
                    wins_vs_sft += 1
                elif pref == "TIE":
                    ties_vs_sft += 1
                elif pref == "B":
                    losses_vs_sft += 1
                    
                # Verifier-judge agreement:
                # Verifier preference: 
                rew_policy = exact_reward(response, gold_final)
                rew_sft = exact_reward(sft_resp, gold_final)
                
                if rew_policy > rew_sft:
                    v_pref = "A"
                elif rew_sft > rew_policy:
                    v_pref = "B"
                else:
                    v_pref = "TIE"
                
                if v_pref == pref:
                    agree_count += 1
                agree_total += 1
                
        stats = {
            "exact_accuracy": exact_correct / len(records),
            "format_compliance": format_compliant / len(records),
            "mean_length": float(np.mean(lengths)),
        }
        
        if policy != "sft":
            total_pairs = wins_vs_sft + ties_vs_sft + losses_vs_sft
            win_rate = (wins_vs_sft + 0.5 * ties_vs_sft) / total_pairs if total_pairs > 0 else 0
            stats["rlaif_win_rate_vs_sft"] = win_rate
            stats["verifier_judge_agreement"] = agree_count / agree_total if agree_total > 0 else 0
            
        summary_stats[policy] = stats
        
        print(f"Exact accuracy: {stats['exact_accuracy']:.4f}")
        print(f"Format compliance: {stats['format_compliance']:.4f}")
        print(f"Mean response length: {stats['mean_length']:.2f}")
        if policy != "sft":
            print(f"RLAIF win rate vs SFT: {stats['rlaif_win_rate_vs_sft']:.4f}")
            print(f"Verifier-judge agreement: {stats['verifier_judge_agreement']:.4f}")

    with open(outdir / f"summary_{args.dataset}.json", "w") as f:
        json.dump(summary_stats, f, indent=2)
        
    print(f"\nSaved evaluation summary to {outdir / f'summary_{args.dataset}.json'}")

if __name__ == "__main__":
    main()
