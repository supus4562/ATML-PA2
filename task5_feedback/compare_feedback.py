from __future__ import annotations

import argparse
import json
from pathlib import Path

from common.data import load_yaml, repo_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/feedback.yaml")
    args = ap.parse_args()
    cfg = load_yaml(args.config)
    
    outdir = Path(repo_path(cfg["results_dir"])) / "task5_feedback"
    
    print("=== Task 5: RLVR vs RLAIF Comparison ===\n")
    
    # 1. In-domain comparison (GSM8K)
    gsm_file = outdir / "summary_gsm.json"
    if gsm_file.exists():
        with open(gsm_file) as f:
            gsm_stats = json.load(f)
        print("--- In-Domain (GSM8K) ---")
        for policy in ["sft", "rlvr", "rlaif"]:
            if policy not in gsm_stats: continue
            stats = gsm_stats[policy]
            print(f"[{policy.upper()}]")
            print(f"  Exact Accuracy:      {stats['exact_accuracy']:.4f}")
            print(f"  Format Compliance:   {stats['format_compliance']:.4f}")
            print(f"  Mean Length:         {stats['mean_length']:.2f}")
            if policy != "sft":
                print(f"  RLAIF Win Rate vs SFT: {stats['rlaif_win_rate_vs_sft']:.4f}")
                print(f"  Verifier-Judge Agree:  {stats['verifier_judge_agreement']:.4f}")
        print()
    else:
        print(f"Missing {gsm_file}")
        
    # 2. Controlled diagnostic study
    diag_file = outdir / "diagnostic_results.json"
    if diag_file.exists():
        with open(diag_file) as f:
            diag_stats = json.load(f)
        print("--- Controlled Diagnostic Study ---")
        for mech in ["rlvr", "rlaif"]:
            if mech not in diag_stats: continue
            stats = diag_stats[mech]
            print(f"[{mech.upper()}]")
            for k, v in stats.items():
                if isinstance(v, dict):
                    print(f"  {k}: Better={v['better_rate']:.2f}, Tie={v['tie_rate']:.2f}, Worse={v['worse_rate']:.2f}")
                else:
                    print(f"  {k}: {v:.2f}")
        print()
    else:
        print(f"Missing {diag_file}")
        
    # 3. Out-of-domain comparison (Transfer)
    transfer_file = outdir / "summary_transfer.json"
    if transfer_file.exists():
        with open(transfer_file) as f:
            transfer_stats = json.load(f)
        print("--- Out-of-Domain (SVAMP) ---")
        for policy in ["sft", "rlvr", "rlaif"]:
            if policy not in transfer_stats: continue
            stats = transfer_stats[policy]
            print(f"[{policy.upper()}]")
            print(f"  Exact Accuracy:      {stats['exact_accuracy']:.4f}")
            print(f"  Format Compliance:   {stats['format_compliance']:.4f}")
            print(f"  Mean Length:         {stats['mean_length']:.2f}")
            if policy != "sft":
                print(f"  RLAIF Win Rate vs SFT: {stats['rlaif_win_rate_vs_sft']:.4f}")
            
            # calculate drop
            if gsm_file.exists() and policy in gsm_stats:
                drop = gsm_stats[policy]['exact_accuracy'] - stats['exact_accuracy']
                print(f"  Accuracy Drop (GSM -> Transfer): {drop:.4f}")
        print()
    else:
        print(f"Missing {transfer_file}")


if __name__ == "__main__":
    main()
