from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

from common.data import load_yaml, read_jsonl, repo_path
from task5_feedback.rlvr import exact_reward
from task5_feedback.rlaif import PairwiseAIJudge

EXPECTED_VARIANTS = {
    "clean_correct",
    "corrupt_reasoning_correct_final",
    "good_reasoning_wrong_final",
    "persuasive_filler_correct",
    "gold_distractor_wrong_final",
}


def load_diagnostic_groups(path):
    rows = read_jsonl(path)
    by_problem = defaultdict(dict)
    for row in rows:
        by_problem[str(row["problem_id"])][row["variant_type"]] = row
    for pid, variants in by_problem.items():
        missing = EXPECTED_VARIANTS - set(variants)
        if missing:
            raise ValueError(f"Problem {pid} missing variants: {sorted(missing)}")
    return dict(by_problem)


def eval_pair(better_row, worse_row, judge, mechanism="rlvr"):
    better_resp = better_row["response"]
    worse_resp = worse_row["response"]
    gold_final = str(better_row["gold_final"])
    
    if mechanism == "rlvr":
        r_better = exact_reward(better_resp, gold_final)
        r_worse = exact_reward(worse_resp, gold_final)
        if r_better > r_worse:
            return "BETTER"
        elif r_worse > r_better:
            return "WORSE"
        else:
            return "TIE"
    elif mechanism == "rlaif":
        problem = better_row.get("question", better_row.get("prompt", ""))
        pref = judge.compare(problem, better_resp, worse_resp)
        if pref == "A":
            return "BETTER"
        elif pref == "B":
            return "WORSE"
        else:
            return "TIE"
    else:
        raise ValueError(mechanism)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/feedback.yaml")
    args = ap.parse_args()
    cfg = load_yaml(args.config)
    
    groups = load_diagnostic_groups(repo_path(cfg["paths"]["task5_diagnostics"]))
    print("Diagnostic problems:", len(groups))
    print("Variants/problem:", sorted(EXPECTED_VARIANTS))
    
    outdir = Path(repo_path(cfg["results_dir"])) / "task5_feedback"
    outdir.mkdir(parents=True, exist_ok=True)
    ai_judge = PairwiseAIJudge(cfg, cache_path=outdir / "ai_judge_cache.json")
    
    comparisons = {
        "reasoning_sensitivity": ("clean_correct", "corrupt_reasoning_correct_final"),
        "outcome_sensitivity": ("clean_correct", "good_reasoning_wrong_final"),
        "filler_susceptibility": ("clean_correct", "persuasive_filler_correct"),
        "distractor_robustness": ("clean_correct", "gold_distractor_wrong_final")
    }
    
    results = {}
    
    for mech in ["rlvr", "rlaif"]:
        print(f"\n--- Mechanism: {mech.upper()} ---")
        mech_res = {}
        
        for comp_name, (better_var, worse_var) in comparisons.items():
            better_count = 0
            worse_count = 0
            tie_count = 0
            total = len(groups)
            
            for pid, variants in groups.items():
                res = eval_pair(variants[better_var], variants[worse_var], ai_judge, mechanism=mech)
                if res == "BETTER":
                    better_count += 1
                elif res == "WORSE":
                    worse_count += 1
                else:
                    tie_count += 1
                    
            mech_res[comp_name] = {
                "better_rate": better_count / total,
                "worse_rate": worse_count / total,
                "tie_rate": tie_count / total,
            }
            
            print(f"{comp_name}:")
            print(f"  Prefers better: {better_count}/{total} ({better_count/total:.2f})")
            print(f"  Ties:           {tie_count}/{total} ({tie_count/total:.2f})")
            print(f"  Prefers worse:  {worse_count}/{total} ({worse_count/total:.2f})")
            
        # S_reason and S_outcome
        s_reason = mech_res["reasoning_sensitivity"]["better_rate"]
        s_outcome = mech_res["outcome_sensitivity"]["better_rate"]
        print(f"  S_reason: {s_reason:.2f}")
        print(f"  S_outcome: {s_outcome:.2f}")
        
        mech_res["S_reason"] = s_reason
        mech_res["S_outcome"] = s_outcome
        results[mech] = mech_res

    with open(outdir / "diagnostic_results.json", "w") as f:
        json.dump(results, f, indent=2)
        
    print(f"\nSaved diagnostic results to {outdir / 'diagnostic_results.json'}")


if __name__ == "__main__":
    main()
