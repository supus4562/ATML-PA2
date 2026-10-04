"""Task 1 – Length-Confounding Study.

Standalone: reads only from disk — no hidden notebook state required.

This experiment asks: is standard DPO actually learning response quality,
or is it just learning that longer responses are preferred (length bias)?

Method
------
1. Train DPO on the length-BALANCED dataset (chosen/rejected pairs have
   similar token counts, so length cannot be used as a preference signal).
2. Evaluate on the length-STRATIFIED eval set (examples stratified into
   short / medium / long response strata).
3. Compare per-stratum preference accuracy and reward scores between:
   (a) the standard DPO model (trained on possibly length-confounded data)
   (b) the length-balanced DPO model
4. Compute length–reward Spearman correlation for both models.
   If standard DPO is exploiting length, its correlation will be high;
   the length-balanced model's correlation should be closer to zero.

Usage
-----
  python -m task1_dpo.analyze_length --config configs/dpo.yaml

Output (in results/task1_dpo/)
-------------------------------
  length_analysis.json           – per-stratum metrics + correlations
  train_log_length_balanced.jsonl
  length_balanced_metrics.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from tqdm import tqdm

from common.data import (
    encode_prompt_response,
    load_yaml,
    pad_batch,
    preference_responses,
    prompt_messages_from_preference,
    read_jsonl,
    repo_path,
)
from common.generation import (
    batch_generate,
    response_sequence_logprobs,
    score_reward_pairs,
)
from common.logging_utils import save_json
from common.metrics import safe_corr, word_count
from common.models import (
    load_policy,
    load_reward_model,
    load_tokenizer,
    reference_mode,
)
from task1_dpo.train import run_training
from task1_dpo.evaluate import run_evaluation


# ─────────────────────────────────────────────────────────────────────────────
# Stratum detection
# ─────────────────────────────────────────────────────────────────────────────

def _detect_stratum(row: dict) -> str:
    """Detect the length stratum from the eval row.

    The stratified eval set may encode the stratum explicitly (key 'stratum')
    or we infer it from the chosen response token count.
    Short: <64 tokens, Medium: 64-256, Long: >256 (approximate word proxy).
    """
    if "stratum" in row:
        return str(row["stratum"])
    yc, _ = preference_responses(row)
    wc = word_count(yc)
    if wc < 40:
        return "short"
    elif wc < 160:
        return "medium"
    else:
        return "long"


# ─────────────────────────────────────────────────────────────────────────────
# Per-stratum evaluation
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate_length_stratified(
    config_path: str,
    adapter: str,
    run_name: str,
    eval_rows: list[dict],
) -> dict:
    """Evaluate a model on the length-stratified eval set.

    Returns per-stratum and aggregate metrics including length–reward correlation.
    """
    cfg         = load_yaml(config_path)
    tokenizer   = load_tokenizer(cfg["base_model"])
    policy      = load_policy(cfg, adapter_path=adapter, trainable=False)
    rm_model, rm_tokenizer = load_reward_model(cfg)
    device      = next(policy.parameters()).device

    max_seq     = int(cfg["max_sequence_length"])
    max_gen_tok = int(cfg["max_generation_tokens"])
    batch_size  = int(cfg["batch_size"])
    gen_cfg     = cfg.get("generation", {})

    # Group rows by stratum
    strata: dict[str, list[dict]] = {}
    for row in eval_rows:
        s = _detect_stratum(row)
        strata.setdefault(s, []).append(row)

    print(f"  Strata: { {k: len(v) for k, v in strata.items()} }")

    def _make_batches(rows, bs):
        for i in range(0, len(rows), bs):
            yield rows[i: i + bs]

    all_results: list[dict] = []

    for stratum, rows in sorted(strata.items()):
        print(f"    Evaluating stratum='{stratum}' ({len(rows)} examples)...")
        pref_accs, rewards, lengths = [], [], []

        for batch_rows in _make_batches(rows, batch_size):
            chosen_list, rejected_list, prompts_list = [], [], []
            for row in batch_rows:
                prompt = prompt_messages_from_preference(row)
                prompts_list.append(prompt)
                yc, yr = preference_responses(row)
                try:
                    chosen_list.append(encode_prompt_response(tokenizer, prompt, yc, max_seq))
                    rejected_list.append(encode_prompt_response(tokenizer, prompt, yr, max_seq))
                except ValueError:
                    chosen_list.append(None)
                    rejected_list.append(None)

            # Filter out None pairs
            valid = [(c, r, p) for c, r, p in zip(chosen_list, rejected_list, prompts_list)
                     if c is not None and r is not None]
            if not valid:
                continue
            valid_chosen, valid_rejected, valid_prompts = zip(*valid)

            cb = {k: v.to(device) for k, v in pad_batch(tokenizer, list(valid_chosen)).items()}
            rb = {k: v.to(device) for k, v in pad_batch(tokenizer, list(valid_rejected)).items()}

            pc_logp, _, _ = response_sequence_logprobs(policy, cb)
            pr_logp, _, _ = response_sequence_logprobs(policy, rb)
            with reference_mode(policy):
                rc_logp, _, _ = response_sequence_logprobs(policy, cb)
                rr_logp, _, _ = response_sequence_logprobs(policy, rb)

            policy_margin = pc_logp - pr_logp
            ref_margin    = rc_logp - rr_logp
            pref_acc_batch = ((policy_margin - ref_margin) > 0).float()
            pref_accs.extend(pref_acc_batch.cpu().tolist())

            # Generate responses for reward scoring
            gen_out = batch_generate(
                policy, tokenizer, list(valid_prompts),
                max_prompt_length=max_seq,
                max_new_tokens=max_gen_tok,
                temperature=gen_cfg.get("temperature", 0.7),
                top_p=gen_cfg.get("top_p", 0.9),
                do_sample=gen_cfg.get("do_sample", True),
            )
            batch_rewards = score_reward_pairs(
                rm_model, rm_tokenizer, list(valid_prompts), gen_out["responses"]
            )
            rewards.extend(batch_rewards.cpu().tolist())
            lengths.extend(gen_out["response_lengths"])

        rewards_arr = np.array(rewards, dtype=float)
        lengths_arr = np.array(lengths, dtype=float)
        all_results.append({
            "run_name":           run_name,
            "stratum":            stratum,
            "n":                  len(rows),
            "preference_accuracy": float(np.mean(pref_accs)) if pref_accs else float("nan"),
            "reward_mean":        float(rewards_arr.mean()) if len(rewards_arr) else float("nan"),
            "reward_std":         float(rewards_arr.std())  if len(rewards_arr) else float("nan"),
            "length_mean":        float(lengths_arr.mean()) if len(lengths_arr) else float("nan"),
            "length_reward_corr": safe_corr(lengths_arr, rewards_arr),
        })

    # Aggregate across all strata
    all_rewards  = [r for row in all_results for r in [row["reward_mean"]] if not np.isnan(r)]
    all_pref_acc = [row["preference_accuracy"] for row in all_results if not np.isnan(row["preference_accuracy"])]

    return {
        "run_name":            run_name,
        "per_stratum":         all_results,
        "aggregate_reward":    float(np.mean(all_rewards))  if all_rewards  else float("nan"),
        "aggregate_pref_acc":  float(np.mean(all_pref_acc)) if all_pref_acc else float("nan"),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Main entry point
# ─────────────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description="Task 1 – Length-confounding analysis")
    ap.add_argument("--config", default="configs/dpo.yaml")
    args = ap.parse_args()

    cfg = load_yaml(args.config)
    res_dir = repo_path(cfg["results_dir"])
    res_dir.mkdir(parents=True, exist_ok=True)

    len_train_path = cfg["paths"]["dpo_length_train"]
    len_eval_path  = cfg["paths"]["dpo_length_eval"]

    print(f"\n{'='*60}")
    print(f"  Task 1 – Length-Confounding Study")
    balanced   = read_jsonl(len_train_path)
    stratified = read_jsonl(len_eval_path)
    print(f"  Length-balanced train rows:    {len(balanced)}")
    print(f"  Length-stratified eval rows:   {len(stratified)}")
    print(f"{'='*60}\n")

    # ── Step 1: Train on length-balanced data ─────────────────────────────────
    print("Step 1: Training DPO on length-balanced dataset...")
    len_output = cfg.get("length_output", "outputs/task1_dpo/length_balanced")
    run_training(
        config_path=args.config,
        run_name="length_balanced",
        dataset_path=len_train_path,
        output_path=len_output,
        use_compile=False,
    )

    # ── Step 2: Standard DPO model on stratified eval ─────────────────────────
    # The standard DPO adapter was saved by train.py → outputs/task1_dpo/standard
    standard_output = cfg.get("standard_output", "outputs/task1_dpo/standard")
    standard_adapter = repo_path(standard_output)

    results: dict = {
        "standard": None,
        "length_balanced": None,
    }

    if standard_adapter.exists():
        print("\nStep 2a: Evaluating standard DPO model on stratified eval set...")
        results["standard"] = evaluate_length_stratified(
            args.config, str(standard_adapter), "standard", stratified
        )
    else:
        print(
            f"\nStep 2a: Standard adapter not found at {standard_adapter}. "
            "Run task1_dpo.train first, then re-run this script."
        )

    # ── Step 3: Length-balanced DPO model on stratified eval ──────────────────
    print("\nStep 2b: Evaluating length-balanced DPO model on stratified eval set...")
    results["length_balanced"] = evaluate_length_stratified(
        args.config, len_output, "length_balanced", stratified
    )

    # ── Step 4: Compute length–reward Spearman correlations ───────────────────
    # Also run a standard evaluate.py sweep on the length-balanced adapter
    # to get the full set of eval metrics for comparison.
    print("\nStep 3: Running full evaluation on length-balanced adapter...")
    run_evaluation(
        config_path=args.config,
        adapter=len_output,
        name="length_balanced",
    )

    # ── Save combined length analysis ─────────────────────────────────────────
    output = {
        "description": (
            "Per-stratum preference accuracy and reward for standard DPO vs "
            "length-balanced DPO. If standard DPO is exploiting length bias, "
            "its 'long' stratum reward will be disproportionately high and "
            "its length–reward correlation will exceed the balanced model's."
        ),
        "standard":        results["standard"],
        "length_balanced": results["length_balanced"],
    }
    out_path = res_dir / "length_analysis.json"
    save_json(out_path, output)

    # ── Print comparison table ─────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  Length-Confounding Analysis Complete")
    print(f"  Saved → {out_path}\n")

    for model_key in ["standard", "length_balanced"]:
        r = results.get(model_key)
        if r is None:
            continue
        print(f"  Model: {model_key}")
        print(f"  {'Stratum':<10} | {'Pref Acc':>9} | {'Reward':>8} | {'Len–Reward r':>12}")
        print(f"  {'-'*10}-+-{'-'*9}-+-{'-'*8}-+-{'-'*12}")
        for s in r["per_stratum"]:
            print(
                f"  {s['stratum']:<10} | {s['preference_accuracy']:>9.3f} | "
                f"{s['reward_mean']:>8.4f} | {s['length_reward_corr']:>12.3f}"
            )
        print()

    print(
        "  Interpretation: if length–reward corr is high for 'standard' but\n"
        "  low for 'length_balanced', standard DPO was exploiting length bias.\n"
    )
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
