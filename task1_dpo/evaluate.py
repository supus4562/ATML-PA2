"""Task 1 – DPO Evaluation.

Standalone: reads only from disk — no hidden notebook state required.
Run after task1_dpo.train has saved a checkpoint.

Usage
-----
  python -m task1_dpo.evaluate \\
      --config configs/dpo.yaml \\
      --adapter outputs/task1_dpo/standard \\
      --name standard

Output files (all in results/task1_dpo/)
----------------------------------------
  {name}_metrics.json        – aggregate metrics (loss, pref_acc, reward, KL, lengths)
  {name}_word_limit.json     – per-prompt word-limit compliance (base vs DPO)
  {name}_qualitative.jsonl   – 5 side-by-side base vs DPO response examples
  {name}_eval_rows.jsonl     – per-example results for plotting
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
    load_yaml,
    prompt_messages,
    prompt_messages_from_preference,
    preference_responses,
    read_jsonl,
    repo_path,
)
from common.generation import (
    batch_generate,
    response_sequence_logprobs,
    score_reward_pairs,
)
from common.logging_utils import save_json
from common.metrics import (
    sampled_kl,
    word_count,
    word_limit_compliance,
    parse_word_limit,
    safe_corr,
)
from common.models import (
    load_policy,
    load_reward_model,
    load_tokenizer,
    reference_mode,
)
from task1_dpo.dpo import dpo_loss


# ─────────────────────────────────────────────────────────────────────────────
# Bundle loader  (reused by ablate_beta.py)
# ─────────────────────────────────────────────────────────────────────────────

def load_evaluation_bundle(config_path: str, adapter: str) -> dict:
    """Load everything needed to evaluate one DPO checkpoint.

    Returns
    -------
    dict with keys: cfg, rows, tokenizer, policy, reward (model + tok)
    """
    cfg = load_yaml(config_path)
    rows = read_jsonl(cfg["paths"]["dpo_standard_eval"])
    tokenizer = load_tokenizer(cfg["base_model"])
    policy = load_policy(cfg, adapter_path=adapter, trainable=False)
    reward = load_reward_model(cfg)   # returns (model, tokenizer)
    return {
        "cfg": cfg,
        "rows": rows,
        "tokenizer": tokenizer,
        "policy": policy,
        "reward": reward,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Core evaluation  (called by main() and by ablate_beta.py)
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def run_evaluation(
    config_path: str,
    adapter: str,
    name: str,
    eval_rows: Optional[list[dict]] = None,
) -> dict:
    """Full Task 1 evaluation. Computes all required metrics and saves to disk.

    Parameters
    ----------
    config_path : str
        Path to dpo.yaml.
    adapter : str
        Path to the trained LoRA adapter checkpoint.
    name : str
        Run name used as prefix for all output files.
    eval_rows : list[dict] | None
        Optional pre-loaded eval rows (avoids re-reading disk in ablations).

    Returns
    -------
    dict of aggregate metrics.
    """
    cfg = load_yaml(config_path)
    results_dir = repo_path(cfg["results_dir"])
    results_dir.mkdir(parents=True, exist_ok=True)

    tokenizer    = load_tokenizer(cfg["base_model"])
    policy       = load_policy(cfg, adapter_path=adapter, trainable=False)
    rm_model, rm_tokenizer = load_reward_model(cfg)
    device       = next(policy.parameters()).device

    if eval_rows is None:
        eval_rows = read_jsonl(cfg["paths"]["dpo_standard_eval"])

    max_seq      = int(cfg["max_sequence_length"])
    max_gen_tok  = int(cfg["max_generation_tokens"])
    batch_size   = int(cfg["batch_size"])
    beta         = float(cfg["beta"])
    gen_cfg      = cfg.get("generation", {})

    print(f"\n[Evaluate] '{name}'  |  {len(eval_rows)} eval examples")

    # ── Encode eval set for logprob computation ───────────────────────────────
    from common.data import (
        encode_prompt_response,
        pad_batch,
        preference_responses as _pref_resp,
        prompt_messages_from_preference as _pmfp,
    )
    from common.models import reference_mode

    def _make_batches(rows, bs):
        for i in range(0, len(rows), bs):
            yield rows[i: i + bs]

    # ── 1. DPO loss + preference accuracy on held-out eval set ───────────────
    print("  Step 1/5: Computing DPO loss and preference accuracy on eval set...")
    all_policy_chosen_logp   = []
    all_policy_rejected_logp = []
    all_ref_chosen_logp      = []
    all_ref_rejected_logp    = []

    for batch_rows in tqdm(list(_make_batches(eval_rows, batch_size)), leave=False):
        chosen_list, rejected_list = [], []
        for row in batch_rows:
            prompt = _pmfp(row)
            yc, yr = _pref_resp(row)
            try:
                chosen_list.append(encode_prompt_response(tokenizer, prompt, yc, max_seq))
                rejected_list.append(encode_prompt_response(tokenizer, prompt, yr, max_seq))
            except ValueError:
                continue
        if not chosen_list:
            continue
        from common.data import pad_batch
        cb = {k: v.to(device) for k, v in pad_batch(tokenizer, chosen_list).items()}
        rb = {k: v.to(device) for k, v in pad_batch(tokenizer, rejected_list).items()}

        pc_logp, _, _ = response_sequence_logprobs(policy, cb)
        pr_logp, _, _ = response_sequence_logprobs(policy, rb)

        with reference_mode(policy):
            rc_logp, _, _ = response_sequence_logprobs(policy, cb)
            rr_logp, _, _ = response_sequence_logprobs(policy, rb)

        all_policy_chosen_logp.extend(pc_logp.cpu().tolist())
        all_policy_rejected_logp.extend(pr_logp.cpu().tolist())
        all_ref_chosen_logp.extend(rc_logp.cpu().tolist())
        all_ref_rejected_logp.extend(rr_logp.cpu().tolist())

    pc  = torch.tensor(all_policy_chosen_logp)
    pr_ = torch.tensor(all_policy_rejected_logp)
    rc  = torch.tensor(all_ref_chosen_logp)
    rr  = torch.tensor(all_ref_rejected_logp)

    eval_loss, eval_diag = dpo_loss(pc, pr_, rc, rr, beta)
    eval_loss_val  = eval_loss.item()
    eval_pref_acc  = eval_diag["preference_accuracy"].item()
    eval_logit_mean = eval_diag["logit_mean"].item()

    # Sampled KL (token-level, using chosen responses as sample points)
    policy_margin_all = pc - pr_
    ref_margin_all    = rc - rr
    # KL approximation: E[log π_θ - log π_ref] summed over sequences
    kl_vals = (pc - rc).numpy()
    kl_mean = float(np.mean(kl_vals))
    kl_std  = float(np.std(kl_vals))

    print(f"    eval_loss={eval_loss_val:.4f}  pref_acc={eval_pref_acc:.3f}  kl={kl_mean:.4f}")

    # ── 2. Generate responses + reward scores ─────────────────────────────────
    print("  Step 2/5: Generating responses and computing reward-model scores...")
    prompts_list  = [prompt_messages_from_preference(r) for r in eval_rows]
    all_responses = []
    all_lengths   = []
    all_truncated = []

    for i in tqdm(range(0, len(prompts_list), batch_size), leave=False):
        batch_prompts = prompts_list[i: i + batch_size]
        gen_out = batch_generate(
            policy, tokenizer, batch_prompts,
            max_prompt_length=max_seq,
            max_new_tokens=max_gen_tok,
            temperature=gen_cfg.get("temperature", 0.7),
            top_p=gen_cfg.get("top_p", 0.9),
            do_sample=gen_cfg.get("do_sample", True),
        )
        all_responses.extend(gen_out["responses"])
        all_lengths.extend(gen_out["response_lengths"])
        all_truncated.extend(gen_out["truncated"])

    # Reward-model scoring
    all_rewards = []
    for i in tqdm(range(0, len(prompts_list), batch_size), leave=False,
                  desc="Reward scoring"):
        batch_p   = prompts_list[i: i + batch_size]
        batch_r   = all_responses[i: i + batch_size]
        scores    = score_reward_pairs(rm_model, rm_tokenizer, batch_p, batch_r)
        all_rewards.extend(scores.cpu().tolist())

    lengths_arr = np.array(all_lengths, dtype=float)
    rewards_arr = np.array(all_rewards, dtype=float)
    length_reward_corr = safe_corr(lengths_arr, rewards_arr)

    print(f"    mean_reward={rewards_arr.mean():.4f}  "
          f"length–reward corr={length_reward_corr:.3f}")

    # ── 3. Word-limit compliance ───────────────────────────────────────────────
    # Run BOTH the base model AND the DPO model on the 10 fixed prompts.
    # This is the Task 1 required word-limit analysis.
    print("  Step 3/5: Word-limit compliance analysis (base vs DPO)...")
    wl_prompts = read_jsonl(cfg["paths"]["word_limit_prompts"])
    wl_results = []

    # --- DPO model ---
    dpo_wl_gen = batch_generate(
        policy, tokenizer,
        [p["messages"] for p in wl_prompts],
        max_prompt_length=max_seq,
        max_new_tokens=max_gen_tok,
        do_sample=False,       # greedy for reproducibility in compliance test
    )
    dpo_wl_texts = dpo_wl_gen["responses"]

    # --- Base model (reference mode = no LoRA adapter) ---
    with reference_mode(policy):
        base_wl_gen = batch_generate(
            policy, tokenizer,
            [p["messages"] for p in wl_prompts],
            max_prompt_length=max_seq,
            max_new_tokens=max_gen_tok,
            do_sample=False,
        )
    base_wl_texts = base_wl_gen["responses"]

    for p_row, dpo_text, base_text in zip(wl_prompts, dpo_wl_texts, base_wl_texts):
        prompt_str = p_row["messages"][0]["content"]
        limit      = parse_word_limit(prompt_str)
        wl_results.append({
            "prompt_id":          p_row["prompt_id"],
            "prompt":             prompt_str,
            "word_limit":         limit,
            "base_response":      base_text,
            "base_word_count":    word_count(base_text),
            "base_compliant":     word_limit_compliance(prompt_str, base_text),
            "dpo_response":       dpo_text,
            "dpo_word_count":     word_count(dpo_text),
            "dpo_compliant":      word_limit_compliance(prompt_str, dpo_text),
        })

    base_compliance = float(np.mean([r["base_compliant"] for r in wl_results
                                     if r["base_compliant"] is not None]))
    dpo_compliance  = float(np.mean([r["dpo_compliant"]  for r in wl_results
                                     if r["dpo_compliant"] is not None]))
    print(f"    base compliance={base_compliance:.2%}  dpo compliance={dpo_compliance:.2%}")

    # ── 4. Qualitative examples ───────────────────────────────────────────────
    print("  Step 4/5: Generating qualitative examples (5 prompts)...")
    qual_rows   = eval_rows[:5]
    qual_prompts = [prompt_messages_from_preference(r) for r in qual_rows]

    dpo_qual_gen  = batch_generate(
        policy, tokenizer, qual_prompts,
        max_prompt_length=max_seq, max_new_tokens=max_gen_tok, do_sample=False,
    )
    with reference_mode(policy):
        base_qual_gen = batch_generate(
            policy, tokenizer, qual_prompts,
            max_prompt_length=max_seq, max_new_tokens=max_gen_tok, do_sample=False,
        )

    qualitative = []
    for i, row in enumerate(qual_rows):
        yc, yr = preference_responses(row)
        qualitative.append({
            "prompt":              qual_prompts[i],
            "label_chosen":        yc,
            "label_rejected":      yr,
            "base_response":       base_qual_gen["responses"][i],
            "dpo_response":        dpo_qual_gen["responses"][i],
        })

    # ── 5. Per-example rows ────────────────────────────────────────────────────
    print("  Step 5/5: Saving per-example results...")
    n_min = min(len(all_responses), len(eval_rows),
                len(all_policy_chosen_logp), len(all_rewards))
    eval_rows_out = []
    for i in range(n_min):
        eval_rows_out.append({
            "idx":                   i,
            "policy_chosen_logp":    all_policy_chosen_logp[i],
            "policy_rejected_logp":  all_policy_rejected_logp[i],
            "ref_chosen_logp":       all_ref_chosen_logp[i],
            "ref_rejected_logp":     all_ref_rejected_logp[i],
            "kl":                    all_policy_chosen_logp[i] - all_ref_chosen_logp[i],
            "response":              all_responses[i],
            "response_length":       all_lengths[i],
            "truncated":             all_truncated[i],
            "reward":                all_rewards[i],
        })

    # ── Save all outputs ───────────────────────────────────────────────────────
    metrics = {
        "run_name":             name,
        "adapter":              str(adapter),
        "n_eval":               len(eval_rows),
        # DPO objective
        "eval_loss":            eval_loss_val,
        "eval_logit_mean":      eval_logit_mean,
        "preference_accuracy":  eval_pref_acc,
        # KL from reference
        "kl_mean":              kl_mean,
        "kl_std":               kl_std,
        # Reward model
        "reward_mean":          float(rewards_arr.mean()),
        "reward_std":           float(rewards_arr.std()),
        "reward_p5":            float(np.percentile(rewards_arr, 5)),
        "reward_p95":           float(np.percentile(rewards_arr, 95)),
        # Response lengths
        "length_mean":          float(lengths_arr.mean()),
        "length_median":        float(np.median(lengths_arr)),
        "length_p5":            float(np.percentile(lengths_arr, 5)),
        "length_p95":           float(np.percentile(lengths_arr, 95)),
        "truncation_rate":      float(np.mean(all_truncated)),
        # Length–reward relationship
        "length_reward_corr":   length_reward_corr,
        # Word-limit compliance
        "base_wl_compliance":   base_compliance,
        "dpo_wl_compliance":    dpo_compliance,
    }

    save_json(results_dir / f"{name}_metrics.json", metrics)
    save_json(results_dir / f"{name}_word_limit.json", wl_results)

    # Write qualitative JSONL
    qual_path = results_dir / f"{name}_qualitative.jsonl"
    with qual_path.open("w", encoding="utf-8") as f:
        for q in qualitative:
            f.write(json.dumps(q, ensure_ascii=False) + "\n")

    # Write per-example JSONL
    rows_path = results_dir / f"{name}_eval_rows.jsonl"
    with rows_path.open("w", encoding="utf-8") as f:
        for row in eval_rows_out:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"\n[Evaluate] Results saved to {results_dir}/ (prefix: '{name}_')")
    print(f"  eval_loss={eval_loss_val:.4f}  pref_acc={eval_pref_acc:.3f}  "
          f"reward={rewards_arr.mean():.4f}  kl={kl_mean:.4f}")
    print(f"  base_wl={base_compliance:.2%}  dpo_wl={dpo_compliance:.2%}\n")

    return metrics


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description="Task 1 – DPO evaluation")
    ap.add_argument("--config",  default="configs/dpo.yaml")
    ap.add_argument("--adapter", required=True, help="Path to LoRA adapter checkpoint")
    ap.add_argument("--name",    default="standard", help="Run name prefix for output files")
    args = ap.parse_args()

    run_evaluation(args.config, args.adapter, args.name)


if __name__ == "__main__":
    main()
