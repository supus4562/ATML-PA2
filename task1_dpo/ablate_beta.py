"""Task 1 – β Ablation Study.

Standalone: reads only from disk — no hidden notebook state required.
Trains DPO with each β ∈ {0.03, 0.10, 0.30} from the SAME random seed
and base initialisation, then evaluates all three under a common protocol.

Usage
-----
  python -m task1_dpo.ablate_beta --config configs/dpo.yaml

Output (in results/task1_dpo/)
-------------------------------
  beta_ablation_summary.json   – side-by-side metrics table for all β values
  train_log_beta_{b}.jsonl     – per-step training log for each β
  beta_{b}_metrics.json        – per-β evaluation metrics
  (all other evaluate.py outputs for each β run)
"""

from __future__ import annotations

import argparse

from common.data import load_yaml, repo_path
from common.logging_utils import save_json
from task1_dpo.train import run_training
from task1_dpo.evaluate import run_evaluation


def main():
    ap = argparse.ArgumentParser(description="Task 1 – DPO β ablation")
    ap.add_argument("--config", default="configs/dpo.yaml")
    args = ap.parse_args()

    cfg      = load_yaml(args.config)
    betas    = cfg["betas"]                          # [0.03, 0.10, 0.30] — fixed by assignment
    n_ex     = int(cfg["short_ablation_examples"])   # 600 — fixed by assignment
    base_out = cfg.get("standard_output", "outputs/task1_dpo/standard")
    res_dir  = repo_path(cfg["results_dir"])
    res_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"  β Ablation Study")
    print(f"  β values:        {betas}")
    print(f"  Examples / run:  {n_ex}")
    print(f"  NOTE: all runs share the same seed ({cfg['seed']}) and start")
    print(f"  from the same base model initialisation for a fair comparison.")
    print(f"{'='*60}\n")

    all_metrics: list[dict] = []

    for b in betas:
        run_name    = f"beta_{b}"
        output_path = f"outputs/task1_dpo/{run_name}"

        print(f"\n{'─'*50}")
        print(f"  Training: β={b}  run_name={run_name}")
        print(f"{'─'*50}\n")

        # ── Train ────────────────────────────────────────────────────────────
        # max_examples=n_ex limits to the short ablation budget.
        # beta=b overrides the config default.
        # use_compile=False (safe default for ablation stability).
        train_summary = run_training(
            config_path=args.config,
            run_name=run_name,
            beta=b,
            max_examples=n_ex,
            output_path=output_path,
            use_compile=False,
        )

        print(f"\n{'─'*50}")
        print(f"  Evaluating: β={b}")
        print(f"{'─'*50}\n")

        # ── Evaluate ─────────────────────────────────────────────────────────
        metrics = run_evaluation(
            config_path=args.config,
            adapter=output_path,
            name=run_name,
        )

        # Attach training summary fields for the combined table.
        metrics["beta"]              = b
        metrics["train_mean_loss"]   = train_summary["mean_loss"]
        metrics["train_pref_acc"]    = train_summary["mean_preference_accuracy"]
        metrics["train_elapsed_s"]   = train_summary["elapsed_s"]

        all_metrics.append(metrics)
        print(f"  β={b} done. reward={metrics['reward_mean']:.4f}  "
              f"kl={metrics['kl_mean']:.4f}  pref_acc={metrics['preference_accuracy']:.3f}")

    # ── Combined summary table ────────────────────────────────────────────────
    # One entry per β — useful for plotting and the PA report table.
    summary = {
        "betas":  betas,
        "runs":   all_metrics,
        # Flatten key metrics for quick inspection
        "reward_by_beta":    {str(m["beta"]): m["reward_mean"]           for m in all_metrics},
        "kl_by_beta":        {str(m["beta"]): m["kl_mean"]               for m in all_metrics},
        "pref_acc_by_beta":  {str(m["beta"]): m["preference_accuracy"]   for m in all_metrics},
        "eval_loss_by_beta": {str(m["beta"]): m["eval_loss"]             for m in all_metrics},
    }
    out_path = res_dir / "beta_ablation_summary.json"
    save_json(out_path, summary)

    print(f"\n{'='*60}")
    print(f"  β Ablation Complete")
    print(f"  Summary saved → {out_path}")
    print(f"\n  β      | reward    | KL       | pref_acc")
    print(f"  -------|-----------|----------|---------")
    for m in all_metrics:
        print(
            f"  {m['beta']:<6} | {m['reward_mean']:>8.4f}  | "
            f"{m['kl_mean']:>7.4f}  | {m['preference_accuracy']:.3f}"
        )
    print(f"\n  Expected pattern: as β↑, KL↓ (tighter constraint to reference).")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
