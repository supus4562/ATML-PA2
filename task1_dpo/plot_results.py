"""Task 1 – Results plotting.

Standalone: reads ONLY from results/ and outputs/ on disk.
No model loading, no hidden notebook state required.
Gracefully skips any plot if the required results file is missing.

Usage
-----
  python -m task1_dpo.plot_results --config configs/dpo.yaml

Generated plots (all 300 DPI, saved to results/task1_dpo/plots/)
-----------------------------------------------------------------
  01_train_loss.png            – training loss curve
  02_train_pref_acc.png        – training preference accuracy curve
  03_train_lr.png              – learning-rate schedule curve
  04_beta_reward.png           – β ablation: reward score per β
  05_beta_kl.png               – β ablation: KL divergence per β
  06_beta_pref_acc.png         – β ablation: preference accuracy per β
  07_beta_heatmap.png          – β ablation: all metrics as seaborn heatmap
  08_length_strat_reward.png   – per-stratum reward (standard vs length-balanced)
  09_length_strat_pref_acc.png – per-stratum preference accuracy comparison
  10_length_reward_corr.png    – length vs reward scatter + regression line
  11_word_limit.png            – word-limit compliance bar (base vs DPO per prompt)
  12_length_dist.png           – response length violin plot (base vs DPO)
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use("Agg")   # non-interactive backend (safe for servers)
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import seaborn as sns

from common.data import load_yaml, repo_path

sns.set_theme(style="whitegrid", palette="muted", font_scale=1.15)
DPI = 300


# ─────────────────────────────────────────────────────────────────────────────
# I/O helpers
# ─────────────────────────────────────────────────────────────────────────────

def _load_json(path: Path) -> Optional[dict | list]:
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    print(f"  [skip] {path.name} not found.")
    return None


def _load_jsonl(path: Path) -> Optional[list[dict]]:
    if not path.exists():
        print(f"  [skip] {path.name} not found.")
        return None
    rows = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows if rows else None


def _savefig(fig, path: Path, title: str):
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path.name}")


# ─────────────────────────────────────────────────────────────────────────────
# Individual plot functions
# ─────────────────────────────────────────────────────────────────────────────

def plot_train_loss(train_log: list[dict], plots_dir: Path):
    steps  = [r["optimizer_step"] for r in train_log]
    losses = [r["loss"] for r in train_log]

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(steps, losses, lw=0.8, alpha=0.5, color="steelblue", label="micro-step loss")
    # Smoothed curve (running mean over 20 steps)
    if len(losses) >= 20:
        smooth = np.convolve(losses, np.ones(20) / 20, mode="valid")
        ax.plot(steps[19:], smooth, lw=2.0, color="navy", label="smoothed (20-step)")
    ax.set_xlabel("Optimizer Step")
    ax.set_ylabel("DPO Loss")
    ax.set_title("Task 1 – DPO Training Loss")
    ax.legend()
    _savefig(fig, plots_dir / "01_train_loss.png", "train loss")


def plot_train_pref_acc(train_log: list[dict], plots_dir: Path):
    steps    = [r["optimizer_step"] for r in train_log]
    pref_acc = [r["preference_accuracy"] for r in train_log]

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(steps, pref_acc, lw=0.8, alpha=0.5, color="seagreen", label="micro-step")
    if len(pref_acc) >= 20:
        smooth = np.convolve(pref_acc, np.ones(20) / 20, mode="valid")
        ax.plot(steps[19:], smooth, lw=2.0, color="darkgreen", label="smoothed (20-step)")
    ax.axhline(0.5, color="red", linestyle="--", lw=1.0, label="chance (0.5)")
    ax.set_xlabel("Optimizer Step")
    ax.set_ylabel("Preference Accuracy")
    ax.set_ylim(0, 1)
    ax.set_title("Task 1 – DPO Training Preference Accuracy")
    ax.legend()
    _savefig(fig, plots_dir / "02_train_pref_acc.png", "train pref acc")


def plot_train_lr(train_log: list[dict], plots_dir: Path):
    steps = [r["optimizer_step"] for r in train_log]
    lrs   = [r.get("learning_rate", float("nan")) for r in train_log]
    if all(np.isnan(x) for x in lrs):
        return

    fig, ax = plt.subplots(figsize=(9, 3))
    ax.plot(steps, lrs, lw=1.5, color="darkorange")
    ax.set_xlabel("Optimizer Step")
    ax.set_ylabel("Learning Rate")
    ax.yaxis.set_major_formatter(mticker.ScalarFormatter(useMathText=True))
    ax.set_title("Task 1 – Learning-Rate Schedule (Linear Warmup + Decay)")
    _savefig(fig, plots_dir / "03_train_lr.png", "LR schedule")


def plot_beta_metric(summary: dict, metric_key: str,
                     ylabel: str, title: str, fname: str, plots_dir: Path,
                     higher_is_better: bool = True):
    """Generic bar chart for a single metric across β values."""
    betas  = [str(b) for b in summary["betas"]]
    values = [summary["runs"][i][metric_key] for i in range(len(betas))]

    fig, ax = plt.subplots(figsize=(7, 4))
    colors  = sns.color_palette("muted", len(betas))
    bars    = ax.bar(betas, values, color=colors, edgecolor="black", linewidth=0.7)
    ax.bar_label(bars, fmt="%.4f", padding=3, fontsize=10)
    best_idx = int(np.argmax(values) if higher_is_better else np.argmin(values))
    bars[best_idx].set_edgecolor("crimson")
    bars[best_idx].set_linewidth(2.5)
    ax.set_xlabel("β (KL-penalty strength)")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.annotate(
        "← tighter constraint    looser constraint →",
        xy=(0.5, -0.22), xycoords="axes fraction",
        ha="center", fontsize=9, color="gray",
    )
    _savefig(fig, plots_dir / fname, title)


def plot_beta_heatmap(summary: dict, plots_dir: Path):
    betas  = [str(b) for b in summary["betas"]]
    metric_keys = [
        ("reward_mean",          "Reward"),
        ("kl_mean",              "KL div"),
        ("preference_accuracy",  "Pref Acc"),
        ("eval_loss",            "Eval Loss"),
    ]
    data = np.array([
        [summary["runs"][i][mk] for i in range(len(betas))]
        for mk, _ in metric_keys
    ])
    row_labels = [ml for _, ml in metric_keys]

    # Normalise each row to [0,1] for visual comparison across different scales.
    data_norm = np.zeros_like(data)
    for r in range(data.shape[0]):
        rmin, rmax = data[r].min(), data[r].max()
        if rmax > rmin:
            data_norm[r] = (data[r] - rmin) / (rmax - rmin)
        else:
            data_norm[r] = 0.5

    fig, ax = plt.subplots(figsize=(8, 5))
    sns.heatmap(
        data_norm,
        ax=ax,
        xticklabels=betas,
        yticklabels=row_labels,
        annot=data,           # show raw values in cells
        fmt=".4f",
        cmap="YlOrRd",
        linewidths=0.5,
        cbar_kws={"label": "Row-normalised value"},
    )
    ax.set_xlabel("β")
    ax.set_title("Task 1 – β Ablation: All Metrics Heatmap")
    _savefig(fig, plots_dir / "07_beta_heatmap.png", "beta heatmap")


def plot_length_strat(length_analysis: dict, metric: str,
                      ylabel: str, title: str, fname: str, plots_dir: Path):
    """Grouped bar chart: standard vs length-balanced DPO, per stratum."""
    std_data = length_analysis.get("standard")
    bal_data = length_analysis.get("length_balanced")
    if std_data is None and bal_data is None:
        return

    strata = ["short", "medium", "long"]
    def _get(data, stratum, key):
        if data is None:
            return float("nan")
        for s in data["per_stratum"]:
            if s["stratum"] == stratum:
                return s.get(key, float("nan"))
        return float("nan")

    std_vals = [_get(std_data, s, metric) for s in strata]
    bal_vals = [_get(bal_data, s, metric) for s in strata]

    x  = np.arange(len(strata))
    w  = 0.35
    fig, ax = plt.subplots(figsize=(8, 4))
    if std_data:
        ax.bar(x - w / 2, std_vals, w, label="Standard DPO", color="#4878D0")
    if bal_data:
        ax.bar(x + w / 2, bal_vals, w, label="Length-balanced DPO", color="#EE854A")
    ax.set_xticks(x)
    ax.set_xticklabels(strata)
    ax.set_xlabel("Response Length Stratum")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()
    _savefig(fig, plots_dir / fname, title)


def plot_length_reward_corr(eval_rows: list[dict], name: str, plots_dir: Path):
    lengths = [r["response_length"] for r in eval_rows]
    rewards = [r["reward"] for r in eval_rows]
    if not lengths:
        return

    lengths_arr = np.array(lengths, dtype=float)
    rewards_arr = np.array(rewards, dtype=float)
    corr = np.corrcoef(lengths_arr, rewards_arr)[0, 1]

    # Regression line
    m, b = np.polyfit(lengths_arr, rewards_arr, 1)
    x_line = np.linspace(lengths_arr.min(), lengths_arr.max(), 200)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(lengths_arr, rewards_arr, alpha=0.35, s=18, color="#4878D0", label="examples")
    ax.plot(x_line, m * x_line + b, color="crimson", lw=2.0,
            label=f"regression (r={corr:.3f})")
    ax.set_xlabel("Response Length (tokens)")
    ax.set_ylabel("Reward Score")
    ax.set_title(f"Task 1 – Length vs Reward Correlation [{name}]")
    ax.legend()
    _savefig(fig, plots_dir / "10_length_reward_corr.png", "length reward corr")


def plot_word_limit(wl_results: list[dict], plots_dir: Path):
    prompt_ids    = [r["prompt_id"] for r in wl_results]
    base_compliant = [r.get("base_compliant") or 0 for r in wl_results]
    dpo_compliant  = [r.get("dpo_compliant")  or 0 for r in wl_results]

    x = np.arange(len(prompt_ids))
    w = 0.35
    fig, ax = plt.subplots(figsize=(11, 4))
    ax.bar(x - w / 2, base_compliant, w, label="Base model", color="#DD8452")
    ax.bar(x + w / 2, dpo_compliant,  w, label="DPO model",  color="#4878D0")
    ax.set_xticks(x)
    ax.set_xticklabels(prompt_ids, rotation=30, ha="right")
    ax.set_ylabel("Compliant (1 = yes)")
    ax.set_ylim(0, 1.25)
    ax.set_title("Task 1 – Word-Limit Compliance: Base vs DPO (10 fixed prompts)")
    ax.legend()

    # Annotate aggregate compliance rates
    base_agg = np.mean(base_compliant)
    dpo_agg  = np.mean(dpo_compliant)
    ax.text(0.01, 1.18, f"Base avg: {base_agg:.0%}  |  DPO avg: {dpo_agg:.0%}",
            transform=ax.transAxes, fontsize=10, color="black")
    _savefig(fig, plots_dir / "11_word_limit.png", "word limit compliance")


def plot_length_distribution(eval_rows: list[dict], name: str, plots_dir: Path):
    """Violin plot of response lengths — requires at least some DPO data."""
    lengths = [r["response_length"] for r in eval_rows]
    if not lengths:
        return

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.violinplot([lengths], positions=[1], showmedians=True)
    ax.set_xticks([1])
    ax.set_xticklabels([f"DPO\n({name})"])
    ax.set_ylabel("Response Length (tokens)")
    ax.set_title(f"Task 1 – Response Length Distribution [{name}]")
    _savefig(fig, plots_dir / "12_length_dist.png", "length distribution")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description="Task 1 – Generate all result plots")
    ap.add_argument("--config", default="configs/dpo.yaml")
    ap.add_argument("--run-name", default="standard",
                    help="Run name to use for single-run plots (train log, eval rows)")
    args = ap.parse_args()

    cfg      = load_yaml(args.config)
    res_dir  = repo_path(cfg["results_dir"])
    plots_dir = res_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    name = args.run_name

    print(f"\n[plot_results] Generating all Task 1 plots → {plots_dir}/\n")

    # ── Training curves ───────────────────────────────────────────────────────
    train_log = _load_jsonl(res_dir / f"train_log_{name}.jsonl")
    if train_log:
        plot_train_loss(train_log, plots_dir)
        plot_train_pref_acc(train_log, plots_dir)
        plot_train_lr(train_log, plots_dir)

    # ── β ablation ────────────────────────────────────────────────────────────
    beta_summary = _load_json(res_dir / "beta_ablation_summary.json")
    if beta_summary:
        plot_beta_metric(
            beta_summary, "reward_mean",
            "Mean Reward Score", "Task 1 – β Ablation: Reward Score",
            "04_beta_reward.png", plots_dir, higher_is_better=True,
        )
        plot_beta_metric(
            beta_summary, "kl_mean",
            "Sampled KL Divergence", "Task 1 – β Ablation: KL from Reference",
            "05_beta_kl.png", plots_dir, higher_is_better=False,
        )
        plot_beta_metric(
            beta_summary, "preference_accuracy",
            "Held-out Preference Accuracy",
            "Task 1 – β Ablation: Preference Accuracy",
            "06_beta_pref_acc.png", plots_dir, higher_is_better=True,
        )
        plot_beta_heatmap(beta_summary, plots_dir)

    # ── Length analysis ───────────────────────────────────────────────────────
    length_analysis = _load_json(res_dir / "length_analysis.json")
    if length_analysis:
        plot_length_strat(
            length_analysis, "reward_mean",
            "Mean Reward Score",
            "Task 1 – Length Study: Per-Stratum Reward",
            "08_length_strat_reward.png", plots_dir,
        )
        plot_length_strat(
            length_analysis, "preference_accuracy",
            "Preference Accuracy",
            "Task 1 – Length Study: Per-Stratum Preference Accuracy",
            "09_length_strat_pref_acc.png", plots_dir,
        )

    # ── Eval rows (length–reward correlation + length distribution) ───────────
    eval_rows = _load_jsonl(res_dir / f"{name}_eval_rows.jsonl")
    if eval_rows:
        plot_length_reward_corr(eval_rows, name, plots_dir)
        plot_length_distribution(eval_rows, name, plots_dir)

    # ── Word-limit compliance ─────────────────────────────────────────────────
    wl_results = _load_json(res_dir / f"{name}_word_limit.json")
    if wl_results:
        plot_word_limit(wl_results, plots_dir)

    print(f"\n[plot_results] Done. All plots saved to: {plots_dir}/\n")


if __name__ == "__main__":
    main()
