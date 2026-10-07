import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import argparse

plt.rcParams.update({
    "font.size": 9,
    "axes.titlesize": 10,
    "axes.labelsize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "figure.dpi": 200,
    "lines.linewidth": 1.5,
    "lines.markersize": 4
})

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", default="results/task5_feedback")
    args = ap.parse_args()
    
    res_dir = Path(args.results_dir)
    plots_dir = res_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. In-domain vs Transfer Exact Accuracy
    gsm_file = res_dir / "summary_gsm.json"
    transfer_file = res_dir / "summary_transfer.json"
    
    if gsm_file.exists() and transfer_file.exists():
        with open(gsm_file) as f:
            gsm_data = json.load(f)
        with open(transfer_file) as f:
            transfer_data = json.load(f)
            
        policies = ["sft", "rlvr", "rlaif"]
        valid_policies = [p for p in policies if p in gsm_data and p in transfer_data]
        
        if valid_policies:
            gsm_acc = [gsm_data[p]["exact_accuracy"] for p in valid_policies]
            transfer_acc = [transfer_data[p]["exact_accuracy"] for p in valid_policies]
            
            x = np.arange(len(valid_policies))
            width = 0.35
            
            fig, ax = plt.subplots(figsize=(6, 4))
            rects1 = ax.bar(x - width/2, gsm_acc, width, label='GSM8K (In-Domain)')
            rects2 = ax.bar(x + width/2, transfer_acc, width, label='SVAMP (Transfer)')
            
            ax.set_ylabel('Exact Answer Accuracy')
            ax.set_title('Accuracy by Policy and Domain')
            ax.set_xticks(x)
            ax.set_xticklabels([p.upper() for p in valid_policies])
            ax.legend()
            
            plt.tight_layout()
            plt.savefig(plots_dir / "accuracy_comparison.pdf")
            plt.savefig(plots_dir / "accuracy_comparison.png")
            plt.close()

    # 2. Diagnostic Preferences (Stacked Bar Chart)
    diag_file = res_dir / "diagnostic_results.json"
    if diag_file.exists():
        with open(diag_file) as f:
            diag_data = json.load(f)
            
        categories = ["reasoning_sensitivity", "outcome_sensitivity", "filler_susceptibility", "distractor_robustness"]
        mechanisms = ["rlvr", "rlaif"]
        
        valid_mechs = [m for m in mechanisms if m in diag_data]
        if valid_mechs:
            fig, axes = plt.subplots(1, len(valid_mechs), figsize=(5 * len(valid_mechs), 4), sharey=True)
            if len(valid_mechs) == 1:
                axes = [axes]
                
            for ax, mech in zip(axes, valid_mechs):
                better = [diag_data[mech][cat]["better_rate"] for cat in categories]
                tie = [diag_data[mech][cat]["tie_rate"] for cat in categories]
                worse = [diag_data[mech][cat]["worse_rate"] for cat in categories]
                
                x = np.arange(len(categories))
                
                ax.bar(x, better, label='Prefers Better', color='green', alpha=0.7)
                ax.bar(x, tie, bottom=better, label='Ties', color='gray', alpha=0.7)
                ax.bar(x, worse, bottom=np.array(better)+np.array(tie), label='Prefers Worse', color='red', alpha=0.7)
                
                ax.set_title(f"{mech.upper()} Preferences")
                ax.set_xticks(x)
                ax.set_xticklabels([c.replace('_', '\n').title() for c in categories], rotation=45, ha='right')
                if ax == axes[0]:
                    ax.set_ylabel('Fraction of Pairs')
                    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1))
            
            plt.tight_layout()
            plt.savefig(plots_dir / "diagnostic_preferences.pdf")
            plt.savefig(plots_dir / "diagnostic_preferences.png")
            plt.close()

if __name__ == "__main__":
    main()
