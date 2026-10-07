from __future__ import annotations

import argparse
from common.data import load_yaml
from task2_ppo.continue_train import run_ppo

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/ppo.yaml")
    args = ap.parse_args()
    cfg = load_yaml(args.config)
    print("KL beta conditions:", cfg["kl_values"])
    print("Fork update budget:", cfg["fork_updates"])
    
    import subprocess
    
    for beta in cfg["kl_values"]:
        print(f"\n--- Running KL Beta {beta} ---")
        run_name = f"beta_{beta}"
        cmd = [
            "python", "-m", "task2_ppo.continue_train",
            "--config", args.config,
            "--run-name", run_name,
            "--output", f"outputs/task2_ppo/{run_name}",
            "--updates", str(cfg["fork_updates"]),
            "--kl-beta", str(beta)
        ]
        subprocess.run(cmd, check=True)

if __name__ == "__main__":
    main()
