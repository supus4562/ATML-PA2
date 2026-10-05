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
    
    for beta in cfg["kl_values"]:
        print(f"\n--- Running KL Beta {beta} ---")
        run_name = f"beta_{beta}"
        run_ppo(
            config_path=args.config,
            output=f"outputs/task2_ppo/{run_name}",
            updates=cfg["fork_updates"],
            kl_beta=beta,
            run_name=run_name
        )

if __name__ == "__main__":
    main()
