import argparse
from common.data import load_yaml
from task3_grpo.continue_train import run_grpo

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/grpo.yaml")
    args = ap.parse_args()
    cfg = load_yaml(args.config)
    
    print("Fork updates:", cfg["fork_updates"])
    print("Running Canonical GRPO vs Dr. GRPO short continuations...")
    
    # Run canonical GRPO
    print("\n--- Canonical GRPO (loss_type='grpo') ---")
    run_grpo(
        config_path=args.config,
        output="outputs/task3_grpo/norm_grpo",
        updates=cfg["fork_updates"],
        loss_type="grpo",
        run_name="norm_grpo"
    )
    
    # Run Dr. GRPO
    print("\n--- Dr. GRPO (loss_type='dr_grpo') ---")
    run_grpo(
        config_path=args.config,
        output="outputs/task3_grpo/norm_dr_grpo",
        updates=cfg["fork_updates"],
        loss_type="dr_grpo",
        run_name="norm_dr_grpo"
    )

if __name__ == "__main__":
    main()
