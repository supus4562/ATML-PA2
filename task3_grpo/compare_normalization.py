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
    
    import subprocess
    
    # Run canonical GRPO
    print("\n--- Canonical GRPO (loss_type='grpo') ---")
    cmd1 = [
        "python", "-m", "task3_grpo.continue_train",
        "--config", args.config,
        "--run-name", "norm_grpo",
        "--output", "outputs/task3_grpo/norm_grpo",
        "--updates", str(cfg["fork_updates"]),
        "--loss-type", "grpo"
    ]
    subprocess.run(cmd1, check=True)
    
    # Run Dr. GRPO
    print("\n--- Dr. GRPO (loss_type='dr_grpo') ---")
    cmd2 = [
        "python", "-m", "task3_grpo.continue_train",
        "--config", args.config,
        "--run-name", "norm_dr_grpo",
        "--output", "outputs/task3_grpo/norm_dr_grpo",
        "--updates", str(cfg["fork_updates"]),
        "--loss-type", "dr_grpo"
    ]
    subprocess.run(cmd2, check=True)

if __name__ == "__main__":
    main()
