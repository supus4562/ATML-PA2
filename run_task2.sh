#!/bin/bash
set -e
echo "1. Running Standard PPO Continuation (20 updates)..."
python -m task2_ppo.continue_train --config configs/ppo.yaml --run-name standard

echo "2. Running Clipping Analysis..."
python -m task2_ppo.analyze_clipping --config configs/ppo.yaml

echo "3. Running KL Ablation..."
python -m task2_ppo.ablate_kl --config configs/ppo.yaml

echo "4. Evaluating Checkpoints..."
python -m task2_ppo.evaluate --config configs/ppo.yaml --adapter outputs/task2_ppo/standard --name standard
python -m task2_ppo.evaluate --config configs/ppo.yaml --adapter outputs/task2_ppo/clip_0.05 --name clip_0.05
python -m task2_ppo.evaluate --config configs/ppo.yaml --adapter outputs/task2_ppo/clip_0.2 --name clip_0.2
python -m task2_ppo.evaluate --config configs/ppo.yaml --adapter outputs/task2_ppo/clip_0.5 --name clip_0.5
python -m task2_ppo.evaluate --config configs/ppo.yaml --adapter outputs/task2_ppo/beta_0.0 --name beta_0.0
python -m task2_ppo.evaluate --config configs/ppo.yaml --adapter outputs/task2_ppo/beta_0.1 --name beta_0.1
python -m task2_ppo.evaluate --config configs/ppo.yaml --adapter outputs/task2_ppo/beta_0.2 --name beta_0.2

echo "Task 2 Done!"
echo '5. Generating Plots...'
python -m task2_ppo.plot_results
