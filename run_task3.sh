#!/bin/bash
set -e
echo "1. Running Standard GRPO Continuation (20 updates)..."
python -m task3_grpo.continue_train --config configs/grpo.yaml --run-name standard

echo "2. Running Group-Size Analysis..."
python -m task3_grpo.analyze_group_size --config configs/grpo.yaml

echo "3. Running Length-Normalization Study..."
python -m task3_grpo.compare_normalization --config configs/grpo.yaml

echo "4. Evaluating Checkpoints..."
python -m task3_grpo.evaluate --config configs/grpo.yaml --adapter outputs/task3_grpo/standard --name standard
python -m task3_grpo.evaluate --config configs/grpo.yaml --adapter outputs/task3_grpo/norm_grpo --name norm_grpo
python -m task3_grpo.evaluate --config configs/grpo.yaml --adapter outputs/task3_grpo/norm_dr_grpo --name norm_dr_grpo

echo "Task 3 Done!"
echo '5. Generating Plots...'
python -m task3_grpo.plot_results
