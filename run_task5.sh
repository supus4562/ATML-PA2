#!/bin/bash
set -e

echo "=== Task 5: Evaluate Math (GSM8K) ==="
python -m task5_feedback.evaluate_math --dataset gsm

echo "=== Task 5: Score Perturbations ==="
python -m task5_feedback.score_perturbations

echo "=== Task 5: Evaluate Math (Transfer) ==="
python -m task5_feedback.evaluate_math --dataset transfer

echo "=== Task 5: Compare Feedback ==="
python -m task5_feedback.compare_feedback

echo "=== Task 5: Generating Plots ==="
python -m task5_feedback.plot_results
