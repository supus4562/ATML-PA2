#!/bin/bash
set -e

echo "=== Task 4: Generating Responses ==="
python -m task4_safety.generate_responses

echo "=== Task 4: Creating Manual Audit Sheet ==="
python -m task4_safety.make_audit_sheet

echo "=== Task 4: Judging Responses ==="
python -m task4_safety.judge_responses

echo "=== Task 4: Evaluating Safety ==="
python -m task4_safety.evaluate_safety
