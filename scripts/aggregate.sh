#!/bin/bash
# scripts/aggregate.sh

# Default to the newest eval if no argument is provided
TARGET_DIR=${1:-"results/qwen3vl_eval"}

echo "Aggregating stats for: $TARGET_DIR"
# export PYTHONPATH=$PYTHONPATH:.
python3 src/evaluation/aggregate_results.py "$TARGET_DIR"