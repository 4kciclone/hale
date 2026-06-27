#!/bin/bash
set -e
echo "=== HALE Phase 1 ==="
echo "Step 1: Main benchmarks"
python experiments/phase1_main.py
echo "Step 2: delta sweep"
python experiments/sweep_delta.py
echo "Step 3: Depth ablation"
python experiments/ablation_depth.py
echo "=== Phase 1 Complete ==="
echo "Results: ./results/hale_phase1_results.json"
