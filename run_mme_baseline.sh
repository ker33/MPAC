#!/bin/bash
set -e

EXPERIMENT_NAME="llava_baseline"

echo "========================================================="
echo "Step 1: Running Model Baseline Inference on MME Dataset..."
echo "========================================================="
python src/eval_mme_baseline.py --experiment $EXPERIMENT_NAME

echo "========================================================="
echo "Step 2: Switching to MME Directory & Parsing to TXT..."
echo "========================================================="
cd data/MME
python convert_answer_to_mme.py --experiment $EXPERIMENT_NAME

echo "========================================================="
echo "Step 3: Launching Official MME Evaluation Tool..."
echo "========================================================="
cd eval_tool
python calculation.py --results_dir answers/$EXPERIMENT_NAME

echo "========================================================="
echo "MME Baseline Evaluation Flow Successfully Completed!"
echo "========================================================="