#!/bin/bash
# 如果发生任何错误，立即退出脚本
set -e

EXPERIMENT_NAME="llava_adaptive_b013"

echo "========================================================="
echo "Step 1: Running Model Inference on MME Dataset..."
echo "========================================================="
# 在根目录下调用 python 推理生成原始原始 answers/llava_adaptive.jsonl
python src/eval_mme_adaptive.py --experiment $EXPERIMENT_NAME

echo "========================================================="
echo "Step 2: Switching to MME Directory & Parsing to TXT..."
echo "========================================================="
# 切换到 MME 目录运行格式转换
cd data/MME
python convert_answer_to_mme.py --experiment $EXPERIMENT_NAME

echo "========================================================="
echo "Step 3: Launching Official MME Evaluation Tool..."
echo "========================================================="
# 进入计算模块计算最终得分
cd eval_tool
python calculation.py --results_dir answers/$EXPERIMENT_NAME

echo "========================================================="
echo "MME Evaluation Flow Successfully Completed!"
echo "========================================================="