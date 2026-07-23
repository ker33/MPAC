# src/eval_pope_baseline.py

import torch

# ==========================================
# 临时兼容性修复 (Mock Patch) 与随机种子固化
# ==========================================
if not hasattr(torch, "float8_e8m0fnu"):
    setattr(torch, "float8_e8m0fnu", torch.uint8)

import os
import json
import random
import numpy as np
from PIL import Image
from tqdm import tqdm
from transformers import AutoTokenizer, LlavaForConditionalGeneration, AutoProcessor

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed(42) # 固化随机性

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_ID = "models/llava-1.5-7b-hf"

# 控制评测样本量：设为 None 代表跑完每个子集全量 3000 个样本；设为 300 用于快速调试
LIMIT_SAMPLES = None 

# ==========================================
# 步骤 1: 加载处理器与官方原生 LLaVA 模型
# ==========================================
print(f"Loading processor and tokenizer...")
processor = AutoProcessor.from_pretrained(MODEL_ID, local_files_only=True)
tokenizer = processor.tokenizer

print(f"Loading LLaVA-1.5-7B (Baseline Official Model)...")
model = LlavaForConditionalGeneration.from_pretrained(
    MODEL_ID, 
    torch_dtype=torch.float16, 
    device_map="auto", 
    local_files_only=True
)

# ==========================================
# 步骤 2: 循环一键运行三个子集
# ==========================================
splits = ["adversarial", "popular", "random"]
all_split_results = {}

for split in splits:
    pope_file = f"data/pope/coco_pope_{split}.json"
    print("\n" + "="*60)
    print(f"Starting Evaluation on Split: [{split.upper()}]")
    print("="*60)
    
    if not os.path.exists(pope_file):
        print(f"Error: {pope_file} not found. Skipping this split.")
        continue
        
    pope_data = []
    with open(pope_file, "r", encoding="utf-8") as f:
        try:
            pope_data = [json.loads(line) for line in f]
        except Exception:
            f.seek(0)
            pope_data = json.load(f)

    # 路径诊断自检（仅在第一个子集的第一条数据上输出，防止刷屏）
    if len(pope_data) > 0 and split == "adversarial":
        first_item = pope_data[0]
        print("--- [BASEPAD PATH CHECK] ---")
        img_name = first_item.get("image", first_item.get("image_source", ""))
        expected_path = f"data/coco/val2014/{img_name}"
        print(f"Sample Image Path: '{expected_path}' -> Exists? : {os.path.exists(expected_path)}")
        print("-"*30 + "\n")

    # 样本数量截取
    if LIMIT_SAMPLES is not None:
        pope_data = pope_data[:LIMIT_SAMPLES]
        print(f"Evaluating a subset of {LIMIT_SAMPLES} samples...")
    else:
        print(f"Evaluating full dataset of {len(pope_data)} samples...")

    gts = []
    preds = []

    # 运行推理循环
    for item in tqdm(pope_data, desc=f"Processing {split}"):
        img_name = item.get("image", item.get("image_source", ""))
        question = item.get("question", item.get("query", item.get("text", "")))
        gt_ans = item.get("answer", item.get("label", ""))
        if isinstance(gt_ans, str):
            gt_ans = gt_ans.lower()
        
        if not img_name or not question or not gt_ans:
            continue
            
        img_path = f"data/coco/val2014/{img_name}"
        if not os.path.exists(img_path):
            continue
            
        prompt = f"USER: <image>\n{question}\nASSISTANT:"
        image = Image.open(img_path).convert("RGB")
        inputs = processor(text=prompt, images=image, return_tensors="pt").to(device)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=8,       
                do_sample=False,        
                pad_token_id=tokenizer.pad_token_id
            )
        
        generated_ids = outputs[0][inputs.input_ids.shape[-1]:]
        generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
        
        pred_ans = "no"
        if "yes" in generated_text.lower():
            pred_ans = "yes"
            
        gts.append(gt_ans)
        preds.append(pred_ans)

    # 统计单子集指标
    gts = np.array(gts)
    preds = np.array(preds)

    TP = np.sum((gts == "yes") & (preds == "yes"))
    TN = np.sum((gts == "no") & (preds == "no"))
    FP = np.sum((gts == "no") & (preds == "yes"))
    FN = np.sum((gts == "yes") & (preds == "no"))

    accuracy = (TP + TN) / len(gts) if len(gts) > 0 else 0
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    yes_ratio = np.sum(preds == "yes") / len(preds) if len(preds) > 0 else 0

    split_results = {
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1-Score": f1,
        "Yes-Ratio": yes_ratio,
        "Total Evaluated": len(gts)
    }
    
    # 记录到总字典
    all_split_results[split] = split_results

    # 保存单个子集到对应的独立 JSON 文件
    save_path = f"results/pope/pope_results_baseline_{split}.json"
    with open(save_path, "w") as rf:
        json.dump(split_results, rf, indent=4)
    print(f"Results for [{split}] saved to '{save_path}'")

# ==========================================
# 步骤 3: 打印三数据集对比汇总表
# ==========================================
print("\n" + "="*70)
print(f"POPE BASELINE EVALUATION SUMMARY TABLE (Total Evaluated: {LIMIT_SAMPLES if LIMIT_SAMPLES else 'FULL 3000'})")
print("="*70)
print(f"{'Split Name':<15} | {'Accuracy':<10} | {'Precision':<10} | {'Recall':<10} | {'F1-Score':<10} | {'Yes-Ratio':<10}")
print("-"*70)
for split in splits:
    if split in all_split_results:
        res = all_split_results[split]
        print(f"{split:<15} | "
              f"{res['Accuracy']*100:.2f}%    | "
              f"{res['Precision']*100:.2f}%   | "
              f"{res['Recall']*100:.2f}%   | "
              f"{res['F1-Score']*100:.2f}%   | "
              f"{res['Yes-Ratio']*100:.2f}%")
print("="*70)