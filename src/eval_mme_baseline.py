import torch

# ==========================================
# 临时兼容性修复 (Mock Patch) 与随机种子固化
# ==========================================
if not hasattr(torch, "float8_e8m0fnu"):
    setattr(torch, "float8_e8m0fnu", torch.uint8)

import os
import json
import random
import argparse
import numpy as np
from PIL import Image
from tqdm import tqdm
from transformers import AutoTokenizer, LlavaForConditionalGeneration, AutoProcessor

# ==========================================
# 1. 解析参数与参数初始化
# ==========================================
parser = argparse.ArgumentParser()
parser.add_argument('--experiment', type=str, default='llava_baseline', help='Name of the baseline experiment output file')
args = parser.parse_args()

MODEL_ID = "models/llava-1.5-7b-hf"

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==========================================
# 2. 载入 Tokenizer、Processor 与 LLaVA 模型
# ==========================================
print(f"Loading processor and tokenizer from: {MODEL_ID}...")
processor = AutoProcessor.from_pretrained(MODEL_ID, local_files_only=True)
tokenizer = processor.tokenizer

print(f"Loading LLaVA-1.5-7B (HF official baseline model) from: {MODEL_ID}...")
model = LlavaForConditionalGeneration.from_pretrained(
    MODEL_ID, 
    torch_dtype=torch.float16, 
    device_map="auto", 
    local_files_only=True
)

# ==========================================
# 3. 执行 MME 原向 Baseline 推理循环
# ==========================================
mme_jsonl_path = "data/MME/llava_mme.jsonl"
mme_images_dir = "data/MME/MME_Benchmark_release_version"
output_jsonl_dir = "data/MME/answers"
os.makedirs(output_jsonl_dir, exist_ok=True)
output_jsonl_path = os.path.join(output_jsonl_dir, f"{args.experiment}.jsonl")

print(f"Loading MME questions from: {mme_jsonl_path}")
with open(mme_jsonl_path, "r", encoding="utf-8") as f:
    questions = [json.loads(line) for line in f]

print(f"Running baseline inference on MME, outputting to: {output_jsonl_path}")
with open(output_jsonl_path, "w", encoding="utf-8") as out_f:
    for item in tqdm(questions, desc="MME Baseline Inference"):
        img_name_with_cat = item["image"]
        text = item["text"]
        
        img_path = os.path.join(mme_images_dir, img_name_with_cat)
        if not os.path.exists(img_path):
            img_path = os.path.splitext(img_path)[0] + ".jpg"
            if not os.path.exists(img_path):
                continue
                
        prompt = f"USER: <image>\n{text}\nASSISTANT:"
        image = Image.open(img_path).convert("RGB")
        inputs = processor(text=prompt, images=image, return_tensors="pt").to(device)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=16,  # 保持与自适应解码长度完全一致
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id
            )
            
        generated_ids = outputs[0][inputs.input_ids.shape[-1]:]
        pred_text = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
        
        out_f.write(json.dumps({
            "question_id": item["question_id"],
            "prompt": item["text"],
            "text": pred_text
        }) + "\n")

print(f"\nBaseline inference completed successfully.")