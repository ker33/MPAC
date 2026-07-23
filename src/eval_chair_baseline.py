# src/eval_chair_baseline.py

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

# 控制评测样本量：默认抽取 100 张生成描述进行对比
LIMIT_SAMPLES = 500

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
# 步骤 2: 读取图像池并抽取 100 张图像执行路径自检
# ==========================================
pope_file = "data/pope/coco_pope_adversarial.json"
print(f"\nLoading image index pool from: {pope_file}")

pope_data = []
with open(pope_file, "r", encoding="utf-8") as f:
    try:
        pope_data = [json.loads(line) for line in f]
    except Exception:
        f.seek(0)
        pope_data = json.load(f)
    
# 按图像 ID 去重并提取前 100 张
unique_images = []
seen = set()
for item in pope_data:
    img_name = item.get("image", item.get("image_source", ""))
    if img_name and img_name not in seen:
        seen.add(img_name)
        img_id = int(img_name.split("_")[-1].split(".")[0])
        unique_images.append((img_name, img_id))
    if len(unique_images) >= LIMIT_SAMPLES:
        break

# === 【环境自检与路径诊断】 ===
print(f"Loaded {len(unique_images)} unique images for CHAIR.")
if len(unique_images) > 0:
    test_img = unique_images[0][0]
    expected_path = f"data/coco/val2014/{test_img}"
    path_exists = os.path.exists(expected_path)
    print("\n" + "="*50)
    print("--- [CHAIR BASELINE PATH DIAGNOSTIC] ---")
    print(f"First image name: '{test_img}'")
    print(f"Checking expected path: '{expected_path}' -> Exists? : {path_exists}")
    if not path_exists:
        print("\n[ALERT] Image not found! Checking local file tree:")
        if os.path.exists("data/coco"):
            print(f"- Content inside 'data/coco/': {os.listdir('data/coco')}")
    print("="*50 + "\n")


# ==========================================
# 步骤 3: 运行官方原生模型生成循环
# ==========================================
chair_outputs = []

print("Generating captions with Baseline Official Model...")
for img_name, img_id in tqdm(unique_images):
    img_path = f"data/coco/val2014/{img_name}"
    if not os.path.exists(img_path):
        continue
        
    # 学术长句提问
    prompt = "USER: <image>\nPlease describe this image in detail.\nASSISTANT:"
    image = Image.open(img_path).convert("RGB")
    inputs = processor(text=prompt, images=image, return_tensors="pt").to(device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=128,      # 学术标准的 64 词上限
            do_sample=False,        # Greedy 确保与干预版的基础解码状态对齐
            pad_token_id=tokenizer.pad_token_id
        )
    
    # 过滤输入部分，解码输出
    generated_ids = outputs[0][inputs.input_ids.shape[-1]:]
    generated_caption = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
    
    # 移除提前截断保护，允许生成完整的多句描述段落
    chair_outputs.append({
        "image_id": img_id,
        "caption": generated_caption
    })


# ==========================================
# 步骤 4: 保存结果
# ==========================================
os.makedirs("results", exist_ok=True)
output_path = "results/chair/chair_outputs_baseline.json"
with open(output_path, "w") as f:
    json.dump(chair_outputs, f, indent=4)
print(f"Baseline generated captions saved successfully to '{output_path}'.")