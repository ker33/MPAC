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

# ==========================================
# 介入超参数配置
# ==========================================
ALPHA_T = 8.0      # 文本域联想头 (SAH) 抑制强度 (值越大，对联想脑补头的抑制越深)
ALPHA_V = 8.0      # 跨模态域联想头 (VSAH) 抑制强度
MIN_SCALE = 0.2    # 保底比例 (防止过度剪枝破坏基本语言组织能力)

# 控制评测样本量：None 代表全量 3000 个样本；设为 300 等数值用于快速调试
LIMIT_SAMPLES = None 

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
# 步骤 1: 筛选单 Token 有效词 (概念库初始化)
# ==========================================
print(f"Loading processor and tokenizer from: {MODEL_ID}...")
processor = AutoProcessor.from_pretrained(MODEL_ID, local_files_only=True)
tokenizer = processor.tokenizer

with open("data/coco_config.json", "r", encoding="utf-8") as f:
    coco_data = json.load(f)

raw_words = list(set(coco_data["objects"] + coco_data["contexts"]))

valid_concept_to_id = {}
for word in raw_words:
    tokens_no_space = tokenizer.encode(word, add_special_tokens=False)
    tokens_with_space = tokenizer.encode(" " + word, add_special_tokens=False)
    if len(tokens_no_space) == 1:
        valid_concept_to_id[word] = tokens_no_space[0]
    elif len(tokens_with_space) == 1:
        valid_concept_to_id[word] = tokens_with_space[0]

valid_concepts = list(valid_concept_to_id.keys())
C = len(valid_concepts)
print(f"Validated {C} / {len(raw_words)} concepts as 100% single-token representations.")


# ==========================================
# 步骤 2: 加载 LLaVA 模型并定位组件
# ==========================================
print(f"Loading LLaVA-1.5-7B (Adaptive Calibration)...")
model = LlavaForConditionalGeneration.from_pretrained(
    MODEL_ID, 
    torch_dtype=torch.float16, 
    device_map="auto", 
    local_files_only=True
)

if hasattr(model, "language_model"):
    llm = model.language_model
elif hasattr(model, "model") and hasattr(model.model, "language_model"):
    llm = model.model.language_model
else:
    raise AttributeError("Could not find language model component.")

if hasattr(model, "vision_tower"):
    vision_tower = model.vision_tower
elif hasattr(model, "model") and hasattr(model.model, "vision_tower"):
    vision_tower = model.model.vision_tower
else:
    raise AttributeError("Could not find vision tower component.")

if hasattr(model, "multi_modal_projector"):
    projector = model.multi_modal_projector
elif hasattr(model, "model") and hasattr(model.model, "multi_modal_projector"):
    projector = model.model.multi_modal_projector
else:
    raise AttributeError("Could not find multi-modal projector component.")

d_model = llm.config.hidden_size
num_layers = llm.config.num_hidden_layers
num_heads = llm.config.num_attention_heads
d_head = d_model // num_heads


# ==========================================
# 步骤 3: 提取特征字典 (E_vis, E_txt & U_txt)
# ==========================================
image_dir = "data/images"
E_vis_list = []
embed_layer = model.get_input_embeddings()

@torch.no_grad()
def extract_clean_visual_feature(image_path, model, processor, vision_tower, projector):
    image = Image.open(image_path).convert("RGB")
    inputs = processor(images=image, return_tensors="pt").to(device)
    vision_outputs = vision_tower(inputs.pixel_values, output_hidden_states=True)
    image_features = vision_outputs.last_hidden_state[:, 1:, :]
    projected_features = projector(image_features)
    grid_features = projected_features.view(1, 24, 24, d_model)
    center_features = grid_features[:, 5:19, 5:19, :]
    mean_feature = center_features.reshape(1, -1, d_model).mean(dim=1)
    return mean_feature.squeeze(0)

print("Constructing E_vis, E_txt and U_txt...")
for word in valid_concepts:
    concept_path = os.path.join(image_dir, word)
    if os.path.isdir(concept_path):
        features_for_word = []
        for file_name in os.listdir(concept_path):
            if file_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                img_path = os.path.join(concept_path, file_name)
                try:
                    feat = extract_clean_visual_feature(img_path, model, processor, vision_tower, projector)
                    features_for_word.append(feat.unsqueeze(0))
                except Exception:
                    pass
        if len(features_for_word) > 0:
            feature = torch.cat(features_for_word, dim=0).mean(dim=0)
        else:
            token_id = torch.tensor([valid_concept_to_id[word]], dtype=torch.long, device=device)
            feature = embed_layer(token_id).detach().squeeze(0)
    elif os.path.exists(concept_path + ".png"):
        feature = extract_clean_visual_feature(concept_path + ".png", model, processor, vision_tower, projector)
    elif os.path.exists(concept_path + ".jpg"):
        feature = extract_clean_visual_feature(concept_path + ".jpg", model, processor, vision_tower, projector)
    else:
        token_id = torch.tensor([valid_concept_to_id[word]], dtype=torch.long, device=device)
        feature = embed_layer(token_id).detach().squeeze(0)
    E_vis_list.append(feature.unsqueeze(0))

E_vis = torch.cat(E_vis_list, dim=0).to(device)
E_vis_norm = E_vis / E_vis.norm(dim=-1, keepdim=True)

token_ids_tensor = torch.tensor([valid_concept_to_id[w] for w in valid_concepts], dtype=torch.long, device=device)
E_txt = embed_layer(token_ids_tensor).detach()
E_txt_norm = E_txt / E_txt.norm(dim=-1, keepdim=True)

unembed_layer = model.get_output_embeddings()
U_txt = unembed_layer.weight[token_ids_tensor].detach()
U_txt_norm = U_txt / U_txt.norm(dim=-1, keepdim=True)


# ==========================================
# 步骤 4: 参数映射得分扫描 (计算联想矩阵)
# ==========================================
text_dominance_scores = np.zeros((num_layers, num_heads))
vision_dominance_scores = np.zeros((num_layers, num_heads))

if hasattr(llm, "model") and hasattr(llm.model, "layers"):
    layers_list = llm.model.layers
elif hasattr(llm, "layers"):
    layers_list = llm.layers
else:
    raise AttributeError("Could not find decoder layers.")

mask_diag = torch.eye(C, device=device)
mask_off_diag = 1.0 - mask_diag

print("\nScanning attention heads for dominance metrics...")
for l in tqdm(range(num_layers), desc="Scanning Layers"):
    attn_layer = layers_list[l].self_attn
    W_V_full = attn_layer.v_proj.weight.detach().to(device)
    W_O_full = attn_layer.o_proj.weight.detach().to(device)
    
    for h in range(num_heads):
        W_V_head = W_V_full[h * d_head : (h + 1) * d_head, :]
        W_O_head = W_O_full[:, h * d_head : (h + 1) * d_head]
        W_OV = torch.matmul(W_V_head.t(), W_O_head.t())
        
        # Text-to-Text Domain Score
        proj_txt = torch.matmul(E_txt_norm, W_OV)
        proj_txt_norm = proj_txt / (proj_txt.norm(dim=-1, keepdim=True) + 1e-8)
        M_t2t = torch.matmul(proj_txt_norm, U_txt_norm.t())
        
        # Vision-to-Text Domain Score
        proj_vis = torch.matmul(E_vis_norm, W_OV)
        proj_vis_norm = proj_vis / (proj_vis.norm(dim=-1, keepdim=True) + 1e-8)
        M_v2t = torch.matmul(proj_vis_norm, U_txt_norm.t())
        
        text_dominance_scores[l, h] = M_t2t.diag().mean().item() - (M_t2t * mask_off_diag).sum().item() / (C * C - C)
        vision_dominance_scores[l, h] = M_v2t.diag().mean().item() - (M_v2t * mask_off_diag).sum().item() / (C * C - C)


# ==========================================
# 步骤 5: 动态钩子注入 (Adaptive Intervention Setup)
# ==========================================
print("\n" + "="*50)
print("--- [POPE ADAPTIVE INTERVENTION SETUP] ---")
print(f"Injecting pre-hooks (ALPHA_T={ALPHA_T}, ALPHA_V={ALPHA_V}, MIN_SCALE={MIN_SCALE})")

def make_scaling_pre_hook(head_scales):
    def hook(module, args):
        x = args[0]
        orig_shape = x.shape
        hidden_size = orig_shape[-1]
        n_heads = head_scales.shape[0]
        h_dim = hidden_size // n_heads
        
        x_reshaped = x.view(-1, n_heads, h_dim)
        scales = head_scales.view(1, n_heads, 1).to(x.device).to(x.dtype)
        x_scaled = x_reshaped * scales
        
        return (x_scaled.view(orig_shape),)
    return hook

active_hooks = []
intervened_heads_count = 0

for l in range(num_layers):
    attn_layer = layers_list[l].self_attn
    o_proj = attn_layer.o_proj
    
    layer_scales = torch.ones(num_heads, dtype=torch.float32)
    for h in range(num_heads):
        t_score = text_dominance_scores[l, h]
        v_score = vision_dominance_scores[l, h]
        
        scale = 1.0
        # 如果是文本联想脑补头 (SAH, 分数表现为负)，对其进行抑制
        if t_score < 0:
            scale = min(scale, 1.0 + ALPHA_T * t_score)
        # 如果是视觉联想脑补头 (VSAH, 分数表现为负)，对其进行抑制
        if v_score < 0:
            scale = min(scale, 1.0 + ALPHA_V * v_score)
            
        scale = max(MIN_SCALE, scale)
        layer_scales[h] = scale
        
        if scale < 1.0:
            intervened_heads_count += 1
            
    hook_fn = make_scaling_pre_hook(layer_scales)
    handle = o_proj.register_forward_pre_hook(hook_fn)
    active_hooks.append(handle)

print(f"Adaptive pre-hooks successfully active. Intervened heads: {intervened_heads_count} / {num_layers * num_heads}")
print("="*50 + "\n")


# ==========================================
# 步骤 6: 循环一键运行三个子集
# ==========================================
splits = ["adversarial", "popular", "random"]
all_split_results = {}

for split in splits:
    pope_file = f"data/pope/coco_pope_{split}.json"
    print("\n" + "="*60)
    print(f"Starting Adaptive Evaluation on Split: [{split.upper()}]")
    print("="*60)
    
    if not os.path.exists(pope_file):
        print(f"Error: {pope_file} not found. Skipping.")
        continue
        
    pope_data = []
    with open(pope_file, "r", encoding="utf-8") as f:
        try:
            pope_data = [json.loads(line) for line in f]
        except Exception:
            f.seek(0)
            pope_data = json.load(f)

    if LIMIT_SAMPLES is not None:
        pope_data = pope_data[:LIMIT_SAMPLES]
        print(f"Evaluating a subset of {LIMIT_SAMPLES} samples...")
    else:
        print(f"Evaluating full dataset of {len(pope_data)} samples...")

    gts = []
    preds = []

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

    # 指标计算
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
    
    all_split_results[split] = split_results

    # 保存单个子集的自适应结果
    os.makedirs("results/pope", exist_ok=True)
    save_path = f"results/pope/pope_results_adaptive_{split}_010.json"
    with open(save_path, "w") as rf:
        json.dump(split_results, rf, indent=4)
    print(f"Results for [{split}] saved to '{save_path}'")

# ==========================================
# 步骤 7: 卸载钩子 (还原模型状态)
# ==========================================
for hook in active_hooks:
    hook.remove()
print("\nActive intervention hooks successfully removed.")

# ==========================================
# 步骤 8: 打印三数据集对比汇总表
# ==========================================
print("\n" + "="*70)
print(f"POPE ADAPTIVE EVALUATION SUMMARY TABLE (Total Evaluated: {LIMIT_SAMPLES if LIMIT_SAMPLES else 'FULL 3000'})")
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