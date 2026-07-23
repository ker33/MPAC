# src/find_vth.py

import torch

# ==========================================
# 临时兼容性修复 (Mock Patch)
# ==========================================
if not hasattr(torch, "float8_e8m0fnu"):
    setattr(torch, "float8_e8m0fnu", torch.uint8)

import os
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
from transformers import AutoTokenizer, LlavaForConditionalGeneration, AutoProcessor

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_ID = "models/llava-1.5-7b-hf"

# ==========================================
# 步骤 1: 筛选单 Token 的 COCO 词表
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
print(f"Validated {C} / {len(raw_words)} concepts as single-token representations.")


# ==========================================
# 步骤 2: 加载 LLaVA-1.5-7B 模型并动态兼容提取
# ==========================================
print(f"Loading LLaVA-1.5-7B model from local path: {MODEL_ID}...")
model = LlavaForConditionalGeneration.from_pretrained(
    MODEL_ID, 
    torch_dtype=torch.float16, 
    device_map="auto", 
    local_files_only=True
)

# 1. 动态提取内部 LLM 结构
if hasattr(model, "language_model"):
    llm = model.language_model
elif hasattr(model, "model") and hasattr(model.model, "language_model"):
    llm = model.model.language_model
else:
    raise AttributeError("Could not find the language model component.")

# 2. 动态提取内部 Vision Tower 结构
if hasattr(model, "vision_tower"):
    vision_tower = model.vision_tower
elif hasattr(model, "model") and hasattr(model.model, "vision_tower"):
    vision_tower = model.model.vision_tower
else:
    raise AttributeError("Could not find the vision tower component.")

# 3. 动态提取内部 Multi-modal Projector 结构
if hasattr(model, "multi_modal_projector"):
    projector = model.multi_modal_projector
elif hasattr(model, "model") and hasattr(model.model, "multi_modal_projector"):
    projector = model.model.multi_modal_projector
else:
    raise AttributeError("Could not find the multi-modal projector component.")

d_model = llm.config.hidden_size
num_layers = llm.config.num_hidden_layers
num_heads = llm.config.num_attention_heads
d_head = d_model // num_heads

print(f"Model properties detected: d_model={d_model}, layers={num_layers}, heads={num_heads}, d_head={d_head}")


# ==========================================
# 步骤 3: 提取视觉概念特征字典 (E_vis) —— 支持单图与多图文件夹
# ==========================================
image_dir = "data/images"
E_vis_list = []

embed_layer = model.get_input_embeddings()

@torch.no_grad()
def extract_clean_visual_feature(image_path, model, processor, vision_tower, projector):
    image = Image.open(image_path).convert("RGB")
    inputs = processor(images=image, return_tensors="pt").to(device)
    
    # 1. 提取 Vision Tower 特征
    vision_outputs = vision_tower(inputs.pixel_values, output_hidden_states=True)
    image_features = vision_outputs.last_hidden_state[:, 1:, :] # 丢弃 CLS token
    
    # 2. 映射到 LLM 空间
    projected_features = projector(image_features)
    
    # 3. 仅提取中间 14x14 区域，排除边缘白背景噪声
    grid_features = projected_features.view(1, 24, 24, d_model)
    center_features = grid_features[:, 5:19, 5:19, :]
    
    # 4. 空间平均
    mean_feature = center_features.reshape(1, -1, d_model).mean(dim=1)
    return mean_feature.squeeze(0)

print("Constructing visual concept dictionary E_vis...")
real_images_count = 0

for word in valid_concepts:
    concept_path = os.path.join(image_dir, word)
    
    # 情况 A：如果存在该类别的同名文件夹（多图模式）
    if os.path.isdir(concept_path):
        features_for_word = []
        for file_name in os.listdir(concept_path):
            if file_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                img_path = os.path.join(concept_path, file_name)
                try:
                    feat = extract_clean_visual_feature(img_path, model, processor, vision_tower, projector)
                    features_for_word.append(feat.unsqueeze(0))
                except Exception as e:
                    print(f"Error loading {img_path}: {e}")
                    
        if len(features_for_word) > 0:
            feature = torch.cat(features_for_word, dim=0).mean(dim=0)
            real_images_count += 1
            print(f"-> Extracted averaged visual feature for '{word}' from folder ({len(features_for_word)} images)")
        else:
            token_id = torch.tensor([valid_concept_to_id[word]], dtype=torch.long, device=device)
            feature = embed_layer(token_id).detach().squeeze(0)
            
    # 情况 B：如果存在单张图片（单图模式）
    elif os.path.exists(concept_path + ".png"):
        feature = extract_clean_visual_feature(concept_path + ".png", model, processor, vision_tower, projector)
        real_images_count += 1
        print(f"-> Extracted visual feature for '{word}' from single PNG image")
    elif os.path.exists(concept_path + ".jpg"):
        feature = extract_clean_visual_feature(concept_path + ".jpg", model, processor, vision_tower, projector)
        real_images_count += 1
        print(f"-> Extracted visual feature for '{word}' from single JPG image")
        
    # 情况 C：无图可用（退回文本兜底）
    else:
        token_id = torch.tensor([valid_concept_to_id[word]], dtype=torch.long, device=device)
        feature = embed_layer(token_id).detach().squeeze(0)
        
    E_vis_list.append(feature.unsqueeze(0))

E_vis = torch.cat(E_vis_list, dim=0) # [C, d_model]
E_vis_norm = E_vis / E_vis.norm(dim=-1, keepdim=True) # L2 归一化
print(f"E_vis constructed successfully. (Categories with real images: {real_images_count} / {C})")


# ==========================================
# 步骤 4: 提取文本预测解嵌矩阵 (U_txt)
# ==========================================
token_ids_tensor = torch.tensor([valid_concept_to_id[w] for w in valid_concepts], dtype=torch.long, device=device)
unembed_layer = model.get_output_embeddings()

U_txt = unembed_layer.weight[token_ids_tensor].detach()
U_txt_norm = U_txt / U_txt.norm(dim=-1, keepdim=True) # L2 归一化


# ==========================================
# 步骤 5: 逐个 Attention Head 计算翻译强度
# ==========================================
vth_scores = np.zeros((num_layers, num_heads))

# 动态定位自注意力层列表
if hasattr(llm, "model") and hasattr(llm.model, "layers"):
    layers_list = llm.model.layers
elif hasattr(llm, "layers"):
    layers_list = llm.layers
else:
    raise AttributeError("Could not find decoder layers list inside the language model.")

print("Calculating Visual-Semantic Translation Head (VTH) scores...")
for l in range(num_layers):
    attn_layer = layers_list[l].self_attn
    W_V_full = attn_layer.v_proj.weight.detach()
    W_O_full = attn_layer.o_proj.weight.detach()
    
    for h in range(num_heads):
        W_V_head = W_V_full[h * d_head : (h + 1) * d_head, :]
        W_O_head = W_O_full[:, h * d_head : (h + 1) * d_head]
        W_OV = torch.matmul(W_V_head.t(), W_O_head.t()) # [d_model, d_model]
        
        M_v2t = torch.matmul(torch.matmul(E_vis_norm, W_OV), U_txt_norm.t()) # [C, C]
        
        diag_mean = M_v2t.diag().mean().item()
        
        total_sum = M_v2t.sum().item()
        diag_sum = M_v2t.diag().sum().item()
        off_diag_mean = (total_sum - diag_sum) / (C * C - C)
        
        vth_scores[l, h] = diag_mean - off_diag_mean

print("Calculation completed.")


# ==========================================
# 步骤 6: 保存结果与绘制热力图
# ==========================================
output_dir = "results"
os.makedirs(output_dir, exist_ok=True)

plt.figure(figsize=(14, 9))
sns.heatmap(vth_scores, cmap="RdBu_r", center=0, cbar=True)
plt.title("Visual-Semantic Translation Heads (VTH) Distribution inside LLaVA-1.5-7B", fontsize=16)
plt.xlabel("Attention Head Index", fontsize=14)
plt.ylabel("Decoder Layer Index", fontsize=14)
plt.xticks(fontsize=10)
plt.yticks(fontsize=10)
plt.tight_layout()
plt.savefig(f"{output_dir}/vth_distribution.png", dpi=300)
plt.close()

# 找出得分最高的 Top 5 VTH
flat_indices = np.argsort(vth_scores.flatten())[::-1][:5]
print("\n" + "="*50)
print("TOP 5 VISUAL-SEMANTIC TRANSLATION HEADS (VTH) FOUND:")
print("="*50)
for idx in flat_indices:
    l = idx // num_heads
    h = idx % num_heads
    print(f"Layer {l:2d}, Head {h:2d} | VTH Score: {vth_scores[l, h]:.4f}")
print("="*50)
print(f"Heatmap has been successfully saved to '{output_dir}/vth_distribution.png'.")