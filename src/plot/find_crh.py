# src/find_crh.py

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
# 步骤 1: 读取冲突三元组并过滤单 Token
# ==========================================
print(f"Loading processor and tokenizer from: {MODEL_ID}...")
processor = AutoProcessor.from_pretrained(MODEL_ID, local_files_only=True)
tokenizer = processor.tokenizer

triplets_file = "data/conflict_triplets.json"
with open(triplets_file, "r", encoding="utf-8") as f:
    triplets_data = json.load(f)

conflict_triplets = triplets_data["triplets"]

raw_involved_words = []
for t in conflict_triplets:
    raw_involved_words.extend([t["context"], t["prior"], t["visual"]])
raw_involved_words = list(set(raw_involved_words))

word_to_token_id = {}
for word in raw_involved_words:
    tokens_no_space = tokenizer.encode(word, add_special_tokens=False)
    tokens_with_space = tokenizer.encode(" " + word, add_special_tokens=False)
    
    if len(tokens_no_space) == 1:
        word_to_token_id[word] = tokens_no_space[0]
    elif len(tokens_with_space) == 1:
        word_to_token_id[word] = tokens_with_space[0]

for t in conflict_triplets:
    for role in ["context", "prior", "visual"]:
        if t[role] not in word_to_token_id:
            raise ValueError(f"Word '{t[role]}' is not a single-token! Please adjust conflict_triplets.json.")

print(f"Validated all {len(raw_involved_words)} involved words in conflict triplets as single-token.")


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

if hasattr(model, "language_model"):
    llm = model.language_model
elif hasattr(model, "model") and hasattr(model.model, "language_model"):
    llm = model.model.language_model
else:
    raise AttributeError("Could not find the language model component.")

if hasattr(model, "vision_tower"):
    vision_tower = model.vision_tower
elif hasattr(model, "model") and hasattr(model.model, "vision_tower"):
    vision_tower = model.model.vision_tower
else:
    raise AttributeError("Could not find the vision tower component.")

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
# 步骤 3: 提取视觉概念特征字典 (E_vis) —— 多情况兼容
# ==========================================
image_dir = "data/images"
E_vis_dict = {}

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

print("Constructing visual concept dictionary E_vis for actual visual stimuli...")
real_images_count = 0
visual_words = list(set([t["visual"] for t in conflict_triplets]))

for word in visual_words:
    concept_path = os.path.join(image_dir, word)
    
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
            token_id = torch.tensor([word_to_token_id[word]], dtype=torch.long, device=device)
            feature = embed_layer(token_id).detach().squeeze(0)
            
    elif os.path.exists(concept_path + ".png"):
        feature = extract_clean_visual_feature(concept_path + ".png", model, processor, vision_tower, projector)
        real_images_count += 1
        print(f"-> Extracted visual feature for '{word}' from single PNG image")
    elif os.path.exists(concept_path + ".jpg"):
        feature = extract_clean_visual_feature(concept_path + ".jpg", model, processor, vision_tower, projector)
        real_images_count += 1
        print(f"-> Extracted visual feature for '{word}' from single JPG image")
    else:
        token_id = torch.tensor([word_to_token_id[word]], dtype=torch.long, device=device)
        feature = embed_layer(token_id).detach().squeeze(0)
        print(f"-> Warning: Image for '{word}' not found, fallback to text embedding.")
        
    E_vis_dict[word] = feature / feature.norm(dim=-1, keepdim=True)

print(f"E_vis dictionary constructed. (Real images used: {real_images_count} / {len(visual_words)})")


# ==========================================
# 步骤 4: 提取文本嵌入 (E_txt) 与预测解嵌 (U_txt)
# ==========================================
E_txt_dict = {}
U_txt_dict = {}

unembed_layer = model.get_output_embeddings()

for word in raw_involved_words:
    token_id = torch.tensor([word_to_token_id[word]], dtype=torch.long, device=device)
    
    e_val = embed_layer(token_id).detach().squeeze(0)
    E_txt_dict[word] = e_val / e_val.norm(dim=-1, keepdim=True)
    
    u_val = unembed_layer.weight[token_id].detach().squeeze(0)
    U_txt_dict[word] = u_val / u_val.norm(dim=-1, keepdim=True)


# ==========================================
# 步骤 5: 逐个 Attention Head 计算冲突解析得分 (基于投影余弦相似度)
# ==========================================
crh_scores = np.zeros((num_layers, num_heads))

if hasattr(llm, "model") and hasattr(llm.model, "layers"):
    layers_list = llm.model.layers
elif hasattr(llm, "layers"):
    layers_list = llm.layers
else:
    raise AttributeError("Could not find decoder layers list inside the language model.")

print("Calculating Conflict Resolution Head (CRH) scores (Cosine Alignment)...")
for l in range(num_layers):
    attn_layer = layers_list[l].self_attn
    W_V_full = attn_layer.v_proj.weight.detach()
    W_O_full = attn_layer.o_proj.weight.detach()
    
    for h in range(num_heads):
        W_V_head = W_V_full[h * d_head : (h + 1) * d_head, :]
        W_O_head = W_O_full[:, h * d_head : (h + 1) * d_head]
        W_OV = torch.matmul(W_V_head.t(), W_O_head.t()) # [d_model, d_model]
        
        triplet_scores = []
        for item in conflict_triplets:
            context = item["context"]
            prior = item["prior"]
            vis = item["visual"]
            
            e_vis = E_vis_dict[vis]     # 实际视觉输入向量
            e_ctx = E_txt_dict[context] # 文本上下文输入向量
            u_vis = U_txt_dict[vis]     # 视觉真实词预测向量
            u_pri = U_txt_dict[prior]   # 常识脑补词预测向量
            
            # --- 核心改进：计算带有归一化的投影余弦相似度 ---
            # 1. 投影视觉特征并做 L2 归一化
            proj_vis = torch.matmul(e_vis, W_OV)
            proj_vis_norm = proj_vis / (proj_vis.norm(dim=-1, keepdim=True) + 1e-8)
            # 计算视觉真实预测的对齐度（余弦相似度）
            cos_vis = torch.dot(proj_vis_norm, u_vis).item()
            
            # 2. 投影文本上下文并做 L2 归一化
            proj_ctx = torch.matmul(e_ctx, W_OV)
            proj_ctx_norm = proj_ctx / (proj_ctx.norm(dim=-1, keepdim=True) + 1e-8)
            # 计算常识幻觉预测的对齐度（余弦相似度）
            cos_ctx = torch.dot(proj_ctx_norm, u_pri).item()
            
            # 3. 冲突解析能力评分 = 视觉翻译对齐度 - 文本常识对齐度
            triplet_scores.append(cos_vis - cos_ctx)
            
        crh_scores[l, h] = np.mean(triplet_scores)

print("Calculation completed.")


# ==========================================
# 步骤 6: 保存结果与绘制热力图 (设置合理的固定范围以过滤噪声)
# ==========================================
output_dir = "results"
os.makedirs(output_dir, exist_ok=True)

plt.figure(figsize=(14, 9))
# 固定相似度差值展示区间 [-0.2, 0.2]，使得微弱的随机扰动不会被染成深色
sns.heatmap(crh_scores, cmap="RdBu_r", center=0, vmin=-0.15, vmax=0.15, cbar=True)
plt.title("Conflict Resolution Heads (CRH) Distribution inside LLaVA-1.5-7B (Optimized)", fontsize=16)
plt.xlabel("Attention Head Index", fontsize=14)
plt.ylabel("Decoder Layer Index", fontsize=14)
plt.xticks(fontsize=10)
plt.yticks(fontsize=10)
plt.tight_layout()
plt.savefig(f"{output_dir}/crh_distribution.png", dpi=300)
plt.close()

# 找出得分最高的 Top 5 CRH
flat_indices = np.argsort(crh_scores.flatten())[::-1][:5]
print("\n" + "="*50)
print("TOP 5 OPTIMIZED CONFLICT RESOLUTION HEADS (CRH):")
print("="*50)
for idx in flat_indices:
    l = idx // num_heads
    h = idx % num_heads
    print(f"Layer {l:2d}, Head {h:2d} | CRH Cosine Score: {crh_scores[l, h]:.4f}")
print("="*50)
print(f"Heatmap has been successfully saved to '{output_dir}/crh_distribution.png'.")