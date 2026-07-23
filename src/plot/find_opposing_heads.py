# src/find_opposing_heads.py

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
# 步骤 1: 筛选配置中的单 Token 有效词
# ==========================================
print(f"Loading processor and tokenizer from: {MODEL_ID}...")
processor = AutoProcessor.from_pretrained(MODEL_ID, local_files_only=True)
tokenizer = processor.tokenizer

with open("data/coco_config.json", "r", encoding="utf-8") as f:
    coco_data = json.load(f)

# 仅合并 objects 与 contexts 两个分类
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
# 步骤 2: 加载 LLaVA-1.5-7B 模型并动态提取组件
# ==========================================
print(f"Loading LLaVA-1.5-7B model from: {MODEL_ID}...")
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


# ==========================================
# 步骤 3: 提取视觉概念特征字典 (E_vis) —— 兼容各种情况
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

print("Constructing visual concept dictionary E_vis...")
real_images_count = 0
for word in valid_concepts:
    concept_path = os.path.join(image_dir, word)
    
    # 兼容多图文件夹模式
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
        else:
            token_id = torch.tensor([valid_concept_to_id[word]], dtype=torch.long, device=device)
            feature = embed_layer(token_id).detach().squeeze(0)
            
    # 兼容单图 PNG 模式
    elif os.path.exists(concept_path + ".png"):
        feature = extract_clean_visual_feature(concept_path + ".png", model, processor, vision_tower, projector)
        real_images_count += 1
    # 兼容单图 JPG 模式
    elif os.path.exists(concept_path + ".jpg"):
        feature = extract_clean_visual_feature(concept_path + ".jpg", model, processor, vision_tower, projector)
        real_images_count += 1
    # 文本 Embedding 兜底模式
    else:
        token_id = torch.tensor([valid_concept_to_id[word]], dtype=torch.long, device=device)
        feature = embed_layer(token_id).detach().squeeze(0)
        
    E_vis_list.append(feature.unsqueeze(0))

E_vis = torch.cat(E_vis_list, dim=0)
E_vis_norm = E_vis / E_vis.norm(dim=-1, keepdim=True)
print(f"E_vis constructed. (Categories with images: {real_images_count} / {C})")


# ==========================================
# 步骤 4: 提取文本嵌入及解嵌矩阵 (E_txt & U_txt)
# ==========================================
token_ids_tensor = torch.tensor([valid_concept_to_id[w] for w in valid_concepts], dtype=torch.long, device=device)
E_txt = embed_layer(token_ids_tensor).detach()
E_txt_norm = E_txt / E_txt.norm(dim=-1, keepdim=True)

unembed_layer = model.get_output_embeddings()
U_txt = unembed_layer.weight[token_ids_tensor].detach()
U_txt_norm = U_txt / U_txt.norm(dim=-1, keepdim=True)


# ==========================================
# 步骤 5: 计算对立对称矩阵打分
# ==========================================
# 文本域打分：正值代表复制（ICH），负值代表语义脑补联想（SAH）
text_dominance_scores = np.zeros((num_layers, num_heads))
# 跨模态域打分：正值代表翻译（VTH），负值代表视觉关联脑补（VSAH）
vision_dominance_scores = np.zeros((num_layers, num_heads))

# 动态定位自注意力层列表
if hasattr(llm, "model") and hasattr(llm.model, "layers"):
    layers_list = llm.model.layers
elif hasattr(llm, "layers"):
    layers_list = llm.layers
else:
    raise AttributeError("Could not find decoder layers list inside the language model.")

mask_diag = torch.eye(C, device=device)
mask_off_diag = 1.0 - mask_diag

print("Scanning attention heads and calculating diagonal dominance metrics...")
for l in range(num_layers):
    attn_layer = layers_list[l].self_attn
    W_V_full = attn_layer.v_proj.weight.detach()
    W_O_full = attn_layer.o_proj.weight.detach()
    
    for h in range(num_heads):
        W_V_head = W_V_full[h * d_head : (h + 1) * d_head, :]
        W_O_head = W_O_full[:, h * d_head : (h + 1) * d_head]
        W_OV = torch.matmul(W_V_head.t(), W_O_head.t())
        
        # 1. 计算文本-文本余弦矩阵
        proj_txt = torch.matmul(E_txt_norm, W_OV)
        proj_txt_norm = proj_txt / (proj_txt.norm(dim=-1, keepdim=True) + 1e-8)
        M_t2t = torch.matmul(proj_txt_norm, U_txt_norm.t()) # [C, C]
        
        # 2. 计算视觉-文本余弦矩阵
        proj_vis = torch.matmul(E_vis_norm, W_OV)
        proj_vis_norm = proj_vis / (proj_vis.norm(dim=-1, keepdim=True) + 1e-8)
        M_v2t = torch.matmul(proj_vis_norm, U_txt_norm.t()) # [C, C]
        
        # 计算打分（对角线均值 - 非对角线均值）
        text_dominance_scores[l, h] = M_t2t.diag().mean().item() - (M_t2t * mask_off_diag).sum().item() / (C * C - C)
        vision_dominance_scores[l, h] = M_v2t.diag().mean().item() - (M_v2t * mask_off_diag).sum().item() / (C * C - C)

print("Scanning completed.")


# ==========================================
# 步骤 6: 绘制对立对称热力双图
# ==========================================
output_dir = "results"
os.makedirs(output_dir, exist_ok=True)

fig, axes = plt.subplots(1, 2, figsize=(24, 9))

# 统一的色彩轴范围（对角线主导度的余弦相似度区间）
vmin_val, vmax_val = -0.15, 0.15

# 图 1：文本域 [Text-to-Text] 
# 红色 (Positive) = 恒等复制头 (ICH) 
# 蓝色 (Negative) = 语义联想头 (SAH, 文本幻觉/联想源头)
sns.heatmap(text_dominance_scores, ax=axes[0], cmap="RdBu_r", center=0, vmin=vmin_val, vmax=vmax_val, cbar=True)
#axes[0].set_title("1. Text-to-Text Mapping Score\n[Red(+) -> Copying (ICH)  |  Blue(-) -> Association/Hallucination (SAH)]", fontsize=14)
axes[0].set_title("Text-to-Text Mapping Score", fontsize=24)
axes[0].set_xlabel("Attention Head Index",fontsize=22)
axes[0].set_ylabel("Decoder Layer Index",fontsize=22)

# ======================
# 修改 x 轴和 y 轴刻度字体大小
# ======================
axes[0].tick_params(axis='x', labelsize=13)      # x轴刻度字体大小
axes[0].tick_params(axis='y', labelsize=13)      # y轴刻度字体大小

# ======================
# 修改颜色条（colorbar）刻度字体大小
# ======================
cbar1 = axes[0].collections[0].colorbar
cbar1.ax.tick_params(labelsize=18)              # 色彩轴刻度字体大小

# 图 2：跨模态域 [Vision-to-Text]
# 红色 (Positive) = 视觉翻译头 (VTH)
# 蓝色 (Negative) = 视觉联想头 (VSAH, 视觉幻觉/联想源头)
sns.heatmap(vision_dominance_scores, ax=axes[1], cmap="RdBu_r", center=0, vmin=vmin_val, vmax=vmax_val, cbar=True)
#axes[1].set_title("2. Vision-to-Text Mapping Score\n[Red(+) -> Translation (VTH)  |  Blue(-) -> Association/Hallucination (VSAH)]", fontsize=14)
axes[1].set_title("Vision-to-Text Mapping Score", fontsize=24)
axes[1].set_xlabel("Attention Head Index",fontsize=22)
axes[1].set_ylabel("Decoder Layer Index",fontsize=22)

# ======================
# 修改 x 轴和 y 轴刻度字体大小
# ======================
axes[1].tick_params(axis='x', labelsize=13)      # x轴刻度字体大小
axes[1].tick_params(axis='y', labelsize=13)      # y轴刻度字体大小

# ======================
# 修改颜色条（colorbar）刻度字体大小
# ======================
cbar1 = axes[1].collections[0].colorbar
cbar1.ax.tick_params(labelsize=18)              # 色彩轴刻度字体大小

#plt.suptitle("Symmetric Red-Blue Analysis of Attention Heads inside LLaVA-1.5-7B", fontsize=18, y=0.96)
plt.tight_layout(rect=[0, 0, 1, 0.93])
plt.savefig(f"{output_dir}/opposing_heads_distribution-big.png", dpi=300)
plt.close()

# 打印极性代表头
def print_extremes(scores, name, pos_label, neg_label):
    flat_max = np.argmax(scores)
    flat_min = np.argmin(scores)
    l_max, h_max = flat_max // num_heads, flat_max % num_heads
    l_min, h_min = flat_min // num_heads, flat_min % num_heads
    print(f"- {name}:")
    print(f"  * Strongest {pos_label:<15}: Layer {l_max:2d}, Head {h_max:2d} | Score: {scores[l_max, h_max]:.4f}")
    print(f"  * Strongest {neg_label:<15}: Layer {l_min:2d}, Head {h_min:2d} | Score: {scores[l_min, h_min]:.4f}")

print("\n" + "="*60)
print("EXTREME POLARITY HEADS DETECTED:")
print("="*60)
print_extremes(text_dominance_scores, "Text-to-Text Domain", "Copying (ICH)", "Association (SAH)")
print_extremes(vision_dominance_scores, "Vision-to-Text Domain", "Translation (VTH)", "Association (VSAH)")
print("="*60)
print(f"Symmetric dual heatmap saved to '{output_dir}/opposing_heads_distribution-big.png'.")