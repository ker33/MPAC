import torch

# ==========================================
# 临时兼容性修复 (Mock Patch)
# ==========================================
if not hasattr(torch, "float8_e8m0fnu"):
    setattr(torch, "float8_e8m0fnu", torch.uint8)

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
from tqdm import tqdm
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
print(f"Validated {C} concepts.")


# ==========================================
# 步骤 2: 加载 LLaVA 模型并提取共享组件 (多卡兼容)
# ==========================================
print(f"Loading LLaVA-1.5-7B model...")
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
# 步骤 3: 提取多卡安全的数据表示 (E_vis, E_txt & U_txt)
# ==========================================
image_dir = "data/images"
E_vis_list = []
embed_layer = model.get_input_embeddings()

@torch.no_grad()
def extract_clean_visual_feature(image_path, model, processor, vision_tower, projector):
    image = Image.open(image_path).convert("RGB")
    inputs = processor(images=image, return_tensors="pt")
    
    vt_device = next(vision_tower.parameters()).device
    pixel_values = inputs.pixel_values.to(device=vt_device, dtype=torch.float16)
    vision_outputs = vision_tower(pixel_values, output_hidden_states=True)
    image_features = vision_outputs.last_hidden_state[:, 1:, :]
    
    proj_device = next(projector.parameters()).device
    projected_features = projector(image_features.to(proj_device))
    
    grid_features = projected_features.view(1, 24, 24, d_model)
    center_features = grid_features[:, 5:19, 5:19, :]
    mean_feature = center_features.reshape(1, -1, d_model).mean(dim=1)
    return mean_feature.squeeze(0).to(device)

print("Constructing visual concept dictionary E_vis...")
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
            feature = embed_layer(token_id.to(embed_layer.weight.device)).detach().squeeze(0).to(device)
    elif os.path.exists(concept_path + ".png"):
        feature = extract_clean_visual_feature(concept_path + ".png", model, processor, vision_tower, projector)
    elif os.path.exists(concept_path + ".jpg"):
        feature = extract_clean_visual_feature(concept_path + ".jpg", model, processor, vision_tower, projector)
    else:
        token_id = torch.tensor([valid_concept_to_id[word]], dtype=torch.long, device=device)
        feature = embed_layer(token_id.to(embed_layer.weight.device)).detach().squeeze(0).to(device)
    E_vis_list.append(feature.unsqueeze(0))

E_vis = torch.cat(E_vis_list, dim=0).to(device)
E_vis_norm = E_vis / E_vis.norm(dim=-1, keepdim=True)

# 动态多卡安全解嵌索引
token_ids_tensor = torch.tensor([valid_concept_to_id[w] for w in valid_concepts], dtype=torch.long)
E_txt = embed_layer(token_ids_tensor.to(embed_layer.weight.device)).detach().to(device)
E_txt_norm = E_txt / E_txt.norm(dim=-1, keepdim=True)

unembed_layer = model.get_output_embeddings()
U_txt = unembed_layer.weight[token_ids_tensor.to(unembed_layer.weight.device)].detach().to(device)
U_txt_norm = U_txt / U_txt.norm(dim=-1, keepdim=True)


# ==========================================
# 步骤 4: 计算对立对称矩阵支配度得分
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

print("\nScanning attention heads inside Vicuna backbone...")
for l in tqdm(range(num_layers), desc="Scanning"):
    attn_layer = layers_list[l].self_attn
    W_V_full = attn_layer.v_proj.weight.detach().to(device)
    W_O_full = attn_layer.o_proj.weight.detach().to(device)
    
    for h in range(num_heads):
        W_V_head = W_V_full[h * d_head : (h + 1) * d_head, :]
        W_O_head = W_O_full[:, h * d_head : (h + 1) * d_head]
        W_OV = torch.matmul(W_V_head.t(), W_O_head.t())
        
        # 1. 文本-文本支配度
        proj_txt = torch.matmul(E_txt_norm, W_OV)
        proj_txt_norm = proj_txt / (proj_txt.norm(dim=-1, keepdim=True) + 1e-8)
        M_t2t = torch.matmul(proj_txt_norm, U_txt_norm.t())
        
        # 2. 跨模态域支配度
        proj_vis = torch.matmul(E_vis_norm, W_OV)
        proj_vis_norm = proj_vis / (proj_vis.norm(dim=-1, keepdim=True) + 1e-8)
        M_v2t = torch.matmul(proj_vis_norm, U_txt_norm.t())
        
        text_dominance_scores[l, h] = M_t2t.diag().mean().item() - (M_t2t * mask_off_diag).sum().item() / (C * C - C)
        vision_dominance_scores[l, h] = M_v2t.diag().mean().item() - (M_v2t * mask_off_diag).sum().item() / (C * C - C)


# ==========================================
# 步骤 5: 筛选 4 类功能头与数据映射
# ==========================================
print("\nCategorizing salient attention heads...")
K = 80 

ich_indices = np.argsort(text_dominance_scores.flatten())[::-1][:K]
ich_layers = ich_indices // num_heads

vth_indices = np.argsort(vision_dominance_scores.flatten())[::-1][:K]
vth_layers = vth_indices // num_heads

sah_indices = np.argsort(text_dominance_scores.flatten())[:K]
sah_layers = sah_indices // num_heads

vsah_indices = np.argsort(vision_dominance_scores.flatten())[:K]
vsah_layers = vsah_indices // num_heads

# 数据索引映射映射表
categories = ['VTH', 'VSAH', 'SAH', 'ICH']
data_dict = {
    'VTH': vth_layers,
    'VSAH': vsah_layers,
    'SAH': sah_layers,
    'ICH': ich_layers
}


# ==========================================
# 步骤 6: 绘制高度紧凑、自适应边界的山脊图 (Ridge Plot)
# ==========================================
print("Plotting Layer-wise Kernel Density Ridge Plot (Optimized for compactness)...")
plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

#colors = ['#1f77b4', '#aec7e8', '#ff7f0e', '#2ca02c'] 
colors = ['#4198ac', '#ecb66c', '#ed8d5a', '#bfdfd2']

# 设置重叠度为 -0.32
fig, axes = plt.subplots(len(categories), 1, figsize=(9.5, 7.8), sharex=True, dpi=300)
plt.subplots_adjust(hspace=-0.32)

for i, (cat, color) in enumerate(zip(categories, colors)):
    ax = axes[i]
    cat_data = data_dict[cat]
    mean_val = cat_data.mean()
    
    # 首先绘制平滑的 KDE 密度曲线，将其渲染结果存入内存中
    sns.kdeplot(cat_data, fill=True, color=color, alpha=0.65, linewidth=1.5, edgecolor=color, ax=ax, zorder=2)
    
    # 【核心优化】：动态获取当前 KDE 曲线的绝对峰值高度
    if len(ax.lines) > 0:
        line = ax.lines[-1]
        x_kde, y_kde = line.get_data()
        peak_height = y_kde.max()
    else:
        peak_height = 0.08  # 边界兜底
    
    # 将背景带和均值虚线的截止高度，限制在略高于当前峰顶的 15% 处，消除上空大面积留白
    limit_height = peak_height * 1.15
    
    # 1. 动态绘制受限高度的自适应纵向三处理阶段背景带 (zorder=1)
    ax.fill_between([0, 10], 0, limit_height, color='#000000', alpha=0.015, zorder=1)   # 浅层
    ax.fill_between([10, 22], 0, limit_height, color='#000000', alpha=0.035, zorder=1)  # 中层
    ax.fill_between([22, 31], 0, limit_height, color='#000000', alpha=0.055, zorder=1)  # 深层
    
    # 2. 新增【受限高度的均值中心虚线】（使用普通的 plot 替代 axvline，避免冲破天际，zorder=4）
    ax.plot([mean_val, mean_val], [0, limit_height], color=color, linestyle='--', linewidth=1.4, alpha=0.90, zorder=4)
    
    # 3. 绘制真实注意力头分布 rug 刻度线 (zorder=3)
    sns.rugplot(cat_data, color=color, alpha=0.90, height=0.08, ax=ax, linewidth=1.2, zorder=3)
    
    # 轴刻度控制（LLaVA-1.5-7B 共 32 层）
    ax.set_xlim(0, 31)
    ax.set_xticks(range(0, 32, 4))
    
    # 【核心优化】：动态调整子轴的高限，确保布局饱满不空旷
    ax.set_ylim(0, limit_height * 1.25)
    ax.set_yticks([])
    ax.set_ylabel('')
    
    ax.patch.set_alpha(0.0)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    
    # 4. 在首个子图顶部精细标注各纵向语义层级分工
    if i == 0:
        ax.text(5, limit_height * 1.05, "Shallow Layers\n(Visual Perception)", fontsize=10, color='#666666', ha='center', style='italic', zorder=6)
        ax.text(16, limit_height * 1.05, "Middle Layers\n(Concept Association)", fontsize=10, color='#666666', ha='center', style='italic', zorder=6)
        ax.text(26.5, limit_height * 1.05, "Deep Layers\n(Syntax & Alignment)", fontsize=10, color='#666666', ha='center', style='italic', zorder=6)

    # 5. 【核心优化】：最左侧标签仅保留纯净、简洁的注意力头简称，消除杂乱的文字信息
    ax.text(-0.02, 0.20, cat, fontsize=11, fontweight='bold', color='#333333', ha='right', transform=ax.transAxes, zorder=6)
    
    # 分隔虚线美化
    if i < len(categories) - 1:
        ax.spines['bottom'].set_visible(False)
        ax.xaxis.grid(True, linestyle=':', alpha=0.4, color='#ccc')
    else:
        # 底轴边框线强化
        ax.spines['bottom'].set_color('#333333')
        ax.spines['bottom'].set_linewidth(1.3)
        ax.xaxis.grid(True, linestyle='--', alpha=0.5, color='#999999')

plt.xlabel('Decoder Layer Index (Transformer Depth)', fontsize=12, fontweight='bold', labelpad=12)
#plt.suptitle('Layer-wise Distribution and Division of Labor of Four Attention Head Classes', fontsize=13, fontweight='bold', y=0.96)

# 写入无损 png
output_dir = "results"
os.makedirs(output_dir, exist_ok=True)
output_path = f"{output_dir}/heads_layer_density_ridge-color1-final.png"
plt.savefig(output_path, dpi=300, bbox_inches='tight')
plt.close()

print(f"\nHighly detailed compact ridge plot successfully saved to: {output_path}")