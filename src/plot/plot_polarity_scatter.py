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
# 步骤 5: 绘制自适应边界极性散点分布图 (细节精调对齐版)
# ==========================================
print("\nPlotting Adaptive Dual-Domain Polarity Mapping Scatter Plot...")
plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
plt.style.use('seaborn-v0_8-white')

# 扁平化数据以绘制散点图
layers = np.repeat(np.arange(num_layers), num_heads)
heads = np.tile(np.arange(num_heads), num_layers)
x_data = text_dominance_scores.flatten()
y_data = vision_dominance_scores.flatten()

# 动态计算数据的边界范围，留出 10% 的安全边距，确保不被裁剪
max_val = max(
    np.abs(x_data).max(),
    np.abs(y_data).max(),
    0.05
) * 1.1

# 打印数值统计进行诊断
print("\n=== Data Statistics for Diagnostic ===")
print(f"Number of heads plotted: {len(x_data)}")
print(f"Text dominance scores range: {x_data.min():.4f} to {x_data.max():.4f}")
print(f"Vision dominance scores range: {y_data.min():.4f} to {y_data.max():.4f}")
print(f"Adaptive max value boundary calculated: {max_val:.4f}")
print("======================================\n")

fig, ax = plt.subplots(figsize=(10, 8.5), dpi=300)

# 使用 fill_between 动态填充自适应四象限学术底色
ax.fill_between([-max_val, 0], 0, max_val, color='#f7f7f7', alpha=0.9, zorder=1)   # 第二象限: 跨模态发散区
ax.fill_between([-max_val, 0], -max_val, 0, color='#e0ecf4', alpha=0.9, zorder=1)  # 第三象限: 脑补深渊区 (SAH & VSAH)
ax.fill_between([0, max_val], 0, max_val, color='#fff7ec', alpha=0.9, zorder=1)   # 第一象限: 忠实接地区 (VTH & ICH)
ax.fill_between([0, max_val], -max_val, 0, color='#fef0d9', alpha=0.4, zorder=1)  # 第四象限: 上下文维持区

# 绘制 1024 个自注意力头的状态散点，颜色由浅到深代表网络层深度
sc = ax.scatter(x_data, y_data, c=layers, cmap='plasma', s=35, alpha=0.85, 
                edgecolors='white', linewidths=0.5, zorder=5)

# 添加渐变色条
cbar = plt.colorbar(sc, ax=ax, pad=0.02)
cbar.set_label('Decoder Layer Index (Depth)', fontsize=16, fontweight='bold', labelpad=10)
cbar.ax.tick_params(labelsize=14)  # ← 修改色彩轴刻度字体大小

# 绘制中心轴参考线
ax.axhline(0, color='#333333', linewidth=1.2, linestyle='--', alpha=0.6, zorder=4)
ax.axvline(0, color='#333333', linewidth=1.2, linestyle='--', alpha=0.6, zorder=4)

# 【优化】：微调四象限标签文字坐标，向边缘安全区靠拢，避免与对角线上的点群碰撞
#ax.text(max_val * 0.45, max_val * 0.55, 'Grounded Region\n[VTH & ICH]', fontsize=16, fontweight='bold', color='#7f3b08', ha='center', zorder=6)
#ax.text(-max_val * 0.45, -max_val * 0.55, 'Hallucination Abyss\n[SAH & VSAH]', fontsize=16, fontweight='bold', color='#08519c', ha='center', zorder=6)
#ax.text(-max_val * 0.55, max_val * 0.65, 'Visual Divergence\n[VTH & SAH]', fontsize=16, color='#777777', ha='center', zorder=6)
#ax.text(max_val * 0.55, -max_val * 0.65, 'Context Copying\n[ICH & VSAH]', fontsize=16, color='#777777', ha='center', zorder=6)

# ======================
# 四象限标签 —— 定位在各自象限的中心
# ======================

# 计算每个象限的中心坐标
quadrant_center = max_val * 0.5  # 象限的几何中心距离原点的距离

# 第一象限（右上）：Grounded Region [VTH & ICH]
ax.text(quadrant_center, quadrant_center, 
        'Grounded Region\n[VTH & ICH]', 
        fontsize=14, fontweight='bold', color='#7f3b08', 
        ha='center', va='center', zorder=6)

# 第三象限（左下）：Hallucination Abyss [SAH & VSAH]
ax.text(-quadrant_center, -quadrant_center, 
        'Hallucination Abyss\n[SAH & VSAH]', 
        fontsize=14, fontweight='bold', color='#08519c', 
        ha='center', va='center', zorder=6)

# 第二象限（左上）：Visual Divergence [VTH & SAH]
ax.text(-quadrant_center, quadrant_center, 
        'Visual Divergence\n[VTH & SAH]', 
        fontsize=14, color='#777777', 
        ha='center', va='center', zorder=6)

# 第四象限（右下）：Context Copying [ICH & VSAH]
ax.text(quadrant_center, -quadrant_center, 
        'Context Copying\n[ICH & VSAH]', 
        fontsize=14, color='#777777', 
        ha='center', va='center', zorder=6)

# ==========================================
# 步骤 6: 自动定位四大最强极性头并高亮标注 (极精细对齐版)
# ==========================================
def annotate_extreme_head(score_matrix, x_arr, y_arr, find_max, label_text, color, xytext_offset, rad=-0.15):
    flat_idx = np.argmax(score_matrix) if find_max else np.argmin(score_matrix)
    l, h = flat_idx // num_heads, flat_idx % num_heads
    x, y = x_arr[flat_idx], y_arr[flat_idx]
    
    # 标注外圈粗圈点
    ax.scatter(x, y, s=90, facecolors='none', edgecolors=color, linewidths=2.2, zorder=8)
    
    # 绘制带自适应偏移的文本框和优雅指示角标
    ax.annotate(
        f"{label_text}\n(L{l}, H{h})",
        xy=(x, y),
        xytext=xytext_offset,
        textcoords='offset points',
        fontsize=9,
        fontweight='bold',
        bbox=dict(boxstyle='round,pad=0.35', fc='white', edgecolor='#e2e2e2', alpha=0.95),
        arrowprops=dict(arrowstyle="->", color=color, lw=1.5, connectionstyle=f"arc3,rad={rad}"),
        zorder=10
    )

# 【核心优化】：针对这四个极端代表点，为其量身配置偏移向量和弧度，彻底解决遮挡、冲突与出界
# 1. 最强视觉翻译头 (Max VTH): 位于右上角极点。向左上方（左上内侧）拉出引线，不与右侧 Colorbar 冲突
annotate_extreme_head(vision_dominance_scores, x_data, y_data, True, "Max VTH", "#7f3b08", (-65, 30), rad=-0.15)

# 2. 最强恒等复制头 (Max ICH): 位于右侧边缘。向左下方（左下内侧）拉出引线，完美错开 Max VTH 并不挡住散点
annotate_extreme_head(text_dominance_scores, x_data, y_data, True, "Max ICH", "#b30000", (-42, -60), rad=0.15)

# 3. 最强语义脑补头 (Max SAH): 位于左下。向左上方（左上外侧）拉出引线，彻底与 VSAH 错开
annotate_extreme_head(text_dominance_scores, x_data, y_data, False, "Max SAH", "#08519c", (-65, 35), rad=-0.15)

# 4. 最强视觉脑补头 (Max VSAH): 位于左下偏中。向右下方（右下内侧）拉出引线，实现完美的“剪刀差”排布
annotate_extreme_head(vision_dominance_scores, x_data, y_data, False, "Max VSAH", "#006d2c", (30, -80), rad=0.15)

# 设置轴限
ax.set_xlim(-max_val, max_val)
ax.set_ylim(-max_val, max_val)

ax.set_xlabel(r'Text-to-Text Mapping Score $M_{t2t}$', fontsize=16, fontweight='bold', labelpad=10)
ax.set_ylabel(r'Vision-to-Text Mapping Score $M_{v2t}$', fontsize=16, fontweight='bold', labelpad=10)
#ax.set_title('Polarity State-Space Mapping of All 1024 Attention Heads inside LLaVA-1.5-7B', fontsize=13, fontweight='bold', pad=15)

# ======================
# 修改 x 轴和 y 轴刻度字体大小
# ======================
ax.tick_params(axis='x', labelsize=14)      # x轴刻度字体大小
ax.tick_params(axis='y', labelsize=14)      # y轴刻度字体大小

# 保存图像
output_dir = "results"
os.makedirs(output_dir, exist_ok=True)
output_path = f"{output_dir}/attention_heads_polarity_map-big.png"

# 使用 bbox_inches='tight'，即使不调用 tight_layout 也能自适应无损保留边缘所有文字
plt.savefig(output_path, dpi=300, bbox_inches='tight')
plt.close()

print(f"Polarity scatter map successfully saved to: {output_path}")