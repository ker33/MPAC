import torch

# ==========================================
# 临时兼容性修复 (Mock Patch)
# ==========================================
if not hasattr(torch, "float8_e8m0fnu"):
    setattr(torch, "float8_e8m0fnu", torch.uint8)

import os
import json
import random
import types
import numpy as np
import pandas as pd  
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
from tqdm import tqdm
from scipy.stats import gaussian_kde  # 用于计算局部概率密度
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
print(f"Loading LLaVA-1.5-7B model with Eager Attention...")
model = LlavaForConditionalGeneration.from_pretrained(
    MODEL_ID, 
    torch_dtype=torch.float16, 
    device_map="auto", 
    local_files_only=True,
    attn_implementation="eager"  
)

if hasattr(model, "language_model"):
    llm = model.language_model
elif hasattr(model, "model") and hasattr(model.model, "language_model"):
    llm = model.model.language_model
else:
    raise AttributeError("Could not find the language model component.")

d_model = llm.config.hidden_size
num_layers = llm.config.num_hidden_layers
num_heads = llm.config.num_attention_heads
d_head = d_model // num_heads


# ==========================================
# 步骤 3: 快速支配度计算以匹配对应头
# ==========================================
embed_layer = model.get_input_embeddings()
unembed_layer = model.get_output_embeddings()

token_ids_tensor = torch.tensor([valid_concept_to_id[w] for w in valid_concepts], dtype=torch.long)
E_txt = embed_layer(token_ids_tensor.to(embed_layer.weight.device)).detach().to(device)
E_txt_norm = E_txt / E_txt.norm(dim=-1, keepdim=True)
E_vis_norm = E_txt_norm.clone()

U_txt = unembed_layer.weight[token_ids_tensor.to(unembed_layer.weight.device)].detach().to(device)
U_txt_norm = U_txt / U_txt.norm(dim=-1, keepdim=True)

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

print("\nScanning baseline attention heads mapping...")
for l in range(num_layers):
    attn_layer = layers_list[l].self_attn
    W_V_full = attn_layer.v_proj.weight.detach().to(device)
    W_O_full = attn_layer.o_proj.weight.detach().to(device)
    
    for h in range(num_heads):
        W_V_head = W_V_full[h * d_head : (h + 1) * d_head, :]
        W_O_head = W_O_full[:, h * d_head : (h + 1) * d_head]
        W_OV = torch.matmul(W_V_head.t(), W_O_head.t())
        
        proj_txt = torch.matmul(E_txt_norm, W_OV)
        proj_txt_norm = proj_txt / (proj_txt.norm(dim=-1, keepdim=True) + 1e-8)
        M_t2t = torch.matmul(proj_txt_norm, U_txt_norm.t())
        
        proj_vis = torch.matmul(E_vis_norm, W_OV)
        proj_vis_norm = proj_vis / (proj_vis.norm(dim=-1, keepdim=True) + 1e-8)
        M_v2t = torch.matmul(proj_vis_norm, U_txt_norm.t())
        
        text_dominance_scores[l, h] = M_t2t.diag().mean().item() - (M_t2t * mask_off_diag).sum().item() / (C * C - C)
        vision_dominance_scores[l, h] = M_v2t.diag().mean().item() - (M_v2t * mask_off_diag).sum().item() / (C * C - C)


# ==========================================
# 步骤 4: 锁定各类别中最顶尖的 K 个代表注意力头
# ==========================================
K = 20  
print(f"\nIdentifying top {K} representative heads for each class...")

target_heads = {
    'ICH': [],
    'VTH': [],
    'VSAH': [],
    'SAH': []
}

ich_flat = np.argsort(text_dominance_scores.flatten())[::-1][:K]
for idx in ich_flat:
    target_heads['ICH'].append((idx // num_heads, idx % num_heads))

vth_flat = np.argsort(vision_dominance_scores.flatten())[::-1][:K]
for idx in vth_flat:
    target_heads['VTH'].append((idx // num_heads, idx % num_heads))

sah_flat = np.argsort(text_dominance_scores.flatten())[:K]
for idx in sah_flat:
    target_heads['SAH'].append((idx // num_heads, idx % num_heads))

vsah_flat = np.argsort(vision_dominance_scores.flatten())[:K]
for idx in vsah_flat:
    target_heads['VSAH'].append((idx // num_heads, idx % num_heads))


# ==========================================
# 步骤 5: 动态捕获真实测试图片的注意力信息熵
# ==========================================
print("\nPreparing test images for real-world inference...")
test_images = []
for candidate_dir in ["data/images", "data/coco/val2014"]:
    if os.path.exists(candidate_dir):
        for f in os.listdir(candidate_dir):
            if f.lower().endswith(('.png', '.jpg', '.jpeg')):
                test_images.append(os.path.join(candidate_dir, f))
            if len(test_images) >= 5: 
                break
    if len(test_images) >= 5:
        break

if len(test_images) == 0:
    raise FileNotFoundError("Could not find any test images.")

entropy_records = {
    'ICH': [],
    'VTH': [],
    'VSAH': [],
    'SAH': []
}

print(f"Running generative inference on {len(test_images)} real images...")
for img_path in tqdm(test_images, desc="Tracking Entropy"):
    prompt = "USER: <image>\nPlease describe this image in detail.\nASSISTANT:"
    image = Image.open(img_path).convert("RGB")
    inputs = processor(text=prompt, images=image, return_tensors="pt").to(device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=10, 
            output_attentions=True,
            return_dict_in_generate=True,
            pad_token_id=tokenizer.pad_token_id
        )
    
    for step_attn in outputs.attentions:
        for cat_name, head_list in target_heads.items():
            for layer_idx, head_idx in head_list:
                p_dist = step_attn[layer_idx][0, head_idx, -1, :] 
                p_dist = p_dist[p_dist > 1e-8]
                shannon_entropy = -torch.sum(p_dist * torch.log(p_dist)).item()
                entropy_records[cat_name].append(shannon_entropy)


# ==========================================
# 步骤 6: 绘制【学术标准版】高斯散射小提琴图 (细节极致对齐版)
# ==========================================
print("Plotting standard academic Violin-Box Plot...")
plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
plt.style.use('seaborn-v0_8-whitegrid')

labels = ['ICH', 'VTH', 'VSAH', 'SAH']
#colors = ['#2ca02c', '#1f77b4', '#aec7e8', '#ff7f0e']
colors = ['#BFDFD2', '#4198AC', '#ECB66C', '#ED8D5A']
#colors = ['#fc757b', '#faa26f', '#b0d6a9', '#3c9bc9']

df_list = []
for cat in labels:
    for val in entropy_records[cat]:
        df_list.append({
            "Category": cat,
            "Entropy": val
        })
df_entropy = pd.DataFrame(df_list)

# 提取 Y 轴真实上限
flat_all_data = df_entropy["Entropy"].values
y_max = max(flat_all_data) if len(flat_all_data) > 0 else 5.2

fig, ax = plt.subplots(figsize=(8.5, 6), dpi=300)

# 1. 绘制纯净的外层小提琴图底座 (width=0.65 保持舒适的间距)
sns.violinplot(
    data=df_entropy,
    x="Category",
    y="Entropy",
    hue="Category",
    order=labels,
    palette=colors,
    inner=None,            # 不使用默认箱线图，手动控制内部线层
    width=0.65,            
    linewidth=1.2,
    density_norm="width",  
    legend=False,
    ax=ax
)

for collection in ax.collections:
    collection.set_alpha(0.8)
    collection.set_zorder(2)

# 2. 绘制自适应密度散射散点 (zorder=3)
for i, cat in enumerate(labels):
    y = np.array(entropy_records[cat])
    
    # 动态计算高斯核密度
    kde_func = gaussian_kde(y)
    local_densities = kde_func(y)
    
    # 高斯晕染散射系数
    norm_stds = local_densities / (local_densities.max() + 1e-8) * 0.05
    gaussian_offsets = np.random.normal(loc=0.0, scale=norm_stds, size=len(y))
    x_coords = i + gaussian_offsets
    
    # 绘制散射点
    ax.scatter(x_coords, y, alpha=1, color=colors[i], edgecolors='none', s=15, zorder=3)

# 3. 手动重塑极简箱线图结构并强制置顶 [1.1.2]
for i, cat in enumerate(labels):
    y = np.array(entropy_records[cat])
    q1, q3 = np.percentile(y, [25, 75])
    med = np.median(y)
    iqr = q3 - q1
    
    # 按照标准 Tukey 箱线图算法计算上下须线边界 (1.5 * IQR 规则)
    lower_whisker = max(q1 - 1.5 * iqr, y.min())
    upper_whisker = min(q3 + 1.5 * iqr, y.max())
    
    # A. 绘制垂直中心细须线 (线层 zorder=10)
    ax.plot([i, i], [lower_whisker, upper_whisker], color='#444444', linewidth=1.1, zorder=10)
    
    # B. 绘制上下界限短横线 (Caps, zorder=10)
    ax.plot([i - 0.05, i + 0.05], [lower_whisker, lower_whisker], color='#444444', linewidth=1.2, zorder=10)
    ax.plot([i - 0.05, i + 0.05], [upper_whisker, upper_whisker], color='#444444', linewidth=1.2, zorder=10)
    
    # C. 绘制垂直 IQR 粗体黑色矩形条 (线层 zorder=20)
    ax.plot([i, i], [q1, q3], color='#333333', linewidth=6.5, zorder=20)
    
    # D. 绘制中位数白色实心圆点 (圆点 zorder=30)
    ax.scatter(i, med, color='white', edgecolor='#222222', s=35, linewidths=1.2, zorder=30)

# 4. 紧凑型排版（底轴 0 起点，无白空隙）
ax.set_xticklabels(labels, fontsize=11, fontweight='bold')
ax.set_ylabel('Attention Entropy', fontsize=12, fontweight='bold', labelpad=12)
ax.set_xlabel('Category', fontsize=12, fontweight='bold', labelpad=10)

# 【核心修改】：通过提升上限到 +1.20，为顶部的学术文本引线留出绝对充裕的“安全天花板”，防止被长须穿透
ax.set_ylim(0, y_max + 1.20)

# 5. 【核心修改】：上抬双栏悬浮页眉（将高度由原来的 +0.16 提至 +0.45），彻底杜绝与下层小提琴顶部须线的干涉 [5]
feather_box = dict(boxstyle='round,pad=0.25', fc='white', alpha=0.65, ec='none')

ax.text(
    0.5, y_max + 0.45, 
    '← Global Grounding & Broad Retrieval\n(Distributed Attention)', 
    fontsize=9.5, color='#1b7837', ha='center', style='italic', fontweight='bold',
    bbox=feather_box, zorder=10
)

ax.text(
    2.5, y_max + 0.45, 
    'Localized Bias & Sharp Association →\n(Overfocused Attention)', 
    fontsize=9.5, color='#b2182b', ha='center', style='italic', fontweight='bold',
    bbox=feather_box, zorder=10
)

# 保存大图
output_dir = "results"
os.makedirs(output_dir, exist_ok=True)
output_path = f"{output_dir}/heads_attention_entropy_comparison-7-color2-ds改.png"
plt.savefig(output_path, dpi=300, bbox_inches='tight')
plt.close()

print(f"Empirical attention entropy comparison plot successfully saved to: {output_path}")