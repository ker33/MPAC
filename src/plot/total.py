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
# 步骤 6: 统计分类结果并输出最强极性头
# ==========================================
print("\nPerforming multi-dimensional taxonomy calculations...")

# 1. 独立维度统计 (Binary Classification Filters)
ich_mask = text_dominance_scores > 0
sah_mask = text_dominance_scores <= 0
vth_mask = vision_dominance_scores > 0
vsah_mask = vision_dominance_scores <= 0

ich_count = np.sum(ich_mask)
sah_count = np.sum(sah_mask)
vth_count = np.sum(vth_mask)
vsah_count = np.sum(vsah_mask)

# 2. 2D 状态空间象限统计 (Mutually Exclusive Quadrants)
gp_mask = ich_mask & vth_mask    # 第一象限: Grounded Propagators (ICH & VTH)
vd_mask = sah_mask & vth_mask    # 第二象限: Visual Divergence (SAH & VTH)
cc_mask = ich_mask & vsah_mask   # 第四象限: Context Copying (ICH & VSAH)
ha_mask = sah_mask & vsah_mask   # 第三象限: Hallucination Abyss (SAH & VSAH)

gp_count = np.sum(gp_mask)
vd_count = np.sum(vd_mask)
cc_count = np.sum(cc_mask)
ha_count = np.sum(ha_mask)

total_heads = num_layers * num_heads

# 输出统计面板
print("\n" + "="*80)
print("              CROSS-MODAL HEAD TAXONOMY DISTRIBUTION PANEL")
print("="*80)
print("1. INDEPENDENT BINARY DIMENSIONS:")
print(f"   * Identity Copying Heads (ICH, S_t2t > 0):  {ich_count:4d} / {total_heads} ({ich_count/total_heads*100:.2f}%)")
print(f"   * Semantic Association Heads (SAH, S_t2t <= 0): {sah_count:4d} / {total_heads} ({sah_count/total_heads*100:.2f}%)")
print(f"   * Visual Translation Heads (VTH, S_v2t > 0):    {vth_count:4d} / {total_heads} ({vth_count/total_heads*100:.2f}%)")
print(f"   * Visual Semantic Assoc. Heads (VSAH, S_v2t <= 0): {vsah_count:4d} / {total_heads} ({vsah_count/total_heads*100:.2f}%)")
print("-" * 80)
print("2. MUTUALLY EXCLUSIVE 2D STATE-SPACE QUADRANTS:")
print(f"   * Grounded Propagators (ICH & VTH):      {gp_count:4d} / {total_heads} ({gp_count/total_heads*100:.2f}%)")
print(f"   * Visual Divergence (SAH & VTH):         {vd_count:4d} / {total_heads} ({vd_count/total_heads*100:.2f}%)")
print(f"   * Context Copying (ICH & VSAH):          {cc_count:4d} / {total_heads} ({cc_count/total_heads*100:.2f}%)")
print(f"   * Hallucination Abyss (SAH & VSAH):      {ha_count:4d} / {total_heads} ({ha_count/total_heads*100:.2f}%)")
print("="*80)


# 定义排序与极性检测方法
def get_top_k_heads(scores, mask, descending=True, k=10):
    indices = np.argwhere(mask)
    if len(indices) == 0:
        return []
    paired = [((int(l), int(h)), float(scores[l, h])) for l, h in indices]
    # 对得分进行排序
    paired.sort(key=lambda x: x[1], reverse=descending)
    return paired[:k]

# 对四种分类进行 Top 10 注意力头的排序与输出
# 对于对偶正项（ICH / VTH），数值越大、越主导、越偏向对角线，其属性越强 (descending=True)
# 对于对偶负项（SAH / VSAH），数值负得越多、越代表偏离对角线进行联想与脑补 (descending=False)
categories = [
    ("Identity Copying Heads (ICH) [S_t2t > 0, Descending Sorted]", text_dominance_scores, ich_mask, True),
    ("Semantic Association Heads (SAH) [S_t2t <= 0, Ascending Sorted (Most Negative First)]", text_dominance_scores, sah_mask, False),
    ("Visual Translation Heads (VTH) [S_v2t > 0, Descending Sorted]", vision_dominance_scores, vth_mask, True),
    ("Visual Semantic Association Heads (VSAH) [S_v2t <= 0, Ascending Sorted (Most Negative First)]", vision_dominance_scores, vsah_mask, False)
]

print("\n" + "="*80)
print("              TOP 10 STRONGEST POLARITY HEADS FOR EACH CATEGORY")
print("="*80)

for name, scores, mask, desc_mode in categories:
    print(f"\n>>> Category: {name}")
    top_k = get_top_k_heads(scores, mask, descending=desc_mode, k=10)
    
    if not top_k:
        print("   No attention heads found matching this category's filter criteria.")
    else:
        for rank, ((l, h), score) in enumerate(top_k, 1):
            print(f"   Rank {rank:2d} | Layer {l:2d}, Head {h:2d} | Score: {score:.6f}")
    print("-" * 80)

print("\nProcessing completed.")