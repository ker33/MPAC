# src/find_crh_dynamic.py

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

print(f"Validated all {len(raw_involved_words)} involved words as single-token.")


# ==========================================
# 步骤 2: 加载 LLaVA-1.5-7B 模型 (强制启用 eager 注意力)
# ==========================================
print(f"Loading LLaVA-1.5-7B model from local path: {MODEL_ID}...")
# 关键修复：显式设定 attn_implementation="eager" 以支持注意力矩阵提取
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
# 步骤 3: 提取静态词表 Embedding 矩阵 (E_txt & U_txt)
# ==========================================
embed_layer = model.get_input_embeddings()
unembed_layer = model.get_output_embeddings()

# ==========================================
# 步骤 4: 动态前向传播与注意力提取
# ==========================================
image_dir = "data/images"
dynamic_crh_scores = np.zeros((num_layers, num_heads))

# 动态定位自注意力层列表
if hasattr(llm, "model") and hasattr(llm.model, "layers"):
    layers_list = llm.model.layers
elif hasattr(llm, "layers"):
    layers_list = llm.layers
else:
    raise AttributeError("Could not find decoder layers list inside the language model.")

print("\nRunning dynamic activation-aware analysis on conflict triplets...")
valid_triplet_count = 0

for item in conflict_triplets:
    context = item["context"]
    prior = item["prior"]
    vis = item["visual"]
    
    concept_path = os.path.join(image_dir, vis)
    img_file = None
    if os.path.isdir(concept_path):
        for f_name in os.listdir(concept_path):
            if f_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                img_file = os.path.join(concept_path, f_name)
                break
    elif os.path.exists(concept_path + ".png"):
        img_file = concept_path + ".png"
    elif os.path.exists(concept_path + ".jpg"):
        img_file = concept_path + ".jpg"
        
    if img_file is None:
        continue
        
    valid_triplet_count += 1
    
    # 构建 LLaVA 经典 USER-ASSISTANT Prompt 格式
    prompt = f"USER: <image>\nOn the {context}, there is a\nASSISTANT:"
    
    image = Image.open(img_file).convert("RGB")
    inputs = processor(text=prompt, images=image, return_tensors="pt").to(device)
    
    # 运行前向传播
    with torch.no_grad():
        outputs = model(
            input_ids=inputs.input_ids,
            pixel_values=inputs.pixel_values,
            output_attentions=True,
            return_dict=True
        )
    
    attentions = outputs.attentions 
    
    input_ids = inputs.input_ids[0].cpu().tolist()
    seq_len = len(input_ids)
    
    # 视觉 Tokens 的分布（从索引 1 开始，共 576 个 Token）
    num_visual_tokens = 576
    vis_start, vis_end = 1, 1 + num_visual_tokens
    
    # 定位文本上下文 Token (如 "kitchen") 在展开后序列中的精确位置
    context_token_ids = tokenizer.encode(" " + context, add_special_tokens=False)
    ctx_token_id = context_token_ids[0]
    
    ctx_idx = None
    for idx, tid in enumerate(input_ids):
        if tid == ctx_token_id:
            # 加上视觉 Token 替换带来的物理偏移
            ctx_idx = idx - 1 + num_visual_tokens 
            break
    if ctx_idx is None:
        ctx_idx = seq_len - 2 + num_visual_tokens
        
    target_idx = -1 # 待预测位置的前一个 Token (即最后一个 token)
    
    # 提取静态视觉 Embedding (E_vis)
    image_inputs = processor(images=image, return_tensors="pt").to(device)
    with torch.no_grad():
        vision_outputs = vision_tower(image_inputs.pixel_values, output_hidden_states=True)
        image_features = vision_outputs.last_hidden_state[:, 1:, :]
        projected_features = projector(image_features)
    e_vis = projected_features.view(24, 24, d_model)[5:19, 5:19, :].reshape(-1, d_model).mean(dim=0)
    e_vis = e_vis / e_vis.norm(dim=-1, keepdim=True)
    
    # 提取上下文词文本特征
    ctx_id_tensor = torch.tensor([tokenizer.encode(" " + context, add_special_tokens=False)[0]], dtype=torch.long, device=device)
    e_ctx = embed_layer(ctx_id_tensor).detach().squeeze(0)
    e_ctx = e_ctx / e_ctx.norm(dim=-1, keepdim=True)
    
    # 提取预测词特征
    vis_id_tensor = torch.tensor([tokenizer.encode(" " + vis, add_special_tokens=False)[0]], dtype=torch.long, device=device)
    pri_id_tensor = torch.tensor([tokenizer.encode(" " + prior, add_special_tokens=False)[0]], dtype=torch.long, device=device)
    u_vis = unembed_layer.weight[vis_id_tensor].detach().squeeze(0)
    u_vis = u_vis / u_vis.norm(dim=-1, keepdim=True)
    u_pri = unembed_layer.weight[pri_id_tensor].detach().squeeze(0)
    u_pri = u_pri / u_pri.norm(dim=-1, keepdim=True)

    # 计算动态混合指标
    for l in range(num_layers):
        attn_layer = layers_list[l].self_attn
        W_V_full = attn_layer.v_proj.weight.detach()
        W_O_full = attn_layer.o_proj.weight.detach()
        
        layer_attn = attentions[l][0, :, target_idx, :].detach() 
        
        for h in range(num_heads):
            W_V_head = W_V_full[h * d_head : (h + 1) * d_head, :]
            W_O_head = W_O_full[:, h * d_head : (h + 1) * d_head]
            W_OV = torch.matmul(W_V_head.t(), W_O_head.t())
            
            # --- 关键改进：使用注意力总和（Sum）替代均值（Mean），解决视觉稀释问题 ---
            a_vis = layer_attn[h, vis_start:vis_end].sum().item()
            a_ctx = layer_attn[h, ctx_idx].item()
            
            # 投影并归一化以获得纯角度对齐度
            proj_vis = torch.matmul(e_vis, W_OV)
            proj_vis_norm = proj_vis / (proj_vis.norm(dim=-1, keepdim=True) + 1e-8)
            cos_vis = torch.dot(proj_vis_norm, u_vis).item()
            
            proj_ctx = torch.matmul(e_ctx, W_OV)
            proj_ctx_norm = proj_ctx / (proj_ctx.norm(dim=-1, keepdim=True) + 1e-8)
            cos_ctx = torch.dot(proj_ctx_norm, u_pri).item()
            
            # 动态门控综合评分：(视觉流总关注 * 视觉翻译力) - (上下文关注 * 常识幻觉力)
            triplet_score = (a_vis * cos_vis) - (a_ctx * cos_ctx)
            dynamic_crh_scores[l, h] += triplet_score

if valid_triplet_count > 0:
    dynamic_crh_scores /= valid_triplet_count
print(f"Dynamic analysis completed. Used {valid_triplet_count} active conflict cases.")


# ==========================================
# 步骤 5: 结果可视化 (D-CRH 专用)
# ==========================================
output_dir = "results"
os.makedirs(output_dir, exist_ok=True)

plt.figure(figsize=(14, 9))
# 此时注意力总和和文本注意力尺度完全对齐，可以安全地通过 center=0 来观察对比
sns.heatmap(dynamic_crh_scores, cmap="RdBu_r", center=0, cbar=True)
plt.title("Dynamic Conflict Resolution Heads (D-CRH) Distribution inside LLaVA-1.5-7B", fontsize=16)
plt.xlabel("Attention Head Index", fontsize=14)
plt.ylabel("Decoder Layer Index", fontsize=14)
plt.xticks(fontsize=10)
plt.yticks(fontsize=10)
plt.tight_layout()
plt.savefig(f"{output_dir}/dynamic_crh_distribution.png", dpi=300)
plt.close()

flat_indices = np.argsort(dynamic_crh_scores.flatten())[::-1][:5]
print("\n" + "="*50)
print("TOP 5 DYNAMIC CONFLICT RESOLUTION HEADS (D-CRH):")
print("="*50)
for idx in flat_indices:
    l = idx // num_heads
    h = idx % num_heads
    print(f"Layer {l:2d}, Head {h:2d} | Dynamic Score: {dynamic_crh_scores[l, h]:.6f}")
print("="*50)
print(f"Dynamic Heatmap saved to '{output_dir}/dynamic_crh_distribution.png'.")