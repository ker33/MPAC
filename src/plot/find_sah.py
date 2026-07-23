# src/find_sah.py

import torch

# ==========================================
# 临时兼容性修复 (Mock Patch)
# ==========================================
# 由于较新版 transformers 假设了 torch.float8_e8m0fnu 的存在，
# 而 PyTorch 2.5.1 尚未包含该类型，在此处将其安全 Mock 兼容
if not hasattr(torch, "float8_e8m0fnu"):
    setattr(torch, "float8_e8m0fnu", torch.uint8)

import os
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import AutoTokenizer, LlavaForConditionalGeneration

# ==========================================
# 配置参数
# ==========================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_ID = "models/llava-1.5-7b-hf"

# ==========================================
# 步骤 1: 筛选单 Token 的 COCO 类别词并构建语义关联矩阵
# ==========================================
print(f"Loading tokenizer from local path: {MODEL_ID}...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, local_files_only=True)

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
print(f"Validated {C} / {len(raw_words)} concepts as single-token representations in tokenizer.")

concept_to_idx = {word: i for i, word in enumerate(valid_concepts)}

R_semantic = torch.zeros((C, C), device=device)
priors_added_count = 0
for src, tgt in coco_data["priors"]:
    if src in concept_to_idx and tgt in concept_to_idx:
        idx_src = concept_to_idx[src]
        idx_tgt = concept_to_idx[tgt]
        R_semantic[idx_src, idx_tgt] = 1.0
        R_semantic[idx_tgt, idx_src] = 1.0
        priors_added_count += 1
print(f"Successfully constructed R_semantic prior matrix with {priors_added_count} bidirectional connections.")


# ==========================================
# 步骤 2: 加载 LLaVA-1.5-7B 模型并动态兼容提取权重
# ==========================================
print(f"Loading LLaVA-1.5-7B model from local path: {MODEL_ID}...")
model = LlavaForConditionalGeneration.from_pretrained(
    MODEL_ID, 
    torch_dtype=torch.float16, 
    device_map="auto", 
    local_files_only=True
)

# 动态提取内部 LLM 结构（兼容新旧两个版本的 transformers）
if hasattr(model, "language_model"):
    llm = model.language_model
elif hasattr(model, "model") and hasattr(model.model, "language_model"):
    llm = model.model.language_model
else:
    raise AttributeError("Could not find the language model component inside the loaded Llava model.")

d_model = llm.config.hidden_size        # 隐藏层维度
num_layers = llm.config.num_hidden_layers # LLM 层数
num_heads = llm.config.num_attention_heads # 每层注意力头数
d_head = d_model // num_heads            # 每个注意力头的维度

print(f"Model properties detected: d_model={d_model}, layers={num_layers}, heads={num_heads}, d_head={d_head}")

# 构建词语 ID 张量
token_ids_tensor = torch.tensor([valid_concept_to_id[w] for w in valid_concepts], dtype=torch.long, device=device)

# 1. 提取词表的嵌入矩阵 (E_txt) - 统一使用 Hugging Face 的标准 get_input_embeddings 接口
embed_layer = model.get_input_embeddings()
E_txt = embed_layer(token_ids_tensor).detach()
E_txt_norm = E_txt / E_txt.norm(dim=-1, keepdim=True) # L2 归一化

# 2. 提取预测映射矩阵 (U_txt) - 统一使用 Hugging Face 的标准 get_output_embeddings 接口
unembed_layer = model.get_output_embeddings()
U_txt = unembed_layer.weight[token_ids_tensor].detach()
U_txt_norm = U_txt / U_txt.norm(dim=-1, keepdim=True) # L2 归一化


# ==========================================
# 步骤 3: 逐个 Attention Head 计算语义关联得分
# ==========================================
sah_scores = np.zeros((num_layers, num_heads))

# 动态定位自注意力层列表 (layers)
if hasattr(llm, "model") and hasattr(llm.model, "layers"):
    layers_list = llm.model.layers
elif hasattr(llm, "layers"):
    layers_list = llm.layers
else:
    raise AttributeError("Could not find decoder layers list inside the language model.")

print("Calculating Semantic Association Head (SAH) scores for each attention head...")
for l in range(num_layers):
    # 提取自注意力块
    attn_layer = layers_list[l].self_attn
    
    W_V_full = attn_layer.v_proj.weight.detach()  # 形状: [d_model, d_model]
    W_O_full = attn_layer.o_proj.weight.detach()  # 形状: [d_model, d_model]
    
    for h in range(num_heads):
        # 1. 截取对应 Head 的子权重
        W_V_head = W_V_full[h * d_head : (h + 1) * d_head, :]  # 形状: [d_head, d_model]
        W_O_head = W_O_full[:, h * d_head : (h + 1) * d_head]  # 形状: [d_model, d_head]
        
        # 2. 计算合成的 OV Circuit 转移矩阵 W_OV
        W_OV = torch.matmul(W_V_head.t(), W_O_head.t())  # 形状: [d_model, d_model]
        
        # 3. 得到词语间的纯文本映射关联分数 M_t2t = E_txt_norm @ W_OV @ U_txt_norm.T
        M_t2t = torch.matmul(torch.matmul(E_txt_norm, W_OV), U_txt_norm.t())  # 形状: [C, C]
        
        # 4. 计算 SAH 强度得分
        diag_val = M_t2t.diag().mean().item()
        
        co_occurrence_mask = R_semantic > 0
        if co_occurrence_mask.sum() > 0:
            co_occur_mean = M_t2t[co_occurrence_mask].mean().item()
        else:
            co_occur_mean = 0.0
            
        # SAH 得分 = 共现区域平均强度 - 对角线平均强度
        sah_scores[l, h] = co_occur_mean - diag_val

print("Calculation completed.")


# ==========================================
# 步骤 4: 保存结果与绘制热力图
# ==========================================
output_dir = "results"
os.makedirs(output_dir, exist_ok=True)

plt.figure(figsize=(14, 9))
sns.heatmap(sah_scores, cmap="RdBu_r", center=0, cbar=True)
plt.title("Semantic Association Heads (SAH) Distribution inside LLaVA-1.5-7B", fontsize=16)
plt.xlabel("Attention Head Index", fontsize=14)
plt.ylabel("Decoder Layer Index", fontsize=14)
plt.xticks(fontsize=10)
plt.yticks(fontsize=10)
plt.tight_layout()
plt.savefig(f"{output_dir}/sah_distribution.png", dpi=300)
plt.close()

# 找出得分最高的 Top 5 语义联想头
flat_indices = np.argsort(sah_scores.flatten())[::-1][:5]
print("\n" + "="*50)
print("TOP 5 SEMANTIC ASSOCIATION HEADS FOUND:")
print("="*50)
for idx in flat_indices:
    l = idx // num_heads
    h = idx % num_heads
    print(f"Layer {l:2d}, Head {h:2d} | SAH Score: {sah_scores[l, h]:.4f}")
print("="*50)
print(f"Heatmap has been successfully saved to '{output_dir}/sah_distribution.png'.")