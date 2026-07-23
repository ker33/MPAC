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
import matplotlib.pyplot as plt
from transformers import AutoTokenizer, LlavaForConditionalGeneration, AutoProcessor

# ==========================================
# 超参数配置（可微调）
# ==========================================
ALPHA_T = 9.0      # 文本域联想头 (SAH) 的抑制强度 (值越大，对负分头的压制越狠)
ALPHA_V = 9.0      # 跨模态域联想头 (VSAH) 的抑制强度
MIN_SCALE = 0.2    # 保底保留比例 (防止彻底抹除某些头的表征导致输出崩溃)

LIMIT_SAMPLES = 500
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
print(f"Loading LLaVA-1.5-7B (Adaptive Calibration Model)...")
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
    raise AttributeError("Could not find language model.")

if hasattr(model, "vision_tower"):
    vision_tower = model.vision_tower
elif hasattr(model, "model") and hasattr(model.model, "vision_tower"):
    vision_tower = model.model.vision_tower
else:
    raise AttributeError("Could not find vision tower.")

if hasattr(model, "multi_modal_projector"):
    projector = model.multi_modal_projector
elif hasattr(model, "model") and hasattr(model.model, "multi_modal_projector"):
    projector = model.model.multi_modal_projector
else:
    raise AttributeError("Could not find projector.")

d_model = llm.config.hidden_size
num_layers = llm.config.num_hidden_layers
num_heads = llm.config.num_attention_heads
d_head = d_model // num_heads


# ==========================================
# 步骤 3: 提取视觉概念特征与文本嵌入
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
# 步骤 4: 计算对立对称矩阵得分 (量化联想概率)
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

print("\nScanning attention heads for diagnostic scores...")
for l in tqdm(range(num_layers), desc="Scanning Layers"):
    attn_layer = layers_list[l].self_attn
    W_V_full = attn_layer.v_proj.weight.detach().to(device)
    W_O_full = attn_layer.o_proj.weight.detach().to(device)
    
    for h in range(num_heads):
        W_V_head = W_V_full[h * d_head : (h + 1) * d_head, :]
        W_O_head = W_O_full[:, h * d_head : (h + 1) * d_head]
        W_OV = torch.matmul(W_V_head.t(), W_O_head.t())
        
        # Text-to-Text Mapping
        proj_txt = torch.matmul(E_txt_norm, W_OV)
        proj_txt_norm = proj_txt / (proj_txt.norm(dim=-1, keepdim=True) + 1e-8)
        M_t2t = torch.matmul(proj_txt_norm, U_txt_norm.t())
        
        # Vision-to-Text Mapping
        proj_vis = torch.matmul(E_vis_norm, W_OV)
        proj_vis_norm = proj_vis / (proj_vis.norm(dim=-1, keepdim=True) + 1e-8)
        M_v2t = torch.matmul(proj_vis_norm, U_txt_norm.t())
        
        text_dominance_scores[l, h] = M_t2t.diag().mean().item() - (M_t2t * mask_off_diag).sum().item() / (C * C - C)
        vision_dominance_scores[l, h] = M_v2t.diag().mean().item() - (M_v2t * mask_off_diag).sum().item() / (C * C - C)


# ==========================================
# 步骤 5: 动态钩子注入 (Adaptive Scaling Intervention)
# ==========================================
print("\n" + "="*50)
print("--- [ADAPTIVE INTERVENTION INITIALIZATION] ---")
print(f"Applying pre-hooks with ALPHA_T={ALPHA_T}, ALPHA_V={ALPHA_V}, MIN_SCALE={MIN_SCALE}")

# 定义闭包创建钩子，使其不依赖全局变量且对任意输入形状健壮 (Shape-agnostic)
def make_scaling_pre_hook(head_scales):
    def hook(module, args):
        x = args[0] # 输入形状通常为 (batch_size, seq_len, hidden_size) 或 (seq_len, hidden_size)
        orig_shape = x.shape
        hidden_size = orig_shape[-1]
        n_heads = head_scales.shape[0]
        h_dim = hidden_size // n_heads
        
        # 兼容各种 Batch 维度，压平序列维度
        x_reshaped = x.view(-1, n_heads, h_dim)
        
        # 广播应用缩放系数 (1, num_heads, 1)
        scales = head_scales.view(1, n_heads, 1).to(x.device).to(x.dtype)
        x_scaled = x_reshaped * scales
        
        return (x_scaled.view(orig_shape),)
    return hook

active_hooks = []
intervened_heads_count = 0

for l in range(num_layers):
    attn_layer = layers_list[l].self_attn
    o_proj = attn_layer.o_proj
    
    # 构建当前层所有 Head 的缩放张量
    layer_scales = torch.ones(num_heads, dtype=torch.float32)
    for h in range(num_heads):
        t_score = text_dominance_scores[l, h]
        v_score = vision_dominance_scores[l, h]
        
        scale = 1.0
        # 如果是文本联想头 (SAH, 得分为负)，根据程度进行抑制
        if t_score < 0:
            scale = min(scale, 1.0 + ALPHA_T * t_score)
        # 如果是跨模态联想头 (VSAH, 得分为负)，同样对其抑制
        if v_score < 0:
            scale = min(scale, 1.0 + ALPHA_V * v_score)
            
        # 限制保底缩放比例，防止彻底切断特征流通
        scale = max(MIN_SCALE, scale)
        layer_scales[h] = scale
        
        if scale < 1.0:
            intervened_heads_count += 1
            
    # 注册前向 Pre-hook
    hook_fn = make_scaling_pre_hook(layer_scales)
    handle = o_proj.register_forward_pre_hook(hook_fn)
    active_hooks.append(handle)

print(f"Hook registration complete. Intervened heads: {intervened_heads_count} / {num_layers * num_heads}")
print("="*50 + "\n")


# ==========================================
# 步骤 6: 运行自适应模型生成循环 (CHAIR Eval)
# ==========================================
pope_file = "data/pope/coco_pope_adversarial.json"
pope_data = []
with open(pope_file, "r", encoding="utf-8") as f:
    try:
        pope_data = [json.loads(line) for line in f]
    except Exception:
        f.seek(0)
        pope_data = json.load(f)
    
unique_images = []
seen = set()
for item in pope_data:
    img_name = item.get("image", item.get("image_source", ""))
    if img_name and img_name not in seen:
        seen.add(img_name)
        img_id = int(img_name.split("_")[-1].split(".")[0])
        unique_images.append((img_name, img_id))
    if len(unique_images) >= LIMIT_SAMPLES:
        break

print(f"Loaded {len(unique_images)} unique images for Adaptive CHAIR evaluation.")

adaptive_outputs = []

print("Generating captions with Adaptive Calibrated Model...")
for img_name, img_id in tqdm(unique_images):
    img_path = f"data/coco/val2014/{img_name}"
    if not os.path.exists(img_path):
        continue
        
    prompt = "USER: <image>\nPlease describe this image in detail.\nASSISTANT:"
    image = Image.open(img_path).convert("RGB")
    inputs = processor(text=prompt, images=image, return_tensors="pt").to(device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=128,
            do_sample=False, # 采用 Greedy 模式确保可控性，以便公平对比
            pad_token_id=tokenizer.pad_token_id
        )
    
    generated_ids = outputs[0][inputs.input_ids.shape[-1]:]
    generated_caption = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
    
    adaptive_outputs.append({
        "image_id": img_id,
        "caption": generated_caption
    })


# ==========================================
# 步骤 7: 移除钩子并保存结果
# ==========================================
for hook in active_hooks:
    hook.remove()

os.makedirs("results", exist_ok=True)
output_path = "results/chair/chair_outputs_adaptive_011.json"
with open(output_path, "w") as f:
    json.dump(adaptive_outputs, f, indent=4)
    
print(f"\nAdaptive calibrated captions saved successfully to '{output_path}'.")