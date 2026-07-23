import torch

# ==========================================
# 临时兼容性修复 (Mock Patch) 与随机种子固化
# ==========================================
if not hasattr(torch, "float8_e8m0fnu"):
    setattr(torch, "float8_e8m0fnu", torch.uint8)

import os
import json
import random
import types
import argparse
import numpy as np
from PIL import Image
from tqdm import tqdm
from transformers import AutoTokenizer, LlavaForConditionalGeneration, AutoProcessor

# ==========================================
# 1. 解析参数与参数初始化
# ==========================================
parser = argparse.ArgumentParser()
parser.add_argument('--experiment', type=str, default='llava_adaptive', help='Name of the experiment output file')
args = parser.parse_args()

# 双向协同干预 5 参数配置
ALPHA_T = 8.0
ALPHA_V = 8.0
BETA_V = 5.0
BETA_T = 12.0
MIN_SCALE = 0.2

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
# 2. 载入 Tokenizer、Processor 与有效单词映射
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

# ==========================================
# 3. 载入原生 LLaVA HF 模型并定位组件
# ==========================================
print(f"Loading LLaVA-1.5-7B (HF official model) from: {MODEL_ID}...")
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
# 4. 多卡安全的数据特征字典构建 (E_vis, E_txt & U_txt)
# ==========================================
image_dir = "data/images"
E_vis_list = []
embed_layer = model.get_input_embeddings()

# 多卡安全的特征提取：显式定位子模块所在物理设备
@torch.no_grad()
def extract_clean_visual_feature(image_path, model, processor, vision_tower, projector):
    image = Image.open(image_path).convert("RGB")
    inputs = processor(images=image, return_tensors="pt")
    
    # 动态将输入搬运到视觉塔所在的卡上
    vt_device = next(vision_tower.parameters()).device
    pixel_values = inputs.pixel_values.to(device=vt_device, dtype=torch.float16)
    vision_outputs = vision_tower(pixel_values, output_hidden_states=True)
    image_features = vision_outputs.last_hidden_state[:, 1:, :]
    
    # 动态将视觉特征搬运到 projector 投影层所在的卡上
    proj_device = next(projector.parameters()).device
    projected_features = projector(image_features.to(proj_device))
    
    grid_features = projected_features.view(1, 24, 24, d_model)
    center_features = grid_features[:, 5:19, 5:19, :]
    mean_feature = center_features.reshape(1, -1, d_model).mean(dim=1)
    
    # 特征处理完毕后统一送回主计算 GPU
    return mean_feature.squeeze(0).to(device)

print("Constructing E_vis, E_txt and U_txt with device protection...")
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

# 动态确保索引张量与权重处于相同的 GPU 设备上，获取后再转至主计算 GPU (cuda:0)
token_ids_tensor = torch.tensor([valid_concept_to_id[w] for w in valid_concepts], dtype=torch.long)

E_txt = embed_layer(token_ids_tensor.to(embed_layer.weight.device)).detach().to(device)
E_txt_norm = E_txt / E_txt.norm(dim=-1, keepdim=True)

unembed_layer = model.get_output_embeddings()
U_txt = unembed_layer.weight[token_ids_tensor.to(unembed_layer.weight.device)].detach().to(device)
U_txt_norm = U_txt / U_txt.norm(dim=-1, keepdim=True)

# ==========================================
# 5. 参数映射得分扫描 (计算联想矩阵)
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

print("\nScanning attention heads for dominance metrics...")
for l in tqdm(range(num_layers), desc="Scanning Layers"):
    attn_layer = layers_list[l].self_attn
    W_V_full = attn_layer.v_proj.weight.detach().to(device)
    W_O_full = attn_layer.o_proj.weight.detach().to(device)
    
    for h in range(num_heads):
        W_V_head = W_V_full[h * d_head : (h + 1) * d_head, :]
        W_O_head = W_O_full[:, h * d_head : (h + 1) * d_head]
        W_OV = torch.matmul(W_V_head.t(), W_O_head.t())
        
        # Text-to-Text Domain Score
        proj_txt = torch.matmul(E_txt_norm, W_OV)
        proj_txt_norm = proj_txt / (proj_txt.norm(dim=-1, keepdim=True) + 1e-8)
        M_t2t = torch.matmul(proj_txt_norm, U_txt_norm.t())
        
        # Vision-to-Text Domain Score
        proj_vis = torch.matmul(E_vis_norm, W_OV)
        proj_vis_norm = proj_vis / (proj_vis.norm(dim=-1, keepdim=True) + 1e-8)
        M_v2t = torch.matmul(proj_vis_norm, U_txt_norm.t())
        
        text_dominance_scores[l, h] = M_t2t.diag().mean().item() - (M_t2t * mask_off_diag).sum().item() / (C * C - C)
        vision_dominance_scores[l, h] = M_v2t.diag().mean().item() - (M_v2t * mask_off_diag).sum().item() / (C * C - C)

# ==========================================
# 6. 注册自适应协同引导钩子 (Pre-hooks)
# ==========================================
def make_scaling_pre_hook(head_scales):
    def hook(module, args):
        x = args[0]
        orig_shape = x.shape
        hidden_size = orig_shape[-1]
        n_heads = head_scales.shape[0]
        h_dim = hidden_size // n_heads
        
        x_reshaped = x.view(-1, n_heads, h_dim)
        scales = head_scales.view(1, n_heads, 1).to(x.device).to(x.dtype)
        x_scaled = x_reshaped * scales
        return (x_scaled.view(orig_shape),)
    return hook

active_hooks = []
for l in range(num_layers):
    attn_layer = layers_list[l].self_attn
    o_proj = attn_layer.o_proj
    
    layer_scales = torch.ones(num_heads, dtype=torch.float32)
    for h in range(num_heads):
        t_score = text_dominance_scores[l, h]
        v_score = vision_dominance_scores[l, h]
        
        scale = 1.0
        if v_score > 0:
            boost_v = BETA_V * v_score
            boost_t = BETA_T * t_score if t_score > 0 else 0.0
            scale = 1.0 + boost_v + boost_t
        else:
            suppress_t = ALPHA_T * t_score if t_score < 0 else 0.0
            suppress_v = ALPHA_V * v_score if v_score < 0 else 0.0
            scale = 1.0 + suppress_t + suppress_v
            scale = max(MIN_SCALE, scale)
        layer_scales[h] = scale
        
    hook_fn = make_scaling_pre_hook(layer_scales)
    handle = o_proj.register_forward_pre_hook(hook_fn)
    active_hooks.append(handle)

print("\nDual-steering pre-hooks successfully registered.")

# ==========================================
# 7. 执行 MME 自适应推理循环
# ==========================================
mme_jsonl_path = "data/MME/llava_mme.jsonl"
mme_images_dir = "data/MME/MME_Benchmark_release_version"
output_jsonl_dir = "data/MME/answers"
os.makedirs(output_jsonl_dir, exist_ok=True)
output_jsonl_path = os.path.join(output_jsonl_dir, f"{args.experiment}.jsonl")

print(f"Loading MME questions from: {mme_jsonl_path}")
with open(mme_jsonl_path, "r", encoding="utf-8") as f:
    questions = [json.loads(line) for line in f]

print(f"Running inference on MME, outputting to: {output_jsonl_path}")
with open(output_jsonl_path, "w", encoding="utf-8") as out_f:
    for item in tqdm(questions, desc="MME Inference"):
        img_name_with_cat = item["image"]
        text = item["text"]
        
        img_path = os.path.join(mme_images_dir, img_name_with_cat)
        if not os.path.exists(img_path):
            img_path = os.path.splitext(img_path)[0] + ".jpg"
            if not os.path.exists(img_path):
                continue
                
        prompt = f"USER: <image>\n{text}\nASSISTANT:"
        image = Image.open(img_path).convert("RGB")
        inputs = processor(text=prompt, images=image, return_tensors="pt").to(device)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=16,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id
            )
            
        generated_ids = outputs[0][inputs.input_ids.shape[-1]:]
        pred_text = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
        
        out_f.write(json.dumps({
            "question_id": item["question_id"],
            "prompt": item["text"],
            "text": pred_text
        }) + "\n")

# ==========================================
# 8. 卸载钩子还原模型状态
# ==========================================
for hook in active_hooks:
    hook.remove()
print("\nInference complete. Active pre-hooks removed.")