# src/eval_pope_adaptive.py

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
from transformers import AutoTokenizer, LlavaForConditionalGeneration, AutoProcessor

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed(42) # 固化随机性

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_ID = "models/llava-1.5-7b-hf"

# 控制评测样本量：设为 None 代表跑完每个子集全量 3000 个样本；设为 300 用于快速调试
LIMIT_SAMPLES = 300

# ==========================================
# 自适应干预门控模块 (Ada-VHM)
# ==========================================
class AdaptiveHallucinationMitigator:
    def __init__(self, model, llm, vision_tower, projector, text_scores, vision_scores,
                 tau=1.2, lam=0.5, eta=0.3, init_beta=1.0,
                 img_start=5, img_length=576, attn_threshold=0.3):  # === NEW === 增加图像位置参数
        self.model = model
        self.llm = llm
        self.vision_tower = vision_tower
        self.projector = projector
        self.text_scores = text_scores
        self.vision_scores = vision_scores
        self.tau = tau
        self.lam = lam
        self.eta = eta
        self.init_beta = init_beta
        self.hooks = []
        self.current_beta = init_beta
        self.img_start = img_start          # 视觉 token 起始索引
        self.img_length = img_length        # 视觉 token 数量
        self.attn_threshold = attn_threshold # 视觉注意力阈值

    def reset(self):
        self.current_beta = self.init_beta

    def calculate_entropy_gating(self, logits):
        probs = torch.softmax(logits, dim=-1)
        entropy = -torch.sum(probs * torch.log(probs + 1e-8), dim=-1).mean().item()
        beta = 1.0 / (1.0 + np.exp(-(entropy - self.tau)))
        self.current_beta = beta
        return beta

    def get_attention_hook(self, layer_idx):
        def hook(module, input, output):
            attn_output = output[0]
            if self.current_beta < 0.05:
                return output

            # === NEW === 从 output 中提取注意力权重（当 output_attentions=True 时可用）
            # output 是一个 tuple，通常为 (attn_output, attn_weights, ...)
            # 若模型返回的不是标准格式，可能需要适配。这里假设 output[1] 是注意力权重。
            if len(output) < 2 or output[1] is None:
                return output
            attn_weights = output[1]   # [batch, num_heads, seq_len, seq_len]

            batch_size, seq_len, d_model = attn_output.shape
            num_heads = self.text_scores.shape[1]
            d_head = d_model // num_heads

            # 取当前查询 token（最后一个）对各 key 的注意力分布
            last_query_attn = attn_weights[0, :, -1, :]   # [num_heads, seq_len]
            img_end = self.img_start + self.img_length

            # 计算每个头对视觉 token 的总注意力（0~1 之间）
            a_vis = last_query_attn[:, self.img_start:img_end].sum(dim=1)  # [num_heads]
            # 对文本 token 的注意力（简化近似，也可精确计算）
            # a_text = 1.0 - a_vis

            # 拆分多头输出
            split_output = attn_output.view(batch_size, seq_len, num_heads, d_head)
            alphas = torch.ones((1, 1, num_heads, 1), dtype=attn_output.dtype, device=attn_output.device)

            for h in range(num_heads):
                t_score = self.text_scores[layer_idx, h]
                v_score = self.vision_scores[layer_idx, h]

                # === 动态门控：只有该头正在“看”图像时，才进行压制/增强 ===
                if a_vis[h] > self.attn_threshold:
                    # 压制文本幻觉头（text_dominance 为负且绝对值大）
                    if t_score < -0.05:
                        alphas[0, 0, h, 0] -= self.lam * self.current_beta * abs(t_score) * a_vis[h]
                    # 压制视觉发散头（vision_dominance 为负）
                    if v_score < -0.05:
                        alphas[0, 0, h, 0] -= self.lam * self.current_beta * abs(v_score) * a_vis[h]
                    # 增强视觉翻译头（vision_dominance 为正）
                    if v_score > 0.05:
                        alphas[0, 0, h, 0] += self.eta * self.current_beta * abs(v_score) * a_vis[h]

            alphas = torch.clamp(alphas, min=0.0, max=2.0)
            modified_split = split_output * alphas
            modified_output = modified_split.view(batch_size, seq_len, d_model)
            # 返回时保持原有 output 结构，仅替换第一个元素
            return (modified_output,) + output[1:]
        return hook

    def register(self):
        if hasattr(self.llm, "model") and hasattr(self.llm.model, "layers"):
            layers = self.llm.model.layers
        elif hasattr(self.llm, "layers"):
            layers = self.llm.layers
        else:
            raise AttributeError("Could not locate decoder layers.")

        for l in range(len(layers)):
            h_handle = layers[l].self_attn.register_forward_hook(self.get_attention_hook(l))
            self.hooks.append(h_handle)

    def remove(self):
        for h in self.hooks:
            h.remove()
        self.hooks = []


# ==========================================
# 步骤 1: 加载处理器与 eager 注意力模型
# ==========================================
print(f"Loading processor and tokenizer...")
processor = AutoProcessor.from_pretrained(MODEL_ID, local_files_only=True)
tokenizer = processor.tokenizer

print(f"Loading LLaVA-1.5-7B with Eager Attention...")
model = LlavaForConditionalGeneration.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.float16,
    device_map="auto",
    local_files_only=True,
    attn_implementation="eager"
)

# 动态绑定子模块
if hasattr(model, "language_model"):
    llm = model.language_model
elif hasattr(model, "model") and hasattr(model.model, "language_model"):
    llm = model.model.language_model

if hasattr(model, "vision_tower"):
    vision_tower = model.vision_tower
elif hasattr(model, "model") and hasattr(model.model, "vision_tower"):
    vision_tower = model.model.vision_tower

if hasattr(model, "multi_modal_projector"):
    projector = model.multi_modal_projector
elif hasattr(model, "model") and hasattr(model.model, "multi_modal_projector"):
    projector = model.model.multi_modal_projector

d_model = llm.config.hidden_size
num_layers = llm.config.num_hidden_layers
num_heads = llm.config.num_attention_heads
d_head = d_model // num_heads

# === NEW === 动态计算视觉 token 的准确起始位置（根据实际 prompt 格式）
print("Determining visual token positions...")
# 用一个简单示例确定图像 token 在输入序列中的位置
test_prompt = "USER: <image>\nIs there a cat?\nASSISTANT:"
test_image = Image.new("RGB", (224, 224))
test_inputs = processor(text=test_prompt, images=test_image, return_tensors="pt")
test_input_ids = test_inputs.input_ids[0].tolist()
# 查找 IMAGE_TOKEN_ID（通常为 32000 或类似值，由 processor 自动设置）
# 对于 LLaVA-1.5，IMAGE_TOKEN 的 token ID 由 model.config.image_token_index 给出
image_token_id = model.config.image_token_index if hasattr(model.config, "image_token_index") else 32000
try:
    img_start = test_input_ids.index(image_token_id)
except ValueError:
    # 若找不到，默认从 5 开始（常见值）
    img_start = 5
img_length = 576   # LLaVA-1.5 的固定视觉 token 数量
print(f"Image tokens start at index {img_start}, length {img_length}")

# ==========================================
# 步骤 2: 动态筛选单 Token 词表
# ==========================================
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
print(f"Validated {C} / {len(raw_words)} concepts as 100% single-token.")


# ==========================================
# 步骤 3: 提取视觉字典 (E_vis) —— 多图/单图/文本兜底兼容
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

    elif os.path.exists(concept_path + ".png"):
        feature = extract_clean_visual_feature(concept_path + ".png", model, processor, vision_tower, projector)
        real_images_count += 1
    elif os.path.exists(concept_path + ".jpg"):
        feature = extract_clean_visual_feature(concept_path + ".jpg", model, processor, vision_tower, projector)
        real_images_count += 1
    else:
        token_id = torch.tensor([valid_concept_to_id[word]], dtype=torch.long, device=device)
        feature = embed_layer(token_id).detach().squeeze(0)

    E_vis_list.append(feature.unsqueeze(0))

E_vis = torch.cat(E_vis_list, dim=0)
E_vis_norm = E_vis / E_vis.norm(dim=-1, keepdim=True)
print(f"E_vis constructed. (Categories with images: {real_images_count} / {C})")


# ==========================================
# 步骤 4: 提取文本及预测解嵌
# ==========================================
token_ids_tensor = torch.tensor([valid_concept_to_id[w] for w in valid_concepts], dtype=torch.long, device=device)
E_txt = embed_layer(token_ids_tensor).detach()
E_txt_norm = E_txt / E_txt.norm(dim=-1, keepdim=True)

unembed_layer = model.get_output_embeddings()
U_txt = unembed_layer.weight[token_ids_tensor].detach()
U_txt_norm = U_txt / U_txt.norm(dim=-1, keepdim=True)


# ==========================================
# 步骤 5: 离线扫描获取各个注意力头的对立主导得分
# ==========================================
print("Pre-calculating static functional dominance scores...")
text_dominance = np.zeros((num_layers, num_heads))
vision_dominance = np.zeros((num_layers, num_heads))
mask_diag = torch.eye(C, device=device)
mask_off_diag = 1.0 - mask_diag

if hasattr(llm, "model") and hasattr(llm.model, "layers"):
    layers_list = llm.model.layers
elif hasattr(llm, "layers"):
    layers_list = llm.layers

for l in range(num_layers):
    attn_layer = layers_list[l].self_attn
    W_V = attn_layer.v_proj.weight.detach()
    W_O = attn_layer.o_proj.weight.detach()
    for h in range(num_heads):
        W_V_head = W_V[h * d_head : (h + 1) * d_head, :]
        W_O_head = W_O[:, h * d_head : (h + 1) * d_head]
        W_OV = torch.matmul(W_V_head.t(), W_O_head.t())

        # M_t2t Cosine
        p_t = torch.matmul(E_txt_norm, W_OV)
        p_t_norm = p_t / (p_t.norm(dim=-1, keepdim=True) + 1e-8)
        M_t2t = torch.matmul(p_t_norm, U_txt_norm.t())
        text_dominance[l, h] = M_t2t.diag().mean().item() - (M_t2t * mask_off_diag).sum().item() / (C * C - C)

        # M_v2t Cosine
        p_v = torch.matmul(E_vis_norm, W_OV)
        p_v_norm = p_v / (p_v.norm(dim=-1, keepdim=True) + 1e-8)
        M_v2t = torch.matmul(p_v_norm, U_txt_norm.t())
        vision_dominance[l, h] = M_v2t.diag().mean().item() - (M_v2t * mask_off_diag).sum().item() / (C * C - C)

print("Dominance scanning completed.")


# ==========================================
# 步骤 6: 循环一键运行三个子集的动态评测
# ==========================================
splits = ["adversarial", "popular", "random"]
all_split_results = {}

for split in splits:
    pope_file = f"data/pope/coco_pope_{split}.json"
    print("\n" + "="*60)
    print(f"Starting Evaluation on Split: [{split.upper()}] (Ada-VHM)")
    print("="*60)

    if not os.path.exists(pope_file):
        print(f"Error: {pope_file} not found. Skipping this split.")
        continue

    pope_data = []
    with open(pope_file, "r", encoding="utf-8") as f:
        try:
            pope_data = [json.loads(line) for f in [f] for line in f]
        except Exception:
            f.seek(0)
            pope_data = json.load(f)

    # === 【环境自检与路径诊断】 ===
    if len(pope_data) > 0 and split == "adversarial":
        first_item = pope_data[0]
        print("--- [DIAGNOSTIC LOG] ---")
        img_name = first_item.get("image", first_item.get("image_source", ""))
        question = first_item.get("question", first_item.get("query", first_item.get("text", "")))
        gt_ans = first_item.get("answer", first_item.get("label", ""))
        print(f"Target Image: '{img_name}'")
        print(f"Target Query: '{question}'")
        print(f"Target Label: '{gt_ans}'")
        expected_path = f"data/coco/val2014/{img_name}"
        path_exists = os.path.exists(expected_path)
        print(f"Checking expected path: '{expected_path}' -> Exists? : {path_exists}")
        print("-"*30 + "\n")

    # 样本数量截取
    if LIMIT_SAMPLES is not None:
        pope_data = pope_data[:LIMIT_SAMPLES]
        print(f"Evaluating a subset of {LIMIT_SAMPLES} samples...")
    else:
        print(f"Evaluating full dataset of {len(pope_data)} samples...")

    # 注册自适应钩子（传入动态计算出的 img_start）
    mitigator = AdaptiveHallucinationMitigator(
        model, llm, vision_tower, projector,
        text_dominance, vision_dominance,
        tau=1.2, lam=0.5, eta=0.3, init_beta=1.0,
        img_start=img_start, img_length=img_length, attn_threshold=0.3
    )
    mitigator.register()

    gts = []
    preds = []

    # 动态前向传播解码干预循环
    for item in tqdm(pope_data, desc=f"Processing {split}"):
        img_name = item.get("image", item.get("image_source", ""))
        question = item.get("question", item.get("query", item.get("text", "")))
        gt_ans = item.get("answer", item.get("label", ""))
        if isinstance(gt_ans, str):
            gt_ans = gt_ans.lower()

        if not img_name or not question or not gt_ans:
            continue

        img_path = f"data/coco/val2014/{img_name}"
        if not os.path.exists(img_path):
            continue

        mitigator.reset()

        prompt = f"USER: <image>\n{question}\nASSISTANT:"
        image = Image.open(img_path).convert("RGB")
        inputs = processor(text=prompt, images=image, return_tensors="pt").to(device)

        input_ids = inputs.input_ids
        generated_text = ""

        # 精准前 3 词干预
        for step in range(3):
            with torch.no_grad():
                outputs = model(
                    input_ids=input_ids,
                    pixel_values=inputs.pixel_values,
                    output_attentions=True,
                    return_dict=True
                )
            logits = outputs.logits[:, -1, :]
            mitigator.calculate_entropy_gating(logits)

            next_token_id = torch.argmax(logits, dim=-1, keepdim=True)
            input_ids = torch.cat([input_ids, next_token_id], dim=-1)

            token_str = tokenizer.decode(next_token_id[0], skip_special_tokens=True)
            generated_text += token_str

            if next_token_id.item() == tokenizer.eos_token_id:
                break

        pred_ans = "no"
        if "yes" in generated_text.lower():
            pred_ans = "yes"

        gts.append(gt_ans)
        preds.append(pred_ans)

    mitigator.remove()

    # 统计单子集指标
    gts = np.array(gts)
    preds = np.array(preds)

    TP = np.sum((gts == "yes") & (preds == "yes"))
    TN = np.sum((gts == "no") & (preds == "no"))
    FP = np.sum((gts == "no") & (preds == "yes"))
    FN = np.sum((gts == "yes") & (preds == "no"))

    accuracy = (TP + TN) / len(gts) if len(gts) > 0 else 0
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    yes_ratio = np.sum(preds == "yes") / len(preds) if len(preds) > 0 else 0

    split_results = {
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1-Score": f1,
        "Yes-Ratio": yes_ratio,
        "Total Evaluated": len(gts)
    }

    all_split_results[split] = split_results

    # 独立保存
    save_path = f"results/pope/pope_results_adaptive_{split}_deepsk.json"
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, "w") as rf:
        json.dump(split_results, rf, indent=4)
    print(f"Adaptive Results for [{split}] saved to '{save_path}'")

# ==========================================
# 步骤 7: 打印三数据集动态干预评测汇总表
# ==========================================
print("\n" + "="*70)
print(f"POPE ADAPTIVE (Ada-VHM) EVALUATION SUMMARY TABLE (Total Evaluated: {LIMIT_SAMPLES if LIMIT_SAMPLES else 'FULL 3000'})")
print("="*70)
print(f"{'Split Name':<15} | {'Accuracy':<10} | {'Precision':<10} | {'Recall':<10} | {'F1-Score':<10} | {'Yes-Ratio':<10}")
print("-"*70)
for split in splits:
    if split in all_split_results:
        res = all_split_results[split]
        print(f"{split:<15} | "
              f"{res['Accuracy']*100:.2f}%    | "
              f"{res['Precision']*100:.2f}%   | "
              f"{res['Recall']*100:.2f}%   | "
              f"{res['F1-Score']*100:.2f}%   | "
              f"{res['Yes-Ratio']*100:.2f}%")
print("="*70)