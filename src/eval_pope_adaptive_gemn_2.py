# src/eval_pope_adaptive.py

import torch

# ==========================================
# 临时兼容性修复 (Mock Patch) 与随机种子固化
# ==========================================
if not hasattr(torch, "float8_e8m0fnu"):
    setattr(torch, "float8_e8m0fnu", torch.uint8)

import os
import json
import math
import random
import numpy as np
from PIL import Image
from tqdm import tqdm
from typing import Optional, Tuple
import torch.nn as nn
from transformers import AutoTokenizer, LlavaForConditionalGeneration, AutoProcessor
from transformers.models.llama.modeling_llama import apply_rotary_pos_emb, repeat_kv

# ==========================================
# 核心：将 AD-HH 算法应用到我们筛选出的对立头上 (SAH & VSAH) - Logit-Masking 健壮版
# ==========================================
def patched_llama_attention_forward(
    self,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_value: Optional[any] = None,
    output_attentions: bool = False,
    use_cache: bool = False,
    cache_position: Optional[torch.LongTensor] = None,
    position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    **kwargs,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    bsz, q_len, _ = hidden_states.size()

    # 动态获取注意力配置参数，确保在任何 transformers 版本下均不发生 AttributeError
    num_heads = getattr(self, "num_heads", getattr(self.config, "num_attention_heads", 32))
    head_dim = getattr(self, "head_dim", getattr(self.config, "hidden_size", 4096) // num_heads)
    num_key_value_heads = getattr(self, "num_key_value_heads", getattr(self.config, "num_key_value_heads", num_heads))
    num_key_value_groups = num_heads // num_key_value_heads
    hidden_size = getattr(self, "hidden_size", getattr(self.config, "hidden_size", 4096))
    layer_idx = getattr(self, "layer_idx", None)

    query_states = self.q_proj(hidden_states)
    key_states = self.k_proj(hidden_states)
    value_states = self.v_proj(hidden_states)

    query_states = query_states.view(bsz, q_len, num_heads, head_dim).transpose(1, 2)
    key_states = key_states.view(bsz, q_len, num_key_value_heads, head_dim).transpose(1, 2)
    value_states = value_states.view(bsz, q_len, num_key_value_heads, head_dim).transpose(1, 2)

    kv_seq_len = key_states.shape[-2]
    if past_key_value is not None:
        if hasattr(past_key_value, "get_seq_len"):
            kv_seq_len += past_key_value.get_seq_len(layer_idx)
        elif hasattr(past_key_value, "seen_tokens"):
            kv_seq_len += past_key_value.seen_tokens

    # 动态提取位置编码
    pos_embeddings = kwargs.get("position_embeddings", position_embeddings)
    if pos_embeddings is not None:
        cos, sin = pos_embeddings
    elif hasattr(self, "rotary_emb"):
        cos, sin = self.rotary_emb(value_states, position_ids)
    else:
        raise AttributeError("LlamaAttention has no rotary_emb and position_embeddings is None.")

    # 旋转位置嵌入应用
    query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

    if past_key_value is not None:
        key_states, value_states = past_key_value.update(key_states, value_states, layer_idx, kwargs)

    key_states = repeat_kv(key_states, num_key_value_groups)
    value_states = repeat_kv(value_states, num_key_value_groups)

    # 计算原始内积 Logits
    attn_logits = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(head_dim)

    if attention_mask is not None:
        attn_logits = attn_logits + attention_mask

    # -------------------------------------------------------------
    # [Ada-HH INTERVENTION ENGINE] - Logit-Masking 机制
    # -------------------------------------------------------------
    config = self.config
    if getattr(config, "adaptive_deactivate", False):
        hal_heads_in_layer = [head_idx for l_idx, head_idx in config.hal_attention_heads if l_idx == layer_idx]
        
        if len(hal_heads_in_layer) > 0:
            img_start = config.img_start_pos
            img_len = config.img_length
            img_end = img_start + img_len
            seq_len = attn_logits.shape[-1]
            
            # 当视觉特征展开进入序列时启动过滤
            if seq_len >= img_end:
                text_mask = torch.ones(seq_len, dtype=torch.bool, device=attn_logits.device)
                text_mask[img_start:img_end] = False
                
                for head_idx in hal_heads_in_layer:
                    temp_weights = nn.functional.softmax(attn_logits[:, head_idx, -1, :], dim=-1) # [bsz, seq_len]
                    text_attn_sum = temp_weights[:, text_mask].sum(dim=-1) # [bsz]
                    
                    # 门限判定：如果对上下文文本注意力过强
                    mask_to_apply = (text_attn_sum > config.adhh_threshold)
                    
                    if mask_to_apply.any():
                        for b_idx in range(bsz):
                            if mask_to_apply[b_idx]:
                                # 将文本部分的 Logits 设为极大负值，Softmax 后权重平滑归零。
                                attn_logits[b_idx, head_idx, -1, text_mask] = -10000.0
    # -------------------------------------------------------------

    # 进行高精度、安全的 Softmax 归一化
    attn_weights = nn.functional.softmax(attn_logits, dim=-1, dtype=torch.float32).to(query_states.dtype)

    attn_output = torch.matmul(attn_weights, value_states)

    if attn_output.size() != (bsz, num_heads, q_len, head_dim):
        raise ValueError(
            f"`attn_output` size should be {(bsz, num_heads, q_len, head_dim)}, but is"
            f" {attn_output.size()}"
        )

    attn_output = attn_output.transpose(1, 2).contiguous()
    attn_output = attn_output.reshape(bsz, q_len, hidden_size)
    attn_output = self.o_proj(attn_output)

    if not output_attentions:
        attn_weights = None

    return attn_output, attn_weights


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ==========================================
# 运行环境 - 保护块包裹
# ==========================================
if __name__ == "__main__":
    # 注入自适应猴子补丁
    import transformers.models.llama.modeling_llama as modeling_llama
    print("Injecting Adaptive Deactivation Monkey Patch into LlamaAttention...")
    modeling_llama.LlamaAttention.forward = patched_llama_attention_forward

    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    LIMIT_SAMPLES = None # 全量评测

    # ==========================================
    # 步骤 1: 加载处理器与 LLaVA 模型
    # ==========================================
    print(f"Loading processor and tokenizer...")
    processor = AutoProcessor.from_pretrained(MODEL_ID, local_files_only=True)
    tokenizer = processor.tokenizer

    print(f"Loading LLaVA-1.5-7B with patched attention layers...")
    model = LlavaForConditionalGeneration.from_pretrained(
        MODEL_ID, 
        torch_dtype=torch.float16, 
        device_map="auto", 
        local_files_only=True,
        attn_implementation="eager"
    )

    # 动态绑定组件
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
    # 步骤 6: 基于我们算出的打分，动态提取我们自己的高危对立头 (SAH & VSAH)
    # ==========================================
    # 提取打分最低（最脑补）的前 20 个文本联想头 (SAH)
    flat_text = text_dominance.flatten()
    top_sah_indices = np.argsort(flat_text)[:20]

    # 提取打分最低（最脑补）的前 20 个视觉联想头 (VSAH)
    flat_vision = vision_dominance.flatten()
    top_vsah_indices = np.argsort(flat_vision)[:20]

    # 合并去重，得到我们模型专属的高危选定头
    hal_heads_set = set()
    for idx in top_sah_indices:
        hal_heads_set.add((idx // num_heads, idx % num_heads))
    for idx in top_vsah_indices:
        hal_heads_set.add((idx // num_heads, idx % num_heads))

    # 写入 LLaVA 配置
    model.config.adaptive_deactivate = True
    model.config.hal_attention_heads = list(hal_heads_set)
    model.config.img_length = 576
    model.config.adhh_threshold = 0.4 

    print(f"\nSuccessfully identified {len(model.config.hal_attention_heads)} custom hallucination heads (SAH & VSAH) based on your Red-Blue matrix.")


    # ==========================================
    # 步骤 7: 循环一键运行三个子集的自适应干预评测
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
                pope_data = [json.loads(line) for line in f]
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

        gts = []
        preds = []

        # 运行自回归推理
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
                
            prompt = f"USER: <image>\n{question}\nASSISTANT:"
            image = Image.open(img_path).convert("RGB")
            inputs = processor(text=prompt, images=image, return_tensors="pt").to(device)
            
            # 核心：动态、精准定位首个图像标记的其实位置索引，杜绝硬编码偏移
            image_token_id = tokenizer.encode("<image>", add_special_tokens=False)[0]
            img_indices = (inputs.input_ids[0] == image_token_id).nonzero(as_tuple=True)[0]
            img_start_pos = img_indices[0].item() if len(img_indices) > 0 else 35
            model.config.img_start_pos = img_start_pos
            
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=8,       
                    do_sample=False,        
                    pad_token_id=tokenizer.pad_token_id
                )
                
            generated_ids = outputs[0][inputs.input_ids.shape[-1]:]
            generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
                    
            pred_ans = "no"
            if "yes" in generated_text.lower():
                pred_ans = "yes"
                
            gts.append(gt_ans)
            preds.append(pred_ans)

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
        save_path = f"results/pope/pope_results_adaptive_{split}.json"
        with open(save_path, "w") as rf:
            json.dump(split_results, rf, indent=4)
        print(f"Adaptive Results for [{split}] saved to '{save_path}'")

    # ==========================================
    # 步骤 8: 打印三数据集动态干预评测汇总表
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