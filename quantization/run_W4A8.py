import torch
import torch.nn as nn
from transformers import AutoProcessor, AutoModelForImageTextToText
from PIL import Image
import requests
import json
import os
from tqdm import tqdm

# ==========================================
# 1. 核心量化算子：面向 TMMA 的纯对称 W8A8
# ==========================================
# ==========================================
# 核心量化算子：面向 TMMA 的 SmoothQuant W8A8
# 核心量化算子：面向 TMMA 的 Symmetric W4A8
# ==========================================
class SmoothW8A8Linear(nn.Module):
    def __init__(self, original_linear, input_channel_max, alpha=0.5):
        super().__init__()
        self.in_features = original_linear.in_features
        self.out_features = original_linear.out_features
        
        # 获取原始权重的数据类型 (BFloat16) 和设备 (CUDA)
        orig_dtype = original_linear.weight.dtype
        orig_device = original_linear.weight.device
        
        # ==========================================
        # 核心：执行 Smooth 变换 (使用 float32 保证计算精度)
        # ==========================================
        weight_data = original_linear.weight.data.float() 
        weight_col_max = torch.max(torch.abs(weight_data), dim=0)[0]
        
        # 确保输入的最大值也在同一个设备上
        input_channel_max = torch.clamp(input_channel_max.float().to(orig_device), min=1e-5)
        weight_col_max = torch.clamp(weight_col_max, min=1e-5)
        
        # 计算平滑因子 Scales
        smooth_scales = (input_channel_max.pow(alpha) / weight_col_max.pow(1 - alpha)).clamp(min=1e-5)
        
        # 将 Scale 乘入权重
        smoothed_weight = weight_data * smooth_scales.unsqueeze(0)
        
        # ==========================================
        # 1. 权重 W8 量化
        # ==========================================
        w_abs_max = torch.max(torch.abs(smoothed_weight), dim=1, keepdim=True)[0]
        w_scale = torch.clamp(w_abs_max / 127.0, min=1e-8)
        
        w_q = torch.round(smoothed_weight / w_scale)
        w_q = torch.clamp(w_q, -127, 127)
        
        # 生成最终的硬件权重，并强制转回 BFloat16
        self.weight = nn.Parameter((w_q * w_scale).to(orig_dtype), requires_grad=False)
        
        if original_linear.bias is not None:
            self.bias = nn.Parameter(original_linear.bias.data, requires_grad=False)
        else:
            self.register_parameter('bias', None)
            
        # ==========================================
        # 2. 激活值 A8 Scale
        # ==========================================
        smoothed_act_max = input_channel_max / smooth_scales
        act_scale = torch.clamp(torch.max(smoothed_act_max) / 127.0, min=1e-8)

        # 【修复关键 1】：将这两个 Scale 注册为 Buffer，并强制转换为 BFloat16
        # 这样在模型 .to(device) 或 .half() 时，它们会自动跟进，不会产生类型冲突
        self.register_buffer('smooth_scales', smooth_scales.to(orig_dtype))
        self.register_buffer('act_scale', act_scale.to(orig_dtype))

    def forward(self, x):
        # ==========================================
        # 模拟 FPGA 的真实数据流
        # ==========================================
        x_smoothed = x / self.smooth_scales
        
        # A8 量化截断
        x_q = torch.round(x_smoothed / self.act_scale)
        x_q = torch.clamp(x_q, -127, 127)
        x_simulated = x_q * self.act_scale
        
        # 【修复关键 2】：强制把激活值转回输入本身的类型 (BFloat16)
        # 防止计算过程中产生任何 float32 污染
        x_simulated = x_simulated.to(x.dtype)
        
        return nn.functional.linear(x_simulated, self.weight, self.bias)

import torch
import torch.nn as nn

# ==========================================
# 核心量化算子：面向 FPGA 的分组 W4A8
# ==========================================
class GroupedW4A8Linear(nn.Module):
    def __init__(self, original_linear, act_scale, group_size=128):
        super().__init__()
        self.in_features = original_linear.in_features
        self.out_features = original_linear.out_features
        self.group_size = group_size
        
        orig_dtype = original_linear.weight.dtype
        orig_device = original_linear.weight.device
        
        # ==========================================
        # 1. 权重 W4 分组量化 (Grouped Quantization)
        # ==========================================
        weight_data = original_linear.weight.data.float() # [out_features, in_features]
        
        # 确保 in_features 能被 group_size 整除 (大模型通常都是 128 的倍数)
        assert self.in_features % self.group_size == 0, f"in_features ({self.in_features}) 必须能被 group_size ({self.group_size}) 整除"
        
        num_groups = self.in_features // self.group_size
        
        # 将权重 reshape 为 3D 张量：[输出通道, 分组数, 每组元素数]
        w_grouped = weight_data.view(self.out_features, num_groups, self.group_size)
        
        # 在 group_size 维度（dim=2）上寻找最大绝对值
        w_abs_max = torch.max(torch.abs(w_grouped), dim=2, keepdim=True)[0]
        
        # 4-bit 纯对称量化的范围是 [-7, 7] (保留 0 的绝对中心对称)
        w_scales = torch.clamp(w_abs_max / 7.0, min=1e-8)
        
        # 执行 4-bit 量化与截断
        w_q = torch.round(w_grouped / w_scales)
        w_q = torch.clamp(w_q, -7, 7)
        
        # 模拟：反量化回浮点，并将形状变回二维 [out_features, in_features]
        w_dq = (w_q * w_scales).view(self.out_features, self.in_features)
        
        # 存下带有 W4 真实硬件精度损失的权重
        self.weight = nn.Parameter(w_dq.to(orig_dtype), requires_grad=False)
        
        if original_linear.bias is not None:
            self.bias = nn.Parameter(original_linear.bias.data, requires_grad=False)
        else:
            self.register_parameter('bias', None)
            
        # ==========================================
        # 2. 激活值 A8 全局量化
        # ==========================================
        # 激活值依然保持 8-bit [-127, 127]，保护动态特征
        self.register_buffer('act_scale', act_scale.to(orig_dtype))

    def forward(self, x):
        # A8 量化截断
        x_q = torch.round(x / self.act_scale)
        x_q = torch.clamp(x_q, -127, 127)
        x_simulated = x_q * self.act_scale
        
        # 强制类型对齐，防止 float32 传染
        x_simulated = x_simulated.to(x.dtype)
        
        # 此时的矩阵乘法，本质上是 A8 的激活值 乘以 W4 的权重
        return nn.functional.linear(x_simulated, self.weight, self.bias)

'''
class SymmetricW8A8Linear(nn.Module):
    def __init__(self, original_linear, act_scale):
        super().__init__()
        self.in_features = original_linear.in_features
        self.out_features = original_linear.out_features
        
        # --- 权重处理 (Per-Channel 静态量化) ---
        weight_data = original_linear.weight.data
        # 计算每行的最大绝对值，并算出 Scale
        w_abs_max = torch.max(torch.abs(weight_data), dim=1, keepdim=True)[0]
        w_scale = torch.clamp(w_abs_max / 127.0, min=1e-8)
        
        # 模拟截断误差：量化到 [-127, 127] 再反量化
        w_q = torch.round(weight_data / w_scale)
        w_q = torch.clamp(w_q, -127, 127)
        self.weight = nn.Parameter(w_q * w_scale, requires_grad=False)
        
        # --- 偏置处理 (保持高精度，FPGA中通常在累加器后处理) ---
        if original_linear.bias is not None:
            self.bias = nn.Parameter(original_linear.bias.data, requires_grad=False)
        else:
            self.register_parameter('bias', None)
            
        # --- 激活值 Scale (来自校准阶段) ---
        self.act_scale = act_scale

    def forward(self, x):
        # 模拟 FPGA 输入端的数据解包与截断
        x_q = torch.round(x / self.act_scale)
        x_q = torch.clamp(x_q, -127, 127)
        x_simulated = x_q * self.act_scale
        
        # 带有真实硬件精度损失的矩阵乘法
        return nn.functional.linear(x_simulated, self.weight, self.bias)
'''
# ==========================================
# 2. 校准工具：抓取激活值特征
# ==========================================
activation_channel_max_vals = {}

def smoothquant_calibration_hook(module, input, output):
    # 输入 x 形状通常为 [batch, seq_len, in_features]
    x = input[0].detach().float()
    
    # 将 batch 和 seq_len 维度展平，计算每个 in_features 通道的最大绝对值
    x_flat = x.view(-1, x.shape[-1])
    current_channel_max = torch.max(torch.abs(x_flat), dim=0)[0] # 形状: [in_features]
    
    layer_name = module.layer_name 
    if layer_name not in activation_channel_max_vals:
        activation_channel_max_vals[layer_name] = current_channel_max
    else:
        # 取历史最大值
        activation_channel_max_vals[layer_name] = torch.max(
            activation_channel_max_vals[layer_name], current_channel_max
        )
'''
def calibration_hook(module, input, output):
    x = input[0].detach().float()
    current_max = torch.max(torch.abs(x)).item()
    
    layer_name = module.layer_name 
    if layer_name not in activation_max_vals:
        activation_max_vals[layer_name] = current_max
    else:
        activation_max_vals[layer_name] = max(activation_max_vals[layer_name], current_max)
'''    

def register_smoothquant_hooks(model):
    hooks = []
    for name, module in model.named_modules():
        # 依然使用黑名单机制：保护视觉模块，只拦截大语言模型部分的线性层
        if isinstance(module, nn.Linear) and "vision" not in name.lower():
            module.layer_name = name 
            # 注意看这里：我们把新的 smoothquant_calibration_hook 挂载上去！
            hooks.append(module.register_forward_hook(smoothquant_calibration_hook))
    return hooks
'''  
def register_calibration_hooks(model):
    hooks = []
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear) and "vision" not in name.lower():
            module.layer_name = name 
            hooks.append(module.register_forward_hook(calibration_hook))
            # print(f"已挂载探针: {name}") # 可以取消注释看看挂载了哪些
        # 严格限制：只量化语言模型，保护视觉 Token 的高精度特征
        
        if isinstance(module, nn.Linear) and "text_model" in name:
            module.layer_name = name 
            hooks.append(module.register_forward_hook(calibration_hook))
        
    return hooks
'''

# ==========================================
# 3. 主流程：加载 -> 校准 -> 替换 -> 推理
# ==========================================
def main():
    # --------------------------------------
    # 1. 参数配置
    # --------------------------------------
    model_path = "../models"                # 模型权重地址
    test_data_path = "sampled.json"         # 测试数据 JSON 文件路径
    output_log_path = "output/baseline.json"              # 输出日志文件路径
    image_folder_path = "../data" 
    jiaozhun_path = "jiaozhun.json" # 校准数据 JSON 文件路径

    # --------------------------------------
    # 2. 精细化计分字典
    # --------------------------------------
    OCRBench_score = {
        "Regular Text Recognition": 0, "Irregular Text Recognition": 0, "Artistic Text Recognition": 0,
        "Handwriting Recognition": 0, "Digit String Recognition": 0, "Non-Semantic Text Recognition": 0,
        "Scene Text-centric VQA": 0, "Doc-oriented VQA": 0, "Key Information Extraction": 0,
        "Handwritten Mathematical Expression Recognition": 0,
    }

    AllDataset_score = {
        "IIIT5K": 0, "svt": 0, "IC13_857": 0, "IC15_1811": 0, "svtp": 0, "ct80": 0, "cocotext": 0,
        "ctw": 0, "totaltext": 0, "HOST": 0, "WOST": 0, "WordArt": 0, "IAM": 0, "ReCTS": 0,
        "ORAND": 0, "NonSemanticText": 0, "SemanticText": 0, "STVQA": 0, "textVQA": 0, "ocrVQA": 0,
        "ESTVQA": 0, "ESTVQA_cn": 0, "docVQA": 0, "infographicVQA": 0, "ChartQA": 0,
        "ChartQA_Human": 0, "FUNSD": 0, "SROIE": 0, "POIE": 0, "HME100k": 0,
    }

    num_all = {key: 0 for key in AllDataset_score}
    
    # ---------------------------------------------------------
    # 3. 加载模型与处理器
    # ---------------------------------------------------------
    print("正在加载 SmolVLM2 模型和 Processor...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    processor = AutoProcessor.from_pretrained(model_path)
    model = AutoModelForImageTextToText.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16, 
        _attn_implementation="eager" # 禁用 FlashAttention 以便正确 Hook 线性层
    ).to(device)
    model.eval() # 切换到推理模式
    print("--- 探测线性层名称 ---")
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            print(name)

    
    # ----------------------------------------------------------
    # 4. 准备校准数据
    # ----------------------------------------------------------
    print("正在准备校准数据...")
    try:
        with open(jiaozhun_path, 'r', encoding='utf-8') as f:
            jiaozhun_data = json.load(f)
    except FileNotFoundError:
        print(f"未找到 {jiaozhun_path}，请创建一个 mock 数据文件进行测试。")
        return
    print(f"✅ 成功加载 {len(jiaozhun_data)} 条校准样本。")

    # ---------------------------------------------------------
    # 5. 执行校准
    # ---------------------------------------------------------
    print("\n🔍 开始校准：提取激活值分布...")
    hooks = register_smoothquant_hooks(model)
    # hooks = []
    for name, module in model.named_modules():
        # 依然使用黑名单机制：保护视觉模块，只拦截大语言模型部分的线性层
        if isinstance(module, nn.Linear) and "vision" not in name.lower():
            module.layer_name = name 
            # 注意看这里：我们把新的 smoothquant_calibration_hook 挂载上去！
            hooks.append(module.register_forward_hook(smoothquant_calibration_hook))
    # return hooks(model)
    for i in tqdm(range(len(jiaozhun_data)), desc="Calibrating with Jiaozhun Data"):
        jiaozhun_img_path = os.path.join(image_folder_path, jiaozhun_data[i]["image_path"])
        qs = jiaozhun_data[i]["question"]

        if not os.path.exists(jiaozhun_img_path):
            print(f"Warning: Calibration image not found, skipping: {jiaozhun_img_path}")
            continue

        try:
            jiaozhun_image = Image.open(jiaozhun_img_path).convert("RGB")
            jiaozhun_messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": qs}]}]
            jiaozhun_prompt = processor.apply_chat_template(jiaozhun_messages, add_generation_prompt=True)
            jiaozhun_inputs = processor(text=jiaozhun_prompt, images=[jiaozhun_image], return_tensors="pt").to(device)
            with torch.no_grad():
                model(**jiaozhun_inputs) # 前向传播以触发 Hook
        except Exception as e:
            print(f"Error processing {jiaozhun_img_path} during calibration: {e}")
    
    for h in hooks:
        h.remove()
    print(f"✅ 校准完成，共获取 {len(activation_channel_max_vals)} 个线性层的 Scale。")


    # ---------------------------------------------------------
    # 步骤 6：实施 W8A8 算子替换
    # ---------------------------------------------------------
    print("\n⚙️ 正在将网络转换为硬件感知的 W8A8 结构...")
    replaced_count = 0
    for name, module in dict(model.named_modules()).items():
        if "text_model" in name:
            for child_name, child_module in module.named_children():
                if isinstance(child_module, nn.Linear):
                    full_name = f"{name}.{child_name}" if name else child_name
                    if full_name in activation_channel_max_vals:
                        
                        # 【修复点 2】：从一维张量中提取出全局最大值标量 (item)
                        channel_max_tensor = activation_channel_max_vals[full_name]
                        global_max = torch.max(channel_max_tensor).item() 
                        
                        # 计算静态 Act Scale (A8)
                        act_scale_val = max(global_max / 127.0, 1e-8)
                        # 将 act_scale 转为 Tensor 以匹配参数要求
                        act_scale_tensor = torch.tensor(act_scale_val, dtype=torch.float32)
                        
                        # 实例化替换层 (引入刚刚写好的 GroupedW4A8Linear)
                        quantized_layer = GroupedW4A8Linear(child_module, act_scale_tensor, group_size=64)
                        
                        # 设备对齐
                        quantized_layer.to(device)
                        setattr(module, child_name, quantized_layer)
                        replaced_count += 1
                    
                    '''
                    if full_name in activation_channel_max_vals:
                        # 提取按通道的最大值向量
                        channel_max = activation_channel_max_vals[full_name]
                        # 使用 SmoothQuant 算子，alpha 默认设为 0.5 是最经典的甜点
                        quantized_layer = SmoothW8A8Linear(child_module, channel_max, alpha=0.5)
    
                        # 放到对应设备上，保持数据类型一致（bfloat16）
                        quantized_layer.to(device)
                        setattr(module, child_name, quantized_layer)
                        replaced_count += 1
                    '''
                    '''
                    if full_name in activation_max_vals:
                        # 计算静态 Act Scale
                        act_scale = max(activation_max_vals[full_name] / 127.0, 1e-8)
                        # 实例化替换层
                        quantized_layer = SymmetricW8A8Linear(child_module, act_scale)
                        setattr(module, child_name, quantized_layer)
                        replaced_count += 1
                    '''
                        
    print(f"✅ 成功替换 {replaced_count} 个算子，模型已处于 W8A8 状态！")

    # ---------------------------------------------------------
    # 步骤 7：W8A8 量化推理 数据准备
    # ---------------------------------------------------------
    print("正在准备校准数据...")
    try:
        with open(test_data_path, 'r', encoding='utf-8') as f:
            test_samples = json.load(f)
    except FileNotFoundError:
        print(f"未找到 {test_data_path}，请创建一个 mock 数据文件进行测试。")
        return
    print(f"✅ 成功加载 {len(test_samples)} 条测试样本。")

    # ---------------------------------------------------------
    # 步骤 8：执行 W8A8 量化推理
    # ---------------------------------------------------------
    print("正在执行 W8A8 量化推理...")
    for i in tqdm(range(len(test_samples)), desc="Evaluating BF16 OCRBEnch"):
        img_path = os.path.join(image_folder_path, test_samples[i]["image_path"])
        qs = test_samples[i]["question"]

        if not os.path.exists(img_path):
            print(f"Warning: Image not found, skipping: {img_path}")
            test_samples[i]["predict"] = "ERROR: Image not found"
            continue

        try:
            image = Image.open(img_path).convert("RGB")
            
            # 构建标准的 VLM 对话模板
            messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": qs}]}]
            prompt = processor.apply_chat_template(messages, add_generation_prompt=True)
            
            inputs = processor(text=prompt, images=[image], return_tensors="pt").to(device)

            with torch.no_grad():
            # 贪心解码，保证基线结果的绝对稳定性
                generated_ids = model.generate(**inputs, max_new_tokens=100, do_sample=False)
            
            # 裁剪 Prompt，只保留模型生成的新文本
            generated_ids_trimmed = [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)]
            response_content = processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True)[0].strip()
            print(response_content)
            test_samples[i]["predict"] = response_content

        except Exception as e:
            print(f"Error processing {img_path}: {e}")
            test_samples[i]["predict"] = f"MODEL_ERROR: {e}"
            
    # ---------------------------------------------------------
    # 9. 复刻原脚本的评分与字符串匹配逻辑
    # ---------------------------------------------------------
    for i in range(len(test_samples)):
        data_type = test_samples[i].get("type")
        dataset_name = test_samples[i].get("dataset_name")
        answers = test_samples[i].get("answers")

        if "predict" not in test_samples[i] or "ERROR" in test_samples[i]["predict"]:
            test_samples[i]["result"] = 0
            continue

        predict = test_samples[i]["predict"]
        test_samples[i]["result"] = 0 # 默认为错误

        # 特殊处理 HME100k (数学公式，去除所有空格匹配)
        if dataset_name == "HME100k":
            if type(answers) == list:
                for j in range(len(answers)):
                    answer_norm = answers[j].strip().replace("\n", "").replace(" ", "")
                    predict_norm = predict.strip().replace("\n", "").replace(" ", "")
                    if answer_norm in predict_norm:
                        test_samples[i]["result"] = 1
                        break 
            else:
                answer_norm = answers.strip().replace("\n", "").replace(" ", "")
                predict_norm = predict.strip().replace("\n", "").replace(" ", "")
                if answer_norm in predict_norm:
                    test_samples[i]["result"] = 1
        else:
            # 常规字符匹配 (转小写，去换行)
            if type(answers) == list:
                for j in range(len(answers)):
                    answer_norm = answers[j].lower().strip().replace("\n", " ")
                    predict_norm = predict.lower().strip().replace("\n", " ")
                    if answer_norm in predict_norm:
                        test_samples[i]["result"] = 1
                        break 
            else:
                answer_norm = answers.lower().strip().replace("\n", " ")
                predict_norm = predict.lower().strip().replace("\n", " ")
                if answer_norm in predict_norm:
                    test_samples[i]["result"] = 1
    # 保存带有预测和结果标记的 JSON
    with open(output_log_path, 'w', encoding='utf-8') as file:
        json.dump(test_samples, file, indent=4, ensure_ascii=False)

    # ---------------------------------------------------------
    # 10. 打印最终分类统计报告
    # ---------------------------------------------------------
    OCRBench_num_all = {key: 0 for key in OCRBench_score}
    total_ocrbench_items = 0
    total_dataset_items = 0
    
    for item in test_samples:
        item_type = item.get("type")
        if item_type and item_type in OCRBench_num_all:
            OCRBench_num_all[item_type] += 1 
            total_ocrbench_items += 1
            if item.get("result") == 1: 
                OCRBench_score[item_type] += 1

        dataset_name = item.get("dataset_name")
        if dataset_name and dataset_name in num_all:
            num_all[dataset_name] += 1 
            total_dataset_items += 1
            if item.get("result") == 1: 
                AllDataset_score[dataset_name] += 1
    
    if total_ocrbench_items > 0:
        recognition_score = sum(OCRBench_score[k] for k in ["Regular Text Recognition", "Irregular Text Recognition", "Artistic Text Recognition", "Handwriting Recognition", "Digit String Recognition", "Non-Semantic Text Recognition"])
        recognition_total = sum(OCRBench_num_all[k] for k in ["Regular Text Recognition", "Irregular Text Recognition", "Artistic Text Recognition", "Handwriting Recognition", "Digit String Recognition", "Non-Semantic Text Recognition"])
        
        Final_score = sum(OCRBench_score.values())
        Final_total = sum(OCRBench_num_all.values())

        print("\n" + "#"*27 + " OCRBench FP16 Baseline " + "#"*26)
        
        if recognition_total > 0:
            print(f"Text Recognition(Total {recognition_total}): {recognition_score}")
            print("------------------Details of Recognition Score-------------------")
            for t in ["Regular Text Recognition", "Irregular Text Recognition", "Artistic Text Recognition", "Handwriting Recognition", "Digit String Recognition", "Non-Semantic Text Recognition"]:
                if OCRBench_num_all[t] > 0:
                    print(f"{t}(Total {OCRBench_num_all[t]}): {OCRBench_score[t]}")
            print("----------------------------------------------------------------")

        for t in ["Scene Text-centric VQA", "Doc-oriented VQA", "Key Information Extraction", "Handwritten Mathematical Expression Recognition"]:
            if OCRBench_num_all[t] > 0:
                print(f"{t}(Total {OCRBench_num_all[t]}): {OCRBench_score[t]}")
                print("----------------------------------------------------------------")

        print("----------------------Final Score-------------------------------")
        print(f"Final Score(Total {Final_total}): {Final_score} (Accuracy: {Final_score/Final_total*100:.2f}%)")

    elif total_dataset_items > 0:
        print("###########################AllDataset##############################")
        for key in AllDataset_score.keys():
            if num_all[key] > 0: 
                print(f"{key}: {AllDataset_score[key]/float(num_all[key]):.4f}")

if __name__ == "__main__":
    main()