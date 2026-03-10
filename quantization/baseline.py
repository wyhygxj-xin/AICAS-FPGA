import torch
from transformers import AutoProcessor, AutoModelForImageTextToText
import json
from tqdm import tqdm
from pathlib import Path
from PIL import Image
import os


# 参数配置
model_path = "../models"                # 模型权重地址
test_data_path = "testone.json"         # 测试数据 JSON 文件路径
output_log_path = "output/baseline.json"              # 输出日志文件路径
image_folder_path = "../dataset/OCRBench_v2" # 图片文件夹路径

# ---------------------------------------------------------
# 1. 继承原脚本的精细化计分字典
# ---------------------------------------------------------
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

# ==========================================
# 2. 加载模型与处理器
# ==========================================
print("正在加载 SmolVLM2 模型和 Processor...")
# 你的本地开发环境显存跑这个模型绰绰有余，直接全量加载到 GPU
device = "cuda" if torch.cuda.is_available() else "cpu"

processor = AutoProcessor.from_pretrained(model_path)
# 锁定基线必须使用模型默认的高精度 (bfloat16 或 float16)
model = AutoModelForImageTextToText.from_pretrained(
    model_path,
    torch_dtype=torch.bfloat16, 
).to(device)
model.eval() # 切换到推理模式

# ==========================================
# 3. 准备评测数据
# ==========================================
# 这里模拟读取组委会提供的 100 道题
# 如果组委会提供的数据格式不同，请在这里修改读取逻辑
try:
    with open(test_data_path, 'r', encoding='utf-8') as f:
        test_samples = json.load(f)
except FileNotFoundError:
    print(f"未找到 {test_data_path}，请创建一个 mock 数据文件进行测试。")
    # 生成一条 Mock 数据用于代码跑通测试
    test_samples = [
        {"id": 1, "image_path": "mock_image.jpg", "question": "What is the number on the sign?", "answer": "STOP"}
    ]
    # 创建一张纯黑图像防止报错
    Image.new('RGB', (224, 224), color = 'black').save('mock_image.jpg')

# ---------------------------------------------------------
# 4. 原生前向推理循环
# ---------------------------------------------------------
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
# 4. 完美复刻原脚本的评分与字符串匹配逻辑
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
# 5. 打印最终分类统计报告
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