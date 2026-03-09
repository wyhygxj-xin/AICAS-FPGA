import torch
from transformers import AutoProcessor, AutoModelForImageTextToText

class SmolVLMInference:
    def __init__(self, model_path, device="cuda"):
        self.device = device
        print(f"正在从 {model_path} 加载模型...")
        self.processor = AutoProcessor.from_pretrained(model_path)
        self.model = AutoModelForImageTextToText.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16
        ).to(self.device)
        print("模型加载完成。")

    def infer(self, messages, max_new_tokens=64):
        # 预处理：生成输入张量
        inputs = self.processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        ).to(self.device, dtype=torch.bfloat16)

        # 推理
        generated_ids = self.model.generate(**inputs, do_sample=False, max_new_tokens=max_new_tokens)
        
        # 解码
        generated_texts = self.processor.batch_decode(
            generated_ids,
            skip_special_tokens=True,
        )
        return generated_texts[0]

# --- 使用示例 ---
if __name__ == "__main__":
    # 1. 模型加载 (只运行一次)
    model_path = "../models"                # 模型权重地址
    inference_engine = SmolVLMInference(model_path)

    # 2. 推理 (可多次调用)
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "What is in this image?"},
                {"type": "image", "path": "/workspace/AICAS/smolvlm/photo.jpg"},
            ]
        },
    ]
    
    result = inference_engine.infer(messages)
    print("推理结果:", result)