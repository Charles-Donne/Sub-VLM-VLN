import torch
from transformers import AutoModelForImageTextToText, AutoProcessor

# 本地模型路径
MODEL_PATH = "/root/autodl-tmp/Qwen3-VL-8B-Instruct"

# 1. 加载模型和处理器
print("正在加载模型...")
model = AutoModelForImageTextToText.from_pretrained(
    MODEL_PATH,
    device_map="auto",
    dtype="auto",
)
processor = AutoProcessor.from_pretrained(MODEL_PATH)
print("模型加载完成!")

# 2. 准备输入信息 (纯文本对话测试)
messages = [
    {
        "role": "user",
        "content": [
            {"type": "text", "text": "你好!请做一个简单的自我介绍。"},
        ],
    }
]

# 3. 处理器编码输入
print("\n正在处理输入...")
text = processor.apply_chat_template(
    messages, tokenize=False, add_generation_prompt=True
)
inputs = processor(
    text=[text],
    return_tensors="pt",
).to(model.device)

# 4. 模型推理
print("正在生成回复...\n")
generated_ids = model.generate(
    **inputs,
    max_new_tokens=256,  # 减少生成长度，避免卡死
    do_sample=True,      # 使用采样
    temperature=0.7,     # 添加随机性
    top_p=0.8,
    repetition_penalty=1.1  # 防止重复
)

# 5. 解码输出
generated_ids_trimmed = [
    out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
]
output_text = processor.batch_decode(
    generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
)

print("=" * 50)
print("模型回复:")
print(output_text[0])
print("=" * 50)