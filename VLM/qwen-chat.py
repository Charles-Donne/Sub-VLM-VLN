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
print("=" * 60)
print("欢迎使用 Qwen3-VL 对话系统!")
print("输入 'exit'、'quit' 或 'q' 退出对话")
print("输入 'clear' 清空对话历史")
print("=" * 60)

# 对话历史
conversation_history = []

def generate_response(user_input):
    """生成模型回复"""
    # 添加用户消息到历史
    conversation_history.append({
        "role": "user",
        "content": [
            {"type": "text", "text": user_input}
        ]
    })
    
    # 处理输入
    text = processor.apply_chat_template(
        conversation_history, 
        tokenize=False, 
        add_generation_prompt=True
    )
    inputs = processor(
        text=[text],
        return_tensors="pt",
    ).to(model.device)
    
    # 生成回复
    generated_ids = model.generate(
        **inputs,
        max_new_tokens=512,
        do_sample=True,
        temperature=0.7,
        top_p=0.8,
        repetition_penalty=1.05
    )
    
    # 解码输出
    generated_ids_trimmed = [
        out_ids[len(in_ids):] 
        for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, 
        skip_special_tokens=True, 
        clean_up_tokenization_spaces=False
    )[0]
    
    # 添加助手回复到历史
    conversation_history.append({
        "role": "assistant",
        "content": [
            {"type": "text", "text": output_text}
        ]
    })
    
    return output_text

# 主对话循环
print("\n🤖 助手: 你好！我是 Qwen，有什么我可以帮助你的吗？\n")

while True:
    try:
        # 获取用户输入
        user_input = input("👤 你: ").strip()
        
        # 检查退出命令
        if user_input.lower() in ['exit', 'quit', 'q']:
            print("\n再见！祝你有美好的一天！👋")
            break
        
        # 检查清空历史命令
        if user_input.lower() == 'clear':
            conversation_history = []
            print("\n✅ 对话历史已清空！\n")
            continue
        
        # 跳过空输入
        if not user_input:
            continue
        
        # 生成并显示回复
        print("\n🤖 助手: ", end="", flush=True)
        response = generate_response(user_input)
        print(response + "\n")
        
    except KeyboardInterrupt:
        print("\n\n检测到中断，正在退出...")
        break
    except Exception as e:
        print(f"\n❌ 发生错误: {e}\n")
        # 发生错误时移除最后添加的用户消息
        if conversation_history and conversation_history[-1]["role"] == "user":
            conversation_history.pop()
