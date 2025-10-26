import torch
import os
import re
from pathlib import Path
from transformers import AutoModelForImageTextToText, AutoProcessor
from PIL import Image

def natural_sort_key(text):
    """自然排序：正确处理数字（1, 2, 10 而不是 1, 10, 2）"""
    return [int(c) if c.isdigit() else c.lower() for c in re.split('([0-9]+)', str(text))]

# 本地模型路径
MODEL_PATH = "/root/autodl-tmp/Qwen3-VL-8B-Instruct"

# 固定的图像帧目录（修改为你的实际路径）
FRAMES_DIRECTORY = "/root/autodl-tmp/video_frames"

# 历史管理配置
MAX_FRAMES = 8          # 单次最多处理8帧图像

# 1. 加载模型和处理器
print("正在加载模型...")
model = AutoModelForImageTextToText.from_pretrained(
    MODEL_PATH,
    device_map="auto",
    dtype="auto",
)
processor = AutoProcessor.from_pretrained(MODEL_PATH)
print("模型加载完成!")

# 自动加载图像帧
print("\n正在从固定目录加载图像帧...")
print(f"目录: {FRAMES_DIRECTORY}")

# 图像帧列表
current_frames = []  # 存储当前加载的图像路径列表（按时间顺序）

def load_frames_from_directory(directory_path):
    """从目录加载图像帧，按文件名排序"""
    global current_frames
    
    if not os.path.exists(directory_path):
        print(f"❌ 错误: 目录不存在: {directory_path}")
        return False
    
    # 支持的图像格式
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.gif', '.webp'}
    
    # 获取所有图像文件
    image_files = []
    for file in Path(directory_path).iterdir():
        if file.suffix.lower() in image_extensions:
            image_files.append(file)
    
    if not image_files:
        print(f"❌ 错误: 目录中没有找到图像文件")
        return False
    
    # 按文件名自然排序（正确处理数字：1, 2, 10 而不是 1, 10, 2）
    image_files.sort(key=lambda x: natural_sort_key(x.name))
    
    # 限制帧数
    if len(image_files) > MAX_FRAMES:
        print(f"⚠️  警告: 发现 {len(image_files)} 帧，只加载最近的 {MAX_FRAMES} 帧")
        image_files = image_files[-MAX_FRAMES:]
    
    # 转换为绝对路径字符串
    current_frames = [str(file.absolute()) for file in image_files]
    
    print(f"✅ 成功加载 {len(current_frames)} 帧图像（按文件名自然排序）")
    for i, frame in enumerate(current_frames, 1):
        print(f"   Frame {i}: {Path(frame).name}")
    
    return True

def show_current_frames():
    """显示当前加载的图像帧"""
    if not current_frames:
        print("📭 当前没有加载任何图像帧")
        return
    
    print(f"\n📷 当前加载的图像帧 (共 {len(current_frames)} 帧):")
    for i, frame in enumerate(current_frames, 1):
        # 验证文件是否存在
        exists = "✅" if os.path.exists(frame) else "❌"
        print(f"   {i}. {exists} {Path(frame).name}")
        print(f"      路径: {frame}")

def generate_response_with_frames(user_input):
    """生成带图像帧的模型回复（单次推理，无历史）"""
    if not current_frames:
        print("⚠️  提示: 当前没有加载图像帧")
        return None
    
    # 构建单次消息（图像 + 文本）
    user_content = []
    
    # 添加所有图像帧
    for frame_path in current_frames:
        user_content.append({
            "type": "image",
            "image": f"file://{frame_path}"
        })
    
    # 添加用户文本
    user_content.append({
        "type": "text",
        "text": user_input
    })
    
    # 单次对话消息
    messages = [{
        "role": "user",
        "content": user_content
    }]
    
    try:
        # 处理输入
        text = processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        
        # 加载所有图像
        images = []
        for frame in current_frames:
            try:
                img = Image.open(frame)
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                images.append(img)
            except Exception as e:
                print(f"⚠️  警告: 无法加载图像 {frame}: {e}")
        
        if not images:
            print("❌ 错误: 没有成功加载任何图像")
            return None
        
        print(f"🔍 调试信息: 加载了 {len(images)} 张图像")
        for i, img in enumerate(images, 1):
            print(f"   图像 {i}: 尺寸={img.size}, 模式={img.mode}")
        
        inputs = processor(
            text=[text],
            images=images,
            return_tensors="pt",
            padding=True,
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
        
        return output_text
        
    except Exception as e:
        print(f"❌ 生成回复时出错: {e}")
        import traceback
        traceback.print_exc()
        return None

def generate_text_only_response(user_input):
    """生成纯文本回复（单次推理，无历史）"""
    messages = [{
        "role": "user",
        "content": [
            {"type": "text", "text": user_input}
        ]
    }]
    
    try:
        text = processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        inputs = processor(
            text=[text],
            return_tensors="pt",
            padding=True,
        ).to(model.device)
        
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=512,
            do_sample=True,
            temperature=0.7,
            top_p=0.8,
            repetition_penalty=1.05
        )
        
        generated_ids_trimmed = [
            out_ids[len(in_ids):]
            for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False
        )[0]
        
        return output_text
        
    except Exception as e:
        print(f"❌ 生成回复时出错: {e}")
        import traceback
        traceback.print_exc()
        return None

# 启动时自动加载图像帧
if load_frames_from_directory(FRAMES_DIRECTORY):
    print("\n" + "=" * 70)
    print("欢迎使用 Qwen3-VL 视频帧序列对话系统!")
    print("=" * 70)
    print("📝 命令说明:")
    print("  - 'reload'         : 重新加载图像帧")
    print("  - 'show'           : 显示当前已加载的图像帧")
    print("  - 'exit/quit/q'    : 退出程序")
    print("💡 每次都是独立推理，不保留对话历史")
    print("=" * 70)
    print("\n🤖 助手: 你好！我已经加载了图像序列，可以开始提问了。\n")
else:
    print("\n" + "=" * 70)
    print("⚠️  警告: 图像帧加载失败")
    print(f"请检查目录是否存在: {FRAMES_DIRECTORY}")
    print("你仍然可以进行纯文本对话")
    print("=" * 70)
    print("\n🤖 助手: 你好！虽然图像加载失败，但我仍可以帮你。\n")

# 主对话循环
while True:
    try:
        # 获取用户输入
        user_input = input("👤 你: ").strip()
        
        # 检查退出命令
        if user_input.lower() in ['exit', 'quit', 'q']:
            print("\n再见！祝你有美好的一天！👋")
            break
        
        # 重新加载图像帧
        if user_input.lower() == 'reload':
            print("\n重新加载图像帧...")
            load_frames_from_directory(FRAMES_DIRECTORY)
            continue
        
        # 显示当前图像帧
        if user_input.lower() == 'show':
            show_current_frames()
            continue
        
        # 跳过空输入
        if not user_input:
            continue
        
        # 生成回复
        print("\n🤖 助手: ", end="", flush=True)
        
        if current_frames:
            # 带图像的推理
            response = generate_response_with_frames(user_input)
        else:
            # 纯文本对话
            response = generate_text_only_response(user_input)
        
        if response:
            print(response + "\n")
        
    except KeyboardInterrupt:
        print("\n\n检测到中断，正在退出...")
        break
    except Exception as e:
        print(f"\n❌ 发生错误: {e}\n")
        import traceback
        traceback.print_exc()
