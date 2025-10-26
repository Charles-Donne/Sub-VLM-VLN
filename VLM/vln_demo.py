import torch
import os
from pathlib import Path
from transformers import AutoModelForImageTextToText, AutoProcessor
from PIL import Image

# 本地模型路径
MODEL_PATH = "/root/autodl-tmp/Qwen3-VL-8B-Instruct"

# 固定的单张观察图像路径
OBSERVATION_IMAGE = "/root/autodl-tmp/current_observation/view.jpg"

# 导航指令（英文）
NAVIGATION_INSTRUCTION = "Go to the kitchen"

print("=" * 70)
print("🤖 VLN Navigator Demo - Single Shot Navigation")
print("=" * 70)

# 加载模型和处理器
print("\n[1/2] Loading model...")
model = AutoModelForImageTextToText.from_pretrained(
    MODEL_PATH,
    device_map="auto",
    dtype="auto",
)
processor = AutoProcessor.from_pretrained(MODEL_PATH)

# 设置图像像素预算
processor.image_processor.size = {
    "longest_edge": 1280*32*32,
    "shortest_edge": 256*32*32
}

print("✅ Model loaded successfully!")

def predict_navigation_action(instruction, image_path):
    """
    预测下一步导航动作（一次性 Demo）
    
    Args:
        instruction: str - 英文导航指令
        image_path: str - 当前观察图像路径
    
    Returns:
        str - 预测的动作 (完整动作描述，如 "Move forward 25 centimeters")
    """
    
    print("\n" + "=" * 70)
    print("🧭 VLN Navigation Demo")
    print("=" * 70)
    print(f"📝 Instruction: {instruction}")
    print(f"📷 Image: {image_path}")
    print("-" * 70)
    
    # 检查图像是否存在
    if not os.path.exists(image_path):
        print(f"❌ Error: Image not found: {image_path}")
        return None
    
    # 构建详细的 Prompt（所有细节都在这里）
    detailed_prompt = f"""You are a vision-language navigation agent. You need to navigate in an indoor environment by following natural language instructions.

**TASK:**
Navigate to the goal location described in the instruction by analyzing the current RGB observation image.

**NAVIGATION INSTRUCTION:**
"{instruction}"

**CURRENT OBSERVATION:**
The image above shows your current first-person view in the environment.

**AVAILABLE ACTIONS:**
You can ONLY choose ONE of the following four actions:
1. Move forward 25 centimeters - Move straight forward by 25 centimeters
2. Turn left 45 degrees - Rotate left by 45 degrees (counterclockwise)
3. Turn right 45 degrees - Rotate right by 45 degrees (clockwise)
4. Move backward 25 centimeters - Move straight backward by 25 centimeters

**DECISION CRITERIA:**
- Analyze the visual scene carefully: identify rooms, objects, doorways, corridors, and spatial layout
- Understand the navigation instruction: determine the target location and required path
- Consider your current orientation and position relative to potential goals
- Choose the action that makes the most progress toward the goal
- Prioritize forward movement when the path is clear and aligned with the goal
- Use turning actions to adjust orientation when needed
- Use backward movement only when blocked and need to retreat

**OUTPUT REQUIREMENTS:**
- Output EXACTLY ONE complete action description
- Must be one of: "Move forward 25 centimeters", "Turn left 45 degrees", "Turn right 45 degrees", "Move backward 25 centimeters"
- Do NOT add any explanation, reasoning, or additional text
- Do NOT use phrases like "I think" or "The action is"
- Output format: Just the action description with exact measurements, nothing else

**EXAMPLES:**
Instruction: "Go to the kitchen"
Image: [shows a corridor with kitchen visible ahead]
Correct output: Move forward 25 centimeters

Instruction: "Turn to face the door"
Image: [shows door on the left side]
Correct output: Turn left 45 degrees

**YOUR TURN:**
Based on the instruction "{instruction}" and the current observation image above, what is the next action?

Output (one word only):"""
    
    try:
        # 加载图像
        print("📥 Loading image...")
        img = Image.open(image_path)
        if img.mode != 'RGB':
            img = img.convert('RGB')
        print(f"✅ Image loaded: {img.size}, mode={img.mode}")
        
        # 构建消息（图像 + 详细 Prompt）
        messages = [{
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": f"file://{os.path.abspath(image_path)}"
                },
                {
                    "type": "text",
                    "text": detailed_prompt
                }
            ]
        }]
        
        # 处理输入
        print("⚙️  Processing input...")
        text = processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        
        if isinstance(text, list):
            text = text[0] if text else ""
        
        inputs = processor(
            text=text,
            images=[img],
            return_tensors="pt",
            padding=True,
        ).to(model.device)
        
        print("🤔 Predicting next action...")
        
        # 生成回复（低温度确保稳定输出）
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=20,   # 只需要一个词
            do_sample=False,     # 贪心解码
            temperature=0.1,
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
        
        # 提取动作
        print(f"\n📤 Raw output: '{output_text}'")
        action = parse_action(output_text)
        
        return action
        
    except Exception as e:
        print(f"❌ Error during prediction: {e}")
        import traceback
        traceback.print_exc()
        return None

def parse_action(output_text):
    """严格解析动作输出（直接返回完整动作描述）"""
    # 清理输出
    output_text = output_text.strip().lower()
    
    # 直接匹配完整动作描述
    if "move forward 25 centimeters" in output_text or "move forward 25cm" in output_text:
        return "Move forward 25 centimeters"
    elif "turn left 45 degrees" in output_text:
        return "Turn left 45 degrees"
    elif "turn right 45 degrees" in output_text:
        return "Turn right 45 degrees"
    elif "move backward 25 centimeters" in output_text or "move backward 25cm" in output_text:
        return "Move backward 25 centimeters"
    # 模糊匹配（如果模型输出了简化版本）
    elif "forward" in output_text and "25" in output_text:
        return "Move forward 25 centimeters"
    elif "left" in output_text and "45" in output_text:
        return "Turn left 45 degrees"
    elif "right" in output_text and "45" in output_text:
        return "Turn right 45 degrees"
    elif "backward" in output_text and "25" in output_text:
        return "Move backward 25 centimeters"
    else:
        print(f"⚠️  Warning: Cannot parse valid action from: '{output_text}'")
        print("   Defaulting to forward movement")
        return "Move forward 25 centimeters"  # 默认向前

# 主程序
if __name__ == "__main__":
    print("\n[2/2] Running navigation demo...")
    
    # 检查图像文件
    if not os.path.exists(OBSERVATION_IMAGE):
        print(f"\n❌ Error: Observation image not found!")
        print(f"   Expected: {OBSERVATION_IMAGE}")
        print(f"   Please place your RGB observation image at this path.")
        exit(1)
    
    # 执行导航预测
    action = predict_navigation_action(NAVIGATION_INSTRUCTION, OBSERVATION_IMAGE)
    
    # 输出结果
    if action:
        print("\n" + "=" * 70)
        print("✅ NAVIGATION RESULT")
        print("=" * 70)
        print(f"📝 Instruction: {NAVIGATION_INSTRUCTION}")
        print(f"🎯 Predicted Action: {action}")
        print()
        print("=" * 70)
    else:
        print("\n❌ Failed to predict action")
