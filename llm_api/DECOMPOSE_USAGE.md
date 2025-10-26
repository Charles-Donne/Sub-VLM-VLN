# 指令分解工具 - 快速使用指南

## 🎯 功能

将导航指令分解为多个结构化的子指令，每个子指令包含动作类型、目标地标、完成条件等信息。

## 🚀 快速开始

### 1. 基本使用

```bash
cd llm_api

# 使用示例指令测试
python decompose_instruction.py

# 分解自定义指令
python decompose_instruction.py "Walk to the kitchen and stop at the fridge"

# 使用快捷脚本
bash decompose.sh "Turn left and enter the bedroom"
```

### 2. 保存结果

```bash
# 保存为JSON文件
python decompose_instruction.py "Walk forward" -o result.json

# 简化输出（只显示关键信息）
python decompose_instruction.py "Turn right" -s
```

## 📝 输出示例

```
================================================================================
📋 指令分解结果
================================================================================

原始指令: Walk across the room toward the bedroom. Stop just inside the doorway.

共分解为 3 个子指令:

[子指令 1]
  ▸ 动作: Walk across the room
  ▸ 类型: move_forward
  ▸ 目标地标: room
  ▸ 空间关系: across
  ▸ 完成条件: Reached the other side of the room

[子指令 2]
  ▸ 动作: Approach the bedroom
  ▸ 类型: approach
  ▸ 目标地标: bedroom
  ▸ 空间关系: toward
  ▸ 完成条件: Standing near the bedroom entrance

[子指令 3]
  ▸ 动作: Stop inside the doorway
  ▸ 类型: stop
  ▸ 目标地标: doorway
  ▸ 空间关系: inside
  ▸ 场景转换: Entering bedroom area
  ▸ 完成条件: Agent positioned just inside doorway

================================================================================
```

## 💻 在Python代码中调用

```python
from decompose_instruction import decompose_instruction, load_config

# 1. 加载API配置
config = load_config()

# 2. 分解指令
instruction = "Walk to the kitchen"
result = decompose_instruction(instruction, config)

# 3. 使用结果
print(f"原始指令: {result['instruction_original']}")
print(f"子指令数量: {len(result['sub_instructions'])}")

for sub in result['sub_instructions']:
    print(f"{sub['sub_id']}: {sub['sub_instruction']}")
    print(f"  - 类型: {sub['action_type']}")
    print(f"  - 目标: {sub['target_landmark']}")
```

## 📦 JSON输出格式

```json
{
  "instruction_original": "Walk to the kitchen",
  "sub_instructions": [
    {
      "sub_id": 1,
      "sub_instruction": "Walk forward",
      "action_type": "move_forward",
      "target_landmark": "kitchen",
      "spatial_relation": "toward",
      "scene_transition": "",
      "completion_condition": "Arrived at the kitchen entrance"
    }
  ]
}
```

## 🔧 动作类型说明

| 类型 | 说明 | 示例 |
|------|------|------|
| `move_forward` | 向前移动 | "Walk across the room" |
| `turn` | 转向 | "Turn left" |
| `enter` | 进入房间 | "Enter the bedroom" |
| `exit` | 离开房间 | "Exit the kitchen" |
| `stop` | 停止 | "Stop at the door" |
| `look` | 观察/寻找 | "Look for the chair" |
| `approach` | 接近目标 | "Approach the table" |
| `navigate` | 导航到某处 | "Navigate to the hallway" |

## ⚙️ 修改系统提示词

编辑 `decompose_instruction.py` 文件的第 24-51 行，修改 `SYSTEM_PROMPT` 变量。

## 📂 文件说明

- `decompose_instruction.py` - 主程序（包含所有核心函数）
- `decompose.sh` - 快捷调用脚本
- `DECOMPOSE_USAGE.md` - 本使用说明

## 🔍 批量处理示例

```python
from decompose_instruction import decompose_instruction, load_config, save_decomposition

config = load_config()

# 要处理的指令列表
instructions = [
    "Walk to the kitchen",
    "Turn left and enter the bedroom",
    "Go through the hallway"
]

# 批量分解并保存
for i, inst in enumerate(instructions):
    result = decompose_instruction(inst, config)
    save_decomposition(result, f"results/decomp_{i}.json")
    print(f"✅ 完成 {i+1}/{len(instructions)}")
```

## ⚠️ 注意事项

1. 需要先配置 `llm_api/api_config.yaml` 文件
2. 确保API密钥有效且账户有余额
3. 首次使用建议先测试示例指令

## 📞 遇到问题？

```bash
# 查看帮助
bash decompose.sh -h

# 测试API连接
python test_api.py
```
