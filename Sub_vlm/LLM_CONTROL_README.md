# LLM辅助的Habitat导航控制系统

## 📖 概述

这是一个结合大语言模型（LLM）规划和人工执行的室内导航控制系统。系统使用LLM分析8方向视觉观察，生成可执行的子任务，并验证完成情况，同时保留人工控制的灵活性。

## 🏗️ 系统架构

```
llm_manual_control.py          # 主程序：协调LLM和人工控制
├── thinking.py                # LLM规划器：子任务生成和验证
├── observation_collector.py   # 观察收集器：8方向图像采集
├── llm_config.py             # 配置管理：API设置
└── llm_config.yaml           # 配置文件：API密钥等
```

### 模块说明

#### 1. **llm_config.py** - LLM配置管理器
负责管理OpenRouter API的配置和认证信息。

**关键功能**:
- 加载和验证配置文件
- 提供API访问凭证
- 管理请求参数（temperature, max_tokens等）

**使用示例**:
```python
from llm_config import LLMConfig

config = LLMConfig("llm_config.yaml")
print(config.model)  # anthropic/claude-3-5-sonnet
headers = config.get_headers()
```

#### 2. **thinking.py** - LLM规划与思考模块
核心的AI规划器，负责所有与LLM的交互。

**关键类**:
- `SubTask`: 子任务数据结构（描述、规划提示、完成标准）
- `LLMPlanner`: LLM规划器

**关键方法**:
```python
# 生成初始子任务（任务开始时）
subtask = planner.generate_initial_subtask(
    instruction="走到卧室",
    observation_images=[path1, path2, ...],  # 8张图
    direction_names=["前方(0°)", "右前方(45°)", ...]
)

# 验证子任务完成并规划下一个
is_completed, next_subtask, advice = planner.verify_and_plan_next(
    instruction="走到卧室",
    current_subtask=subtask,
    observation_images=[...],
    direction_names=[...]
)

# 检查整体任务是否完成
is_done, confidence, analysis = planner.check_task_completion(
    instruction="走到卧室",
    observation_images=[...],
    direction_names=[...]
)
```

**Prompt设计要点**:
- **初始规划Prompt**: 分析完整指令和8方向观察，生成第一个可执行子任务
- **验证Prompt**: 对比当前观察和完成标准，判断子任务是否完成，生成下一个子任务
- **任务完成Prompt**: 判断是否到达最终目标位置

所有Prompt都要求LLM返回严格的JSON格式，便于解析。

#### 3. **observation_collector.py** - 观察收集模块
负责从Habitat环境中采集和处理8方向视觉观察。

**关键方法**:
```python
collector = ObservationCollector(output_dir)

# 收集8方向图像
image_paths, direction_names = collector.collect_8_directions(
    observations,  # Habitat观测字典
    save_prefix="initial"
)

# 创建罗盘式可视化（3x3网格）
compass_img = collector.create_compass_visualization(
    observations,
    save_path="compass.jpg"
)

# 生成观察摘要文本
summary = collector.get_direction_summary(observations)
```

**8方向定义** (顺时针):
1. 前方 (0°) - `rgb`
2. 右前方 (45°) - `rgb_front_right`
3. 右方 (90°) - `rgb_right`
4. 右后方 (135°) - `rgb_back_right`
5. 后方 (180°) - `rgb_back`
6. 左后方 (225°) - `rgb_back_left`
7. 左方 (270°) - `rgb_left`
8. 左前方 (315°) - `rgb_front_left`

#### 4. **llm_manual_control.py** - 主控制程序
协调所有模块，实现LLM规划 + 人工执行的工作流。

**工作流程**:

```
1. 初始化环境和LLM规划器
   ↓
2. 观察8方向（从前方顺时针）
   ↓
3. LLM生成初始子任务
   ├─ 子任务描述
   ├─ 规划提示
   └─ 完成标准
   ↓
4. 显示子任务，等待人工执行
   ↓
5. 人工手动控制（前进、左转、右转）
   ↓
6. 完成后按 'c' 请求验证
   ↓
7. LLM验证子任务完成情况
   ├─ 已完成 → 生成下一个子任务 → 回到步骤4
   └─ 未完成 → 给出建议 → 继续执行
   ↓
8. 重复直到任务结束或手动停止
```

**关键类**:
```python
class LLMAssistedController:
    def reset(self, episode_id)                    # 开始新episode
    def observe_environment(self, obs, phase)       # 收集8方向观察
    def generate_initial_subtask(self, inst, obs)   # 生成初始子任务
    def verify_subtask_completion(self, inst, obs)  # 验证并规划下一个
    def display_current_subtask()                   # 显示当前子任务
    def save_episode_summary(self, inst, metrics)   # 保存汇总
```

## 🚀 快速开始

### 1. 安装依赖

确保已安装所需包：
```bash
pip install pyyaml requests opencv-python numpy
```

或使用配置向导自动安装：
```bash
bash setup_llm_control.sh
```

### 2. 配置LLM API

**方式1：使用配置向导（推荐）**

```bash
# 运行交互式配置向导
bash setup_llm_control.sh
```

向导会引导你：
- 输入 OpenRouter API 密钥
- 选择 LLM 模型
- 检查并安装依赖包
- 验证配置

**方式2：手动配置**

```bash
# 复制配置模板
cp "llm_config.yaml copy.template" llm_config.yaml

# 编辑配置文件
vim llm_config.yaml
```

新配置格式（支持更多选项）：
```yaml
openrouter:
  api_key: "sk-or-v1-xxxxx"              # 你的OpenRouter API密钥
  base_url: "https://openrouter.ai/api/v1"
  default_model: "anthropic/claude-3-5-sonnet"
  temperature: 0.7
  max_tokens: 2000
  timeout: 60

navigation:
  observation:
    enable_8_directions: true            # 启用8方向观察
    save_compass_view: true              # 保存罗盘视图
  subtask:
    auto_verify: false                   # 手动验证子任务

output:
  base_dir: "llm_control_output"         # 输出目录
```

**获取API密钥**:
1. 访问 https://openrouter.ai/keys
2. 注册账号
3. 创建API密钥
4. 充值账户（按使用量付费）

**推荐模型**:
- `anthropic/claude-3-5-sonnet` - 强大的视觉理解，推荐使用
- `openai/gpt-4-vision-preview` - OpenAI的视觉模型
- `google/gemini-pro-vision` - Google的视觉模型

### 3. 测试配置（推荐）

在运行主程序前，先测试配置是否正确：

```bash
# 测试API连接和配置
python test_config.py
```

测试内容包括：
- ✅ 配置文件加载
- ✅ API连接测试
- ✅ 视觉模型支持检查

如果看到 `🎉 所有测试通过！` 说明系统已就绪。

### 4. 准备Habitat配置

使用8方向相机配置：
```bash
# 使用已有的8视角配置（从项目根目录）
../VLN_CE/habitat_extensions/config/vlnce_task_enhanced.yaml
```

### 5. 运行程序

```bash
# 在 Sub_vlm 目录下运行
cd Sub_vlm

python llm_manual_control.py \
    ../VLN_CE/habitat_extensions/config/vlnce_task_enhanced.yaml \
    ./llm_output \
    llm_config.yaml
```

参数说明：
- 参数1: Habitat配置文件路径（必需）
- 参数2: 输出目录（可选，默认 `./llm_control_output`）
- 参数3: LLM配置文件路径（可选，默认 `llm_config.yaml`）

### 5. 操作说明

程序启动后的交互流程：

```
1. 系统初始化
   - 加载Habitat环境
   - 初始化LLM规划器
   - 选择episode

2. 初始观察和规划
   - 自动观察8个方向
   - LLM生成初始子任务
   - 显示子任务详情

3. 人工执行子任务
   输入操作：
   0 - STOP (停止)
   1 - MOVE_FORWARD (前进)
   2 - TURN_LEFT (左转)
   3 - TURN_RIGHT (右转)
   c - 完成子任务，请求LLM验证
   q - 结束当前episode
   exit - 退出程序

4. 子任务验证循环
   - 按 'c' 触发LLM验证
   - LLM判断是否完成
   - 如已完成，生成下一个子任务
   - 如未完成，给出继续执行的建议

5. 重复直到任务结束
```

## 📁 输出结构

运行后会生成以下目录结构：

```
llm_control_output/
└── episode_12345/
    ├── observations/              # 观察数据
    │   ├── initial/              # 初始观察（8方向）
    │   │   ├── initial_step0_dir0_rgb.jpg
    │   │   ├── initial_step0_dir1_rgb_front_right.jpg
    │   │   └── ...
    │   ├── verification_subtask1/ # 第1次验证观察
    │   └── verification_subtask2/ # 第2次验证观察
    ├── subtasks/                  # 子任务记录
    │   ├── subtask_1_initial.json
    │   ├── subtask_2_subtask2.json
    │   └── ...
    ├── compass_views/             # 罗盘式可视化
    │   ├── initial_step0_compass.jpg
    │   ├── verification_subtask1_step5_compass.jpg
    │   └── ...
    ├── step_0000_action.json      # 每步动作记录
    ├── step_0001_action.json
    └── episode_summary.json       # Episode汇总
```

### 关键文件说明

**episode_summary.json**:
```json
{
  "episode_id": "12345",
  "instruction": "走到卧室",
  "total_steps": 25,
  "total_subtasks": 4,
  "subtask_history": [
    {
      "subtask_id": 1,
      "subtask": {
        "description": "向前走到走廊尽头",
        "planning_hints": "保持直行，注意观察前方门",
        "completion_criteria": "前方可见门框"
      },
      "completed": true,
      "completion_step": 8
    }
  ],
  "final_metrics": {
    "distance_to_goal": 1.5,
    "success": 1,
    "spl": 0.75,
    "path_length": 12.5
  }
}
```

**subtask_N_phase.json**:
```json
{
  "subtask_id": 1,
  "phase": "initial",
  "step": 0,
  "description": "向前走到走廊尽头的门口",
  "planning_hints": "保持直行，注意观察前方是否有门或墙壁。预计需要5-8步前进",
  "completion_criteria": "前方可见门框或墙壁距离小于2米"
}
```

## 🔧 自定义和扩展

### 修改Prompt

Prompt在 `thinking.py` 中定义，可以根据需求修改：

1. **初始规划Prompt** (`_build_initial_planning_prompt`):
   - 控制LLM如何分析初始场景
   - 调整子任务的粒度和风格

2. **验证Prompt** (`_build_verification_prompt`):
   - 控制LLM如何判断完成情况
   - 调整完成标准的严格程度

3. **任务完成Prompt** (`_build_task_completion_prompt`):
   - 控制LLM如何判断总体任务完成

**Prompt修改建议**:
- 保持JSON格式要求不变
- 添加更多示例以提高准确性
- 根据实际场景调整描述语言
- 可以添加Few-shot示例

### 更换LLM模型

在 `llm_config.yaml` 中修改 `model` 字段：

```yaml
# 使用不同的模型
model: "openai/gpt-4-vision-preview"
# 或
model: "google/gemini-pro-vision"
```

**注意**：不同模型的性能和成本差异较大，建议先小规模测试。

### 添加额外传感器

如果需要使用深度信息或其他传感器：

1. 修改 `observation_collector.py` 添加新传感器采集
2. 修改 `thinking.py` 的Prompt以利用新信息
3. 更新LLM API调用以支持更多模态

## ⚠️ 注意事项

### API成本

- OpenRouter按使用量计费
- Claude-3.5-Sonnet约为 $3/百万input tokens, $15/百万output tokens
- 每个子任务生成/验证大约消耗 2000-5000 tokens（包括图像）
- 建议监控使用量：https://openrouter.ai/activity

### 性能优化

1. **减少API调用**:
   - 不必每步都验证，可以手动判断后再请求
   - 使用更小的模型进行初步验证

2. **图像压缩**:
   - 可以在 `observation_collector.py` 中添加图像压缩
   - 降低分辨率可减少token消耗

3. **Prompt优化**:
   - 精简Prompt可减少input token
   - 使用更结构化的输出格式

### 错误处理

系统包含基本的错误处理：
- API请求失败 → 使用默认子任务
- JSON解析失败 → 显示原始响应，使用后备方案
- 网络超时 → 可调整 `timeout` 参数

## 🐛 故障排查

### 问题1: API请求失败

**症状**: `✗ API请求失败: 401 Unauthorized`

**解决**:
```bash
# 检查API密钥是否正确
cat llm_config.yaml

# 测试API连接
curl -X POST https://openrouter.ai/api/v1/chat/completions \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"model":"anthropic/claude-3-5-sonnet","messages":[{"role":"user","content":"Hello"}]}'
```

### 问题2: JSON解析失败

**症状**: `✗ JSON解析失败: Expecting value`

**原因**: LLM输出格式不正确

**解决**:
1. 检查 `thinking.py` 中的Prompt是否明确要求JSON格式
2. 尝试更换模型（有些模型更擅长结构化输出）
3. 在Prompt中添加更多JSON示例

### 问题3: 观察图像缺失

**症状**: `⚠️ 观察视角不足`

**原因**: Habitat配置中缺少8方向相机

**解决**:
```bash
# 确保使用8视角配置文件（在sub-vlm目录下）
python llm_manual_control.py \
    ../VLN_CE/habitat_extensions/config/vlnce_task_enhanced.yaml
```

### 问题4: 内存不足

**症状**: 程序运行缓慢或崩溃

**解决**:
1. 减少图像分辨率（修改Habitat配置）
2. 压缩保存的图像
3. 定期清理旧的观察数据

## 📚 相关文档

- [OpenRouter API文档](https://openrouter.ai/docs)
- [Habitat-Lab文档](https://aihabitat.org/docs/habitat-lab/)
- [VLN-CE任务说明](../VLN_CE/README.md)
- [8视角配置指南](../8VIEW_QUICK_REF.md)
- [动作配置指南](../ACTION_CONFIG_GUIDE.md)

## 🤝 贡献

欢迎提出改进建议！特别是：
- Prompt优化方案
- 新的验证策略
- 性能优化建议
- Bug修复

## 📄 许可

遵循项目主LICENSE文件。
