# 人工控制Habitat环境使用说明

## 🎮 程序特点

这是一个**纯人工控制**的Habitat导航程序，特点：

- ✅ **完全手动控制**：每一步都由你决定
- ✅ **实时保存观测**：每步自动保存RGB图像和地图
- ✅ **决策友好**：查看保存的图像后再做决策
- ✅ **自动生成视频**：episode结束后自动生成历史视频
- ✅ **详细评估报告**：包含所有标准VLN-CE指标

## 🚀 快速开始

### 1. 运行程序

```bash
# 基本用法
python manual_control.py <配置文件路径>

# 指定输出目录
python manual_control.py <配置文件路径> <输出目录>

# 示例
python manual_control.py VLN_CE/habitat_extensions/config/vlnce_task.yaml
```

### 2. Episode选择

程序启动后，会先让你选择要运行哪个episode：

```
3. Episode管理
   总数: 100

--------------------------------------------------------------------------------
📋 Episode选择:
  1. 按顺序运行下一个 (当前索引: 0)
  2. 指定Episode索引 (0-99)
  3. 指定Episode ID
  4. 随机选择
  5. 列出所有Episodes (查看ID和信息)
  q. 退出程序

请选择 (1-5/q):
```

**5种选择方式**：
- **按顺序** - 从0开始依次运行，最简单
- **指定索引** - 直接输入episode编号（0到N-1）
- **指定ID** - 使用数据集中的episode_id
- **随机选择** - 随机抽取一个episode
- **列出所有** - 查看所有episode的ID和场景信息

### 3. 操作流程

选择episode后会显示：

```
Episode ID: 123
场景: scene.glb
指令: Walk down the hallway and turn left into the kitchen...

初始距离: 15.32m

✓ 已保存观测 (Step 0)
  - RGB: ./manual_control_output/episode_123/rgb/step_0000.jpg
  - 地图: ./manual_control_output/episode_123/map/step_0000.jpg
  - 组合: ./manual_control_output/episode_123/combined/step_0000.jpg

--------------------------------------------------------------------------------
Step 0
当前距离目标: 15.32m
已行走路径: 0.00m
--------------------------------------------------------------------------------

可用动作:
  0: STOP (停止)
  1: MOVE_FORWARD (前进 0.25m)
  2: TURN_LEFT (左转 30度)
  3: TURN_RIGHT (右转 30度)
  q: 结束当前episode并查看结果
  exit: 退出程序

请输入动作编号:
```

### 3. 决策流程

选择episode后，每一步的决策流程：

1. **查看保存的图像**：打开 `combined/step_XXXX.jpg` 查看当前视角和地图
2. **阅读指令**：理解导航任务
3. **分析距离**：查看 "当前距离目标" 判断是否接近
4. **输入动作**：
   - 输入 `1` 前进
   - 输入 `2` 左转
   - 输入 `3` 右转
   - 输入 `0` 停止（认为到达目标）
   - 输入 `q` 强制结束当前episode
   - 输入 `exit` 退出程序

### 4. Episode完成后

完成一个episode后：
- 自动显示评估结果
- 自动生成历史视频
- 按回车键返回Episode选择菜单
- 可以选择下一个episode或退出

## 📁 输出文件结构

```
manual_control_output/
├── episode_123/                    # 每个episode独立文件夹
│   ├── rgb/                        # RGB图像序列
│   │   ├── step_0000.jpg
│   │   ├── step_0001.jpg
│   │   └── ...
│   ├── map/                        # 地图序列
│   │   ├── step_0000.jpg
│   │   ├── step_0001.jpg
│   │   └── ...
│   ├── combined/                   # 组合视图（推荐查看这个）
│   │   ├── step_0000.jpg          # 包含RGB+地图+文本信息
│   │   ├── step_0001.jpg
│   │   └── ...
│   ├── step_0000_info.json        # 每步的详细信息
│   ├── step_0001_info.json
│   ├── episode_result.json        # Episode评估结果
│   └── episode_123_history.mp4    # 历史帧视频
│
├── episode_456/                    # 下一个episode
│   └── ...
│
└── overall_summary.json            # 所有episodes的汇总
```

## 📊 输出文件说明

### 1. 组合视图 (combined/)

**最重要的文件**，每帧包含：
- 左侧：第一人称RGB视角
- 右侧：俯视地图（红点是你的位置）
- 底部：步数、距离、指令文本

### 2. 步骤信息 (step_XXXX_info.json)

```json
{
    "step": 0,
    "action": {
        "name": "MOVE_FORWARD (前进 0.25m)",
        "id": 1
    },
    "metrics": {
        "distance_to_goal": 15.32,
        "path_length": 0.25
    }
}
```

### 3. Episode结果 (episode_result.json)

```json
{
    "episode_id": "123",
    "scene_id": "scene.glb",
    "instruction": "Walk down the hallway...",
    "total_steps": 25,
    "final_metrics": {
        "distance_to_goal": 2.5,      // 最终距离(米)
        "success": 1,                  // 成功(1)或失败(0)
        "spl": 0.8523,                 // 路径效率
        "path_length": 12.75,          // 总路径长度
        "oracle_success": 1            // 是否曾到达过
    }
}
```

### 4. 历史视频 (episode_XXX_history.mp4)

自动生成的视频，包含所有步骤的组合视图。

### 5. 总体汇总 (overall_summary.json)

```json
{
    "total_episodes": 3,
    "episodes": [...],
    "average_metrics": {
        "avg_distance_to_goal": 3.45,
        "avg_success_rate": 0.67,    // 67%成功率
        "avg_spl": 0.5234,
        "avg_path_length": 15.23
    }
}
```

## 🎯 评估指标解释

### distance_to_goal (最终距离)
- **含义**：停止时智能体与目标点的直线距离（米）
- **越小越好**：< 3米为成功

### success (成功率)
- **含义**：是否成功到达目标
- **判定标准**：最终距离 < 3米 → success = 1
- **范围**：0（失败）或 1（成功）

### SPL (Success weighted by Path Length)
- **含义**：成功率与路径效率的综合指标
- **公式**：`SPL = success × (最短路径 / 实际路径)`
- **范围**：0 到 1
- **越高越好**：既要成功，又要路径高效

### path_length (路径长度)
- **含义**：实际行走的总距离（米）
- **说明**：每次前进增加0.25米

### oracle_success (预言成功)
- **含义**：整个轨迹中是否**曾经**到达过目标3米内
- **用途**：如果 oracle_success=1 但 success=0，说明找到了目标但错过了停止时机

## 💡 使用技巧

### 1. 边看图边决策

```bash
# 方法1：开两个终端窗口
# 终端1：运行程序
python manual_control.py config.yaml

# 终端2：实时查看图像（macOS）
open manual_control_output/episode_XXX/combined/

# 或者用图片查看器实时刷新
```

### 2. 建立决策策略

根据距离制定策略：
- **距离 > 10米**：大胆前进，快速接近
- **距离 5-10米**：适当转向调整方向
- **距离 < 5米**：谨慎前进，寻找目标
- **距离 < 3米**：确认目标后停止

### 3. 理解地图

- **红色三角形**：你的位置和朝向
- **蓝色区域**：可行走区域
- **白色/灰色**：墙壁或障碍物
- **绿色点**：目标位置（有些配置可见）

### 4. 快速测试

```bash
# 只测试1个episode
python manual_control.py config.yaml
# 完成后输入 n 退出

# 每步都前进（快速测试）
# 连续输入：1 1 1 1 ... 0
```

## 🐛 常见问题

### Q: 图像没有保存？
A: 检查是否有权限创建文件夹，或手动创建输出目录

### Q: 视频生成失败？
A: 需要安装OpenCV：`pip install opencv-python`

### Q: 看不到地图？
A: 确保配置文件中启用了 `TOP_DOWN_MAP` sensor

### Q: 输入后没反应？
A: 确保输入的是 0-3 的数字，不要输入其他字符

### Q: 如何跳过当前episode？
A: 输入 `q` 强制结束，或输入 `0` 立即停止

## 📝 完整示例

```bash
# 1. 启动程序
python manual_control.py VLN_CE/habitat_extensions/config/vlnce_task.yaml

# 2. 选择episode
📋 Episode选择:
  1. 按顺序运行下一个 (当前索引: 0)
  2. 指定Episode索引 (0-99)
  3. 指定Episode ID
  4. 随机选择
  5. 列出所有Episodes
  q. 退出程序

请选择 (1-5/q): 2    # 选择方式2：指定索引
请输入Episode索引 (0-99): 5    # 选择第5个episode

# 3. 程序显示episode信息和指令
# Episode ID: 123
# 指令: Walk down the hallway and turn left...

# 4. 查看保存的图像
# 打开: manual_control_output/episode_123/combined/step_0000.jpg

# 5. 根据图像输入动作
请输入动作编号: 1    # 前进

# 6. 重复步骤4-5，直到认为到达目标
请输入动作编号: 1    # 再前进
请输入动作编号: 2    # 左转
请输入动作编号: 1    # 前进
请输入动作编号: 0    # 停止

# 7. 查看结果
# Episode 123 结果
# 总步数: 4
# 最终距离: 2.5m
# 成功: 是
# SPL: 0.8523

# 8. 自动生成视频
# ✓ 视频已生成: episode_123_history.mp4

# 9. 按回车后返回episode选择菜单
按回车键继续...

# 10. 可以继续选择下一个episode或退出
请选择 (1-5/q): q    # 选择退出
```

## 💡 Episode选择技巧

### 完整评估数据集
```
1. 选择"按顺序运行下一个"
2. 一直选1，自动遍历所有episodes
```

### 测试特定场景
```
1. 先选5"列出所有Episodes"查看
2. 找到想测试的场景
3. 选2"指定索引"运行该episode
```

### 重复测试同一个episode
```
1. 记下episode的索引或ID
2. 用选项2或3多次运行
3. 对比不同决策的效果
```

### 随机抽样测试
```
1. 选择4"随机选择"
2. 运行N个随机episode
3. 获得代表性样本
```

## 🎓 学习建议

1. **先熟悉环境**：前几个episode多探索，理解地图和动作
2. **记录策略**：记下哪些指令类型用什么策略有效
3. **分析失败案例**：查看视频，找出决策失误的地方
4. **对比自动agent**：和 `navid_agent.py` 的结果对比

## 🔗 相关文件

- `manual_control.py` - 主程序（完整中文注释）
- `start_manual_control.py` - 启动脚本
- `EPISODE_SELECTION.md` - Episode选择功能详细说明
- `navid_agent.py` - 自动VLM智能体（参考）
- `habitat_basic_interaction.py` - 简单自动agent（参考）
- `VLN_CE/habitat_extensions/config/` - 配置文件

---

**享受人工导航的乐趣！** 🎮
