# 快速开始指南 - LLM辅助导航系统

## ⚡ 3分钟快速上手

### 方式1：使用配置向导（推荐）

```bash
# 在 Sub_vlm 目录下运行
cd Sub_vlm

# 运行配置向导
bash setup_llm_control.sh
```

配置向导会自动帮你：
- ✅ 创建配置文件
- ✅ 设置API密钥
- ✅ 选择LLM模型
- ✅ 检查依赖包
- ✅ 安装缺失的包

### 方式2：手动配置

```bash
# 1. 复制配置模板
cp "llm_config.yaml copy.template" llm_config.yaml

# 2. 编辑配置文件
vim llm_config.yaml  # 或使用任何编辑器
```

修改配置文件中的 API 密钥：
```yaml
openrouter:
  api_key: "sk-or-v1-YOUR_API_KEY_HERE"  # 替换为你的实际API密钥
```

**获取API密钥**：访问 https://openrouter.ai/keys

---

### 第二步：测试配置（推荐）

```bash
# 测试API连接和配置
python test_config.py
```

如果看到 `🎉 所有测试通过！` 说明配置成功。

---

### 第三步：运行程序

```bash
# 在 Sub_vlm 目录下运行
python llm_manual_control.py \
    ../VLN_CE/habitat_extensions/config/vlnce_task_enhanced.yaml \
    ./llm_output
```

---

### 第四步：开始导航

1. **观察环境**：系统自动采集8方向图像
2. **查看子任务**：LLM生成第一个子任务
3. **手动执行**：使用键盘控制
   - `1`: 前进
   - `2`: 左转
   - `3`: 右转
   - `c`: 完成子任务（请求LLM验证）
4. **继续循环**：直到任务完成

---

## 🎮 操作说明

### 基本控制
```
0 - STOP (停止导航)
1 - MOVE_FORWARD (前进)
2 - TURN_LEFT (左转30°)
3 - TURN_RIGHT (右转30°)
```

### 特殊命令
```
c    - 完成当前子任务，请求LLM验证
q    - 结束当前episode
exit - 退出程序
```

---

## 📋 工作流程示例

### 示例场景：走到卧室

**Step 1: 系统初始规划**
```
🔍 观察环境 - initial
✓ 收集了 8 个方向的观察图像

🤖 LLM规划 - 生成初始子任务
📋 生成的子任务:
  描述: 向前走到走廊尽头的门口
  提示: 保持直行，注意观察前方是否有门或墙壁。预计需要5-8步前进
  标准: 前方可见门框或墙壁距离小于2米
```

**Step 2: 人工执行**
```
📋 当前子任务 #1
描述: 向前走到走廊尽头的门口

规划提示:
  保持直行，注意观察前方是否有门或墙壁。预计需要5-8步前进

完成标准:
  前方可见门框或墙壁距离小于2米

请输入操作: 1  ← 用户输入：前进
执行动作: MOVE_FORWARD (0.25m)
```

**Step 3: 重复执行直到完成**
```
请输入操作: 1  ← 继续前进
请输入操作: 1
请输入操作: 1
请输入操作: c  ← 用户判断已完成，请求验证
```

**Step 4: LLM验证**
```
🤖 LLM验证 - 检查子任务完成情况

🔍 子任务验证结果:
  完成状态: ✓ 已完成
  分析: 前方可以清楚看到门框，距离约1.5米，符合完成标准

📋 下一个子任务:
  描述: 进入门后向右转，找到卧室门
  提示: 通过门后立即右转90°，观察右侧是否有门
  标准: 右侧可见门，门框清晰可见
```

**Step 5: 继续下一个子任务...**

---

## 📊 输出文件

运行结束后，检查输出目录：

```bash
cd llm_output/episode_12345/

# 查看Episode汇总
cat episode_summary.json

# 查看子任务历史
ls subtasks/

# 查看观察图像
ls observations/initial/

# 查看罗盘可视化
open compass_views/initial_step0_compass.jpg
```

---

## ❓ 常见问题

### Q1: API密钥在哪里获取？
**A**: 访问 https://openrouter.ai/keys，注册账号后创建API密钥

### Q2: 需要多少费用？
**A**: OpenRouter按使用量计费，Claude 3.5 Sonnet约为 $3-15/百万tokens。一个episode大约消耗0.01-0.05美元

### Q3: 可以使用其他模型吗？
**A**: 可以！在 `llm_config.yaml` 中修改 `model` 字段：
```yaml
model: "openai/gpt-4-vision-preview"
# 或
model: "google/gemini-pro-vision"
```

### Q4: 子任务生成不准确怎么办？
**A**: 可以修改 `thinking.py` 中的Prompt，添加更多示例和约束

### Q5: 如何减少API调用次数？
**A**: 
- 不必每步都验证，手动完成多个动作后再按 `c`
- 使用更便宜的模型
- 调整子任务的粒度（更大的子任务 = 更少的验证）

### Q6: 程序崩溃或API失败？
**A**: 系统有降级方案，会使用默认子任务继续运行。检查：
- 网络连接
- API密钥是否正确
- 账户余额是否充足

---

## 🔗 更多信息

- **完整文档**: `LLM_CONTROL_README.md`
- **架构说明**: `ARCHITECTURE.md`
- **新增文件清单**: `NEW_FILES_SUMMARY.md`
- **原手动控制**: `manual_control.py`

---

## 💡 使用技巧

### 技巧1: 子任务粒度控制
- 小粒度（频繁验证）→ 更准确但API成本高
- 大粒度（少验证）→ 更快但可能偏离路径

### 技巧2: 观察要仔细
- 在按 `c` 验证前，自己先观察环境
- 确认符合完成标准后再请求验证
- 可以减少不必要的API调用

### 技巧3: 善用规划提示
- LLM生成的规划提示很有价值
- 包含方向、距离、标志物等关键信息
- 帮助你更好地完成子任务

### 技巧4: 保存重要数据
- 罗盘可视化对理解环境很有帮助
- episode_summary.json 包含完整的任务历史
- 可用于后续分析和改进

---

## 🎯 开始你的第一次导航！

```bash
# 1. 确保配置完成
cat llm_config.yaml

# 2. 运行程序 (在sub-vlm目录下)
cd sub-vlm
python llm_manual_control.py \
    ../VLN_CE/habitat_extensions/config/vlnce_task_enhanced.yaml

# 3. 享受LLM辅助的智能导航体验！
```

祝你导航顺利！🚀
