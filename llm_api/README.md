# LLM API 工具集

这个文件夹包含所有与LLM API交互相关的工具程序。

## 📁 文件说明

### 核心文件

| 文件 | 说明 |
|------|------|
| `test_api.py` | API连接测试工具 |
| `analyze_episode.py` | Episode指令分析器（主程序） |
| `analyze.sh` | 快捷启动脚本 |
| `api_config.yaml` | API配置文件（⚠️ 不会被git同步） |
| `README.md` | 本文档 |

---

## 🚀 快速开始

### 1. 配置API密钥

**首次使用**：复制模板文件并填入你的API密钥

```bash
cd llm_api

# 复制模板文件
cp api_config.yaml.template api_config.yaml

# 编辑配置文件，填入你的API密钥
nano api_config.yaml
```

在 `api_config.yaml` 中填入你的密钥：

```yaml
openrouter:
  api_key: "sk-or-v1-你的真实密钥"
  default_model: "qwen/qwen-2.5-72b-instruct"
  temperature: 0.7
  max_tokens: 1000
  timeout: 30

output:
  output_dir: "analysis_results"
  use_timestamp: true
```

**注意**：
- ✅ `api_config.yaml.template` 会被提交到 GitHub（不含真实密钥）
- ✅ `api_config.yaml` 已在 `.gitignore` 中，**不会被提交**（保护你的密钥）
- ✅ 每个人克隆仓库后，从模板复制并填入自己的密钥

### 2. 测试API连接

```bash
cd llm_api
python test_api.py
```

### 3. 分析Episode

```bash
# 随机选择一个episode并分析
bash analyze.sh -a

# 指定episode ID并分析
bash analyze.sh 1589 -a
```

---

## 📖 详细使用说明

### `test_api.py` - API连接测试

**功能**：验证OpenRouter API是否可以正常连接

**使用方法**：
```bash
cd llm_api
python test_api.py
```

**输出示例**：
```
============================================================
LLM API 连接测试
============================================================
模型: qwen/qwen-2.5-72b-instruct
API Key: sk-or-v1-6bb0275ad10...62bc

发送消息: 你好，你是谁？

正在连接API...
============================================================
✅ 连接成功！
============================================================
LLM回复: 你好！我是通义千问，由阿里云开发的AI助手...
============================================================
```

---

### `analyze_episode.py` - Episode分析器

**功能**：从VLN-CE数据集中提取episode的导航指令，并可选地使用LLM进行分析

**命令行参数**：
- `--config`: VLN-CE配置文件路径（默认：`../VLN_CE/vlnce_baselines/config/r2r_baselines/navid_r2r.yaml`）
- `--episode-id`: 指定episode ID（不指定则随机选取）
- `--analyze` 或 `-a`: 使用LLM分析指令

**使用示例**：

```bash
cd llm_api

# 1. 随机选择一个episode（仅打印）
python analyze_episode.py

# 2. 随机选择并使用LLM分析
python analyze_episode.py -a

# 3. 指定episode ID
python analyze_episode.py --episode-id 1589

# 4. 指定episode并分析
python analyze_episode.py --episode-id 1589 -a

# 5. 使用不同的配置文件
python analyze_episode.py --config ../VLN_CE/vlnce_baselines/config/rxr_baselines/navid_rxr.yaml -a
```

**输出示例**：
```
加载数据集...
2025-10-20 21:00:00,000 Initializing dataset VLN-CE-v1
✅ 随机选择episode: 1589

================================================================================
Episode ID: 1589
Scene ID: /path/to/scene.glb
Instruction: Walk across the room toward the bedroom. Stop just inside the doorway.
================================================================================

🤖 LLM分析中...

【任务目标】
从当前位置穿过房间，前往卧室，在进入卧室门口时停下。

【关键地标】
- 房间（起点）
- 卧室门口（终点）

【需要执行的动作】
1. 向前行走穿过当前房间
2. 识别卧室入口位置
3. 在刚进入卧室门口时停止
```

---

### `analyze.sh` - 快捷脚本

**功能**：`analyze_episode.py` 的Shell封装，更方便使用

**使用方法**：

```bash
cd llm_api

# 查看帮助
bash analyze.sh -h

# 随机选择episode
bash analyze.sh

# 随机选择并分析
bash analyze.sh -a

# 指定episode
bash analyze.sh 1589

# 指定并分析
bash analyze.sh 1589 -a
```

---

## 🔧 自定义提示词

在 `analyze_episode.py` 文件的第 23-36 行可以编辑提示词：

```python
# ============================================================================
# 🔧 在这里编辑提示词
# ============================================================================

SYSTEM_PROMPT = """你是一个专业的视觉语言导航（VLN）任务分析专家。
请根据给定的导航指令，提供简洁但专业的分析。"""

USER_PROMPT_TEMPLATE = """请分析以下导航指令：

指令：{instruction}

请简要分析：
1. 任务目标
2. 关键地标
3. 需要执行的动作
"""

# ============================================================================
```

修改后直接运行即可生效，无需重新编译。

---

## ⚙️ 配置说明

### `api_config.yaml` 配置项

```yaml
openrouter:
  api_key: "sk-or-v1-..."        # OpenRouter API密钥（必填）
  default_model: "qwen/qwen-2.5-72b-instruct"  # 模型名称
  temperature: 0.7                # 温度：0-1，越高越随机
  max_tokens: 1000               # 最大输出token数
  timeout: 30                    # 请求超时（秒）

output:
  output_dir: "analysis_results" # 结果保存目录
  use_timestamp: true            # 是否使用时间戳
```

**注意事项**：
- ⚠️ `api_config.yaml` 已加入 `.gitignore`，不会被同步到远程仓库
- 在服务器上需要手动创建此文件
- API密钥从 https://openrouter.ai/keys 获取

---

## 📦 依赖库

```bash
# Python包
pip install pyyaml requests

# Habitat和VLN-CE相关依赖（项目已包含）
# - habitat-sim
# - habitat-lab
# - VLN_CE
```

---

## 🛠️ 技术细节

### 路径处理

所有脚本都使用**相对于脚本自身位置**的路径，确保在任何目录运行都能正确工作：

- `api_config.yaml`: 从脚本所在目录（`llm_api/`）读取
- VLN_CE配置: 从项目根目录读取（`../VLN_CE/...`）

### Python路径管理

`analyze_episode.py` 自动添加项目根目录到 Python 路径：

```python
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
sys.path.insert(0, project_root)
```

这样就能正确导入 `VLN_CE` 模块。

---

## ❓ 常见问题

### Q1: 为什么报 `FileNotFoundError: api_config.yaml`？

**A**: 需要在 `llm_api/` 文件夹中创建 `api_config.yaml` 文件，不是项目根目录。

```bash
cd llm_api
# 创建配置文件
cat > api_config.yaml << 'EOF'
openrouter:
  api_key: "你的密钥"
  default_model: "qwen/qwen-2.5-72b-instruct"
  temperature: 0.7
  max_tokens: 1000
  timeout: 30
EOF
```

### Q2: 服务器上如何配置？

**A**: 在服务器上也需要在 `llm_api/` 目录创建 `api_config.yaml`：

```bash
# 方法1：直接创建
cd /path/to/NaVid-VLN-CE/llm_api
nano api_config.yaml
# 粘贴配置内容后保存

# 方法2：从本地上传
scp llm_api/api_config.yaml user@server:/path/to/NaVid-VLN-CE/llm_api/
```

### Q3: 如何修改模型参数？

**A**: 编辑 `api_config.yaml` 文件：

- `temperature`: 0.7（默认）→ 更高更随机，更低更确定
- `max_tokens`: 1000（默认）→ 根据需要调整输出长度

### Q4: 支持哪些模型？

**A**: 支持所有OpenRouter上的模型，例如：
- `qwen/qwen-2.5-72b-instruct`（推荐，中英文双语）
- `anthropic/claude-3.5-sonnet`
- `openai/gpt-4-turbo`
- 更多模型见：https://openrouter.ai/models

修改 `api_config.yaml` 中的 `default_model` 即可切换。

### Q5: API调用失败怎么办？

**A**: 检查以下几点：
1. API密钥是否正确且有效
2. OpenRouter账户是否有余额
3. 网络连接是否正常
4. 先运行 `python test_api.py` 测试连接

---

## 📂 文件结构

```
NaVid-VLN-CE/
├── llm_api/                     # LLM API工具文件夹
│   ├── README.md               # 本文档
│   ├── test_api.py             # API测试工具
│   ├── analyze_episode.py      # Episode分析器
│   ├── analyze.sh              # 快捷脚本
│   └── api_config.yaml         # API配置（不会同步到git）
├── VLN_CE/                      # VLN-CE模块
│   └── vlnce_baselines/config/ # 配置文件
└── .gitignore                   # Git忽略配置
```

---

## 🔐 安全提醒

- ✅ `api_config.yaml` 已加入 `.gitignore`，不会泄露到GitHub
- ⚠️ 不要在代码中硬编码API密钥
- ⚠️ 不要分享包含API密钥的配置文件
- ✅ 定期检查OpenRouter使用额度

---

## 📅 更新日志

- **2025-10-20**: 
  - 修复路径引用问题，支持在 `llm_api/` 目录内运行
  - 更新 `.gitignore` 配置
  - 完善文档和使用说明
  - 优化错误处理

- **2025-10-18**: 
  - 创建LLM API工具集
  - 实现episode指令分析功能
  - 添加API连接测试工具
