#!/bin/bash
# LLM辅助导航系统配置设置向导

echo "============================================"
echo "  LLM辅助导航系统配置向导"
echo "============================================"
echo

# 检查是否已存在配置文件
if [ -f "llm_config.yaml" ]; then
    echo "⚠️  发现已存在的 llm_config.yaml 文件"
    read -p "是否覆盖？(y/n): " overwrite
    if [ "$overwrite" != "y" ]; then
        echo "✅ 保留现有配置"
        echo
        echo "提示：你可以手动编辑 llm_config.yaml 文件"
        exit 0
    fi
fi

# 检查模板文件
TEMPLATE_FILE="llm_config.yaml copy.template"
if [ ! -f "$TEMPLATE_FILE" ]; then
    echo "❌ 错误：找不到配置模板文件: $TEMPLATE_FILE"
    exit 1
fi

# 复制模板
cp "$TEMPLATE_FILE" llm_config.yaml
echo "✅ 已复制模板文件"
echo

# 询问API密钥
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "📝 请输入你的 OpenRouter API 密钥"
echo "   获取地址: https://openrouter.ai/keys"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
read -p "API Key: " api_key

# 验证输入
if [ -z "$api_key" ]; then
    echo "❌ API密钥不能为空"
    rm llm_config.yaml
    exit 1
fi

if [[ ! "$api_key" =~ ^sk-or-v1- ]]; then
    echo "⚠️  警告：API密钥格式可能不正确（应以 sk-or-v1- 开头）"
    read -p "是否继续？(y/n): " continue
    if [ "$continue" != "y" ]; then
        rm llm_config.yaml
        exit 0
    fi
fi

# 替换API密钥
if [[ "$OSTYPE" == "darwin"* ]]; then
    # macOS
    sed -i '' "s/sk-or-v1-YOUR_API_KEY_HERE/$api_key/" llm_config.yaml
else
    # Linux
    sed -i "s/sk-or-v1-YOUR_API_KEY_HERE/$api_key/" llm_config.yaml
fi

echo
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "🎨 选择LLM模型"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "推荐模型（支持视觉-语言多模态）："
echo "  1. anthropic/claude-3-5-sonnet (推荐，强大的视觉理解)"
echo "  2. openai/gpt-4-vision-preview (OpenAI视觉模型)"
echo "  3. google/gemini-pro-vision (Google视觉模型)"
echo "  4. 使用默认配置"
echo
read -p "选择 (1-4, 默认1): " model_choice

case $model_choice in
    2)
        model="openai/gpt-4-vision-preview"
        ;;
    3)
        model="google/gemini-pro-vision"
        ;;
    1|4|"")
        model="anthropic/claude-3-5-sonnet"
        ;;
    *)
        echo "⚠️  无效选择，使用默认模型"
        model="anthropic/claude-3-5-sonnet"
        ;;
esac

# 替换模型配置
if [[ "$OSTYPE" == "darwin"* ]]; then
    sed -i '' "s/anthropic\/claude-3-5-sonnet/$model/" llm_config.yaml
else
    sed -i "s/anthropic\/claude-3-5-sonnet/$model/" llm_config.yaml
fi

echo "✅ 已设置模型: $model"

# 检查Python环境
echo
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "🔍 检查Python环境"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

if ! command -v python &> /dev/null; then
    echo "⚠️  警告: 未找到Python"
else
    PYTHON_VERSION=$(python --version 2>&1 | awk '{print $2}')
    echo "✅ Python版本: $PYTHON_VERSION"
fi

# 检查依赖包
echo
echo "检查依赖包..."
REQUIRED_PACKAGES=("yaml" "requests" "cv2" "numpy")
MISSING_PACKAGES=()

for package in "${REQUIRED_PACKAGES[@]}"; do
    if python -c "import $package" &> /dev/null; then
        echo "  ✅ $package"
    else
        echo "  ❌ $package (未安装)"
        MISSING_PACKAGES+=($package)
    fi
done

# 询问是否安装缺失的包
if [ ${#MISSING_PACKAGES[@]} -gt 0 ]; then
    echo
    read -p "是否安装缺失的依赖包？(y/n): " install_deps
    if [ "$install_deps" == "y" ]; then
        declare -A PIP_NAMES
        PIP_NAMES["yaml"]="pyyaml"
        PIP_NAMES["cv2"]="opencv-python"
        
        for package in "${MISSING_PACKAGES[@]}"; do
            pip_name=${PIP_NAMES[$package]:-$package}
            echo "正在安装 $pip_name..."
            pip install $pip_name
        done
    fi
fi

# 完成
echo
echo "============================================"
echo "  ✅ 配置完成！"
echo "============================================"
echo
echo "📁 配置文件已创建：llm_config.yaml"
echo "   此文件不会被提交到 git 仓库（已在 .gitignore 中）"
echo
echo "📊 配置摘要："
echo "   - API密钥: ${api_key:0:20}..."
echo "   - 模型: $model"
echo "   - 基础URL: https://openrouter.ai/api/v1"
echo
echo "🚀 下一步："
echo "   1. 查看配置: cat llm_config.yaml"
echo "   2. 运行程序:"
echo "      python llm_manual_control.py \\"
echo "          ../VLN_CE/habitat_extensions/config/vlnce_task_enhanced.yaml"
echo
echo "📖 更多信息: 查看 QUICKSTART.md"
echo
