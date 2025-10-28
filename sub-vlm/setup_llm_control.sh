#!/bin/bash
# LLM辅助导航系统 - 快速安装脚本

echo "=========================================="
echo "LLM辅助导航系统 - 快速安装"
echo "=========================================="
echo ""

# 检查Python环境
echo "1. 检查Python环境..."
if ! command -v python &> /dev/null; then
    echo "✗ 错误: 未找到Python"
    exit 1
fi

PYTHON_VERSION=$(python --version 2>&1 | awk '{print $2}')
echo "✓ Python版本: $PYTHON_VERSION"

# 检查依赖
echo ""
echo "2. 检查依赖包..."

REQUIRED_PACKAGES=("yaml" "requests" "cv2" "numpy")
MISSING_PACKAGES=()

for package in "${REQUIRED_PACKAGES[@]}"; do
    if python -c "import $package" &> /dev/null; then
        echo "✓ $package 已安装"
    else
        echo "✗ $package 未安装"
        MISSING_PACKAGES+=($package)
    fi
done

# 安装缺失的包
if [ ${#MISSING_PACKAGES[@]} -gt 0 ]; then
    echo ""
    echo "3. 安装缺失的依赖..."
    
    # 将包名映射到pip包名
    declare -A PIP_NAMES
    PIP_NAMES["yaml"]="pyyaml"
    PIP_NAMES["cv2"]="opencv-python"
    
    for package in "${MISSING_PACKAGES[@]}"; do
        pip_name=${PIP_NAMES[$package]:-$package}
        echo "正在安装 $pip_name..."
        pip install $pip_name
    done
else
    echo ""
    echo "✓ 所有依赖已安装"
fi

# 设置配置文件
echo ""
echo "4. 设置LLM配置文件..."

if [ -f "llm_config.yaml" ]; then
    echo "⚠️  llm_config.yaml 已存在，跳过创建"
else
    if [ -f "llm_config.yaml.template" ]; then
        cp llm_config.yaml.template llm_config.yaml
        echo "✓ 已创建 llm_config.yaml"
        echo ""
        echo "⚠️  重要: 请编辑 llm_config.yaml 并填入你的API密钥"
        echo "   获取密钥: https://openrouter.ai/keys"
    else
        echo "✗ 错误: 未找到 llm_config.yaml.template"
        exit 1
    fi
fi

# 检查Habitat配置
echo ""
echo "5. 检查Habitat配置..."

HABITAT_CONFIG="VLN_CE/habitat_extensions/config/vlnce_task_enhanced.yaml"
if [ -f "$HABITAT_CONFIG" ]; then
    echo "✓ 找到Habitat配置: $HABITAT_CONFIG"
else
    echo "⚠️  警告: 未找到推荐的Habitat配置文件"
    echo "   期望位置: $HABITAT_CONFIG"
fi

# 完成
echo ""
echo "=========================================="
echo "✓ 安装完成！"
echo "=========================================="
echo ""
echo "下一步:"
echo "1. 编辑 llm_config.yaml 并填入你的OpenRouter API密钥"
echo "2. 运行程序:"
echo "   python llm_manual_control.py \\"
echo "       VLN_CE/habitat_extensions/config/vlnce_task_enhanced.yaml \\"
echo "       ./llm_output"
echo ""
echo "详细文档: LLM_CONTROL_README.md"
echo ""
