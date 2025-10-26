#!/bin/bash
# API配置快速设置脚本

echo "============================================"
echo "  LLM API 配置设置向导"
echo "============================================"
echo

# 检查是否已存在配置文件
if [ -f "api_config.yaml" ]; then
    echo "⚠️  发现已存在的 api_config.yaml 文件"
    read -p "是否覆盖？(y/n): " overwrite
    if [ "$overwrite" != "y" ]; then
        echo "已取消设置"
        exit 0
    fi
fi

# 检查模板文件
if [ ! -f "api_config.yaml.template" ]; then
    echo "❌ 错误：找不到 api_config.yaml.template 模板文件"
    exit 1
fi

# 复制模板
cp api_config.yaml.template api_config.yaml
echo "✅ 已复制模板文件"
echo

# 询问API密钥
echo "请输入你的 OpenRouter API 密钥"
echo "（从 https://openrouter.ai/keys 获取）"
read -p "API Key: " api_key

# 验证输入
if [ -z "$api_key" ]; then
    echo "❌ API密钥不能为空"
    exit 1
fi

if [[ ! "$api_key" =~ ^sk-or-v1- ]]; then
    echo "⚠️  警告：API密钥格式可能不正确（应以 sk-or-v1- 开头）"
    read -p "是否继续？(y/n): " continue
    if [ "$continue" != "y" ]; then
        exit 0
    fi
fi

# 替换API密钥
if [[ "$OSTYPE" == "darwin"* ]]; then
    # macOS
    sed -i '' "s/sk-or-v1-YOUR_API_KEY_HERE/$api_key/" api_config.yaml
else
    # Linux
    sed -i "s/sk-or-v1-YOUR_API_KEY_HERE/$api_key/" api_config.yaml
fi

echo
echo "============================================"
echo "  ✅ 配置完成！"
echo "============================================"
echo
echo "配置文件已创建：api_config.yaml"
echo "此文件不会被提交到 git 仓库（已在 .gitignore 中）"
echo
echo "下一步："
echo "  1. 测试API连接：python test_api.py"
echo "  2. 分析episode：bash analyze.sh -a"
echo
