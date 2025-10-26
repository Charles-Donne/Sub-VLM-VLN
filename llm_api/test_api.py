#!/usr/bin/env python3
"""
LLM API连接测试程序
"""

import os
import yaml
import requests

# 加载配置（从当前脚本所在目录读取）
script_dir = os.path.dirname(os.path.abspath(__file__))
config_path = os.path.join(script_dir, 'api_config.yaml')

with open(config_path, 'r') as f:
    config = yaml.safe_load(f)

api_key = config['openrouter']['api_key']
model = config['openrouter']['default_model']

print("="*60)
print("LLM API 连接测试")
print("="*60)
print(f"模型: {model}")
print(f"API Key: {api_key[:20]}...{api_key[-4:]}")
print()

# 测试对话
test_message = "你好，你是谁？"
print(f"发送消息: {test_message}")
print()

# 调用API
headers = {
    "Authorization": f"Bearer {api_key}",
    "Content-Type": "application/json"
}

data = {
    "model": model,
    "messages": [
        {"role": "user", "content": test_message}
    ],
    "temperature": 0.7,
    "max_tokens": 100
}

try:
    print("正在连接API...")
    response = requests.post(
        "https://openrouter.ai/api/v1/chat/completions",
        headers=headers,
        json=data,
        timeout=30
    )
    
    response.raise_for_status()
    result = response.json()
    answer = result['choices'][0]['message']['content']
    
    print("="*60)
    print("✅ 连接成功！")
    print("="*60)
    print(f"LLM回复: {answer}")
    print("="*60)
    
except requests.exceptions.HTTPError as e:
    print("="*60)
    print(f"❌ HTTP错误: {e}")
    print(f"响应内容: {e.response.text}")
    print("="*60)
    
except Exception as e:
    print("="*60)
    print(f"❌ 连接失败: {e}")
    print("="*60)
