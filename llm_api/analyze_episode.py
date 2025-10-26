#!/usr/bin/env python3
"""
Episode Instruction 分析工具 - 精简版
随机或指定episode，提取instruction并交给LLM分析
"""

import argparse
import os
import sys
import yaml
import requests
import random

# 添加项目根目录到Python路径，以便导入VLN_CE模块
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
sys.path.insert(0, project_root)

from habitat.datasets import make_dataset
from VLN_CE.vlnce_baselines.config.default import get_config


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


def load_config():
    """加载API配置（从llm_api/api_config.yaml读取）"""
    # 获取项目根目录
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    config_path = os.path.join(project_root, 'NaVid-VLN-CE', 'llm_api', 'api_config.yaml')
    
    if not os.path.exists(config_path):
        raise FileNotFoundError(
            f"找不到配置文件: {config_path}\n"
            f"请在 llm_api/ 目录创建 api_config.yaml 文件"
        )
    
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def analyze_with_llm(instruction, config):
    """调用LLM分析指令"""
    api_key = config['openrouter']['api_key']
    model = config['openrouter']['default_model']
    
    # 构建提示词
    user_prompt = USER_PROMPT_TEMPLATE.format(instruction=instruction)
    
    # 调用API
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    
    data = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt}
        ],
        "temperature": config['openrouter'].get('temperature', 0.7),
        "max_tokens": config['openrouter'].get('max_tokens', 1000)
    }
    
    response = requests.post(
        "https://openrouter.ai/api/v1/chat/completions",
        headers=headers,
        json=data,
        timeout=config['openrouter'].get('timeout', 30)
    )
    
    response.raise_for_status()
    return response.json()['choices'][0]['message']['content']


def main():
    parser = argparse.ArgumentParser(description='Episode Instruction分析工具')
    parser.add_argument('--config', default='VLN_CE/vlnce_baselines/config/r2r_baselines/navid_r2r.yaml',
                       help='VLN-CE配置文件（相对于项目根目录的路径）')
    parser.add_argument('--episode-id', type=str, help='指定episode ID（不指定则随机选取）')
    parser.add_argument('--analyze', '-a', action='store_true', help='使用LLM分析')
    
    args = parser.parse_args()
    
    # 切换到项目根目录（重要！）
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    os.chdir(project_root)
    
    # 配置文件路径（现在已经在项目根目录了）
    config_file = args.config
    
    # 加载数据集
    print("加载数据集...")
    vln_config = get_config(config_file)
    dataset = make_dataset(
        id_dataset=vln_config.TASK_CONFIG.DATASET.TYPE,
        config=vln_config.TASK_CONFIG.DATASET
    )
    
    # 选择episode
    if args.episode_id:
        episodes = [ep for ep in dataset.episodes if str(ep.episode_id) == args.episode_id]
        if not episodes:
            print(f"❌ 未找到episode ID: {args.episode_id}")
            return
        episode = episodes[0]
        print(f"✅ 选择指定的episode: {args.episode_id}")
    else:
        episode = random.choice(dataset.episodes)
        print(f"✅ 随机选择episode: {episode.episode_id}")
    
    # 提取信息
    print(f"\n{'='*80}")
    print(f"Episode ID: {episode.episode_id}")
    print(f"Scene ID: {episode.scene_id}")
    
    # 获取instruction
    if hasattr(episode, 'instruction'):
        if hasattr(episode.instruction, 'instruction_text'):
            instruction = episode.instruction.instruction_text
        else:
            instruction = str(episode.instruction)
    else:
        instruction = "无指令"
    
    print(f"Instruction: {instruction}")
    print(f"{'='*80}\n")
    
    # LLM分析
    if args.analyze:
        if not instruction or instruction == "无指令":
            print("❌ 无有效指令可分析")
            return
        
        print("🤖 LLM分析中...\n")
        api_config = load_config()
        
        try:
            analysis = analyze_with_llm(instruction, api_config)
            print(analysis)
        except Exception as e:
            print(f"❌ 分析失败: {e}")


if __name__ == "__main__":
    main()
