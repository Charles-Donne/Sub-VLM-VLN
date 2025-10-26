#!/usr/bin/env python3
"""
导航指令分解模块
提供统一的指令分解接口，供 NaVid Agent 调用
"""

import os
import sys
import yaml
import json
import requests
from typing import Dict, List, Any, Optional

# 添加项目根目录到Python路径
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
sys.path.insert(0, project_root)


# ============================================================================
# 🔧 系统提示词配置
# ============================================================================

SYSTEM_PROMPT = """You are a Language Decomposer for an embodied navigation system.  
Your task is to break a natural language navigation instruction into multiple structured sub-instructions.  

CRITICAL REQUIREMENTS:
1. Each sub-instruction MUST maintain contextual continuity with previous steps
2. Each sub-instruction MUST include BOTH a landmark reference AND an action state
3. Never use vague commands like "turn left" - always specify WHERE to turn and WHAT to face
4. Example: Instead of "turn left", use "at the top of the stairs, turn left to face the double doors"
5. Each sub-instruction should be self-contained yet contextually aware

Output strictly in JSON format following this schema:

{
  "instruction_original": "<the original instruction>",
  "sub_instructions": [
    {
      "sub_id": <int>,
      "sub_instruction": "<context-rich action phrase with landmark and state>",
      "action_type": "<one of: move_forward | turn | enter | exit | stop | look | approach | navigate>",
      "target_landmark": "<main object or area - REQUIRED, never empty>",
      "spatial_relation": "<relation phrase if any, e.g. past / before / through / toward / at / beside>",
      "scene_transition": "<if environment changes, describe it; if staying in same area, state the context>",
      "completion_condition": "<how to determine this subtask is completed, referencing current position and orientation>"
    }
  ]
}

IMPORTANT:
- Return ONLY valid JSON, no markdown, no extra text
- Each sub-instruction must be atomic (one single action) but contextually complete
- sub_id starts from 1
- action_type must be one of the specified types
- target_landmark is MANDATORY - every action must reference a visible landmark
- sub_instruction field must describe WHERE you are, WHAT you do, and WHAT you should see/face
"""

USER_PROMPT_TEMPLATE = """Please decompose the following navigation instruction:

Instruction: {instruction}

Remember:
- Each sub-instruction must include BOTH landmark and action state
- Maintain contextual relationships between consecutive steps
- Be specific about WHERE actions happen and WHAT the agent should face
- Example of good sub-instruction: "At the bedroom doorway, turn right to face the bathroom entrance"
- Example of bad sub-instruction: "Turn right" (missing location context and target)

Return the structured decomposition in JSON format."""


# ============================================================================
# 📦 指令分解器类
# ============================================================================

class InstructionDecomposer:
    """
    导航指令分解器
    
    功能：
    1. 加载API配置
    2. 调用LLM分解指令
    3. 返回结构化的子指令列表
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """
        初始化分解器
        
        Args:
            config_path: API配置文件路径，如果为None则使用默认路径
        """
        self.config = self._load_config(config_path)
        self.api_available = self.config is not None
        
    
    def _load_config(self, config_path: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """加载API配置"""
        try:
            if config_path is None:
                config_path = os.path.join(script_dir, 'api_config.yaml')
            
            if not os.path.exists(config_path):
                print(f"⚠️  配置文件不存在: {config_path}")
                return None
            
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            
            # 验证必要字段
            if 'openrouter' not in config or 'api_key' not in config['openrouter']:
                print("⚠️  配置文件缺少必要字段")
                return None
            
            print("✅ 指令分解器初始化成功")
            return config
            
        except Exception as e:
            print(f"⚠️  加载配置失败: {e}")
            return None
    
    
    def decompose(self, instruction: str) -> List[Dict[str, Any]]:
        """
        分解导航指令为子指令序列
        
        Args:
            instruction: 原始导航指令
            
        Returns:
            子指令列表，每个元素包含以下字段：
            - sub_id: 子指令ID
            - sub_instruction: 子指令文本（包含上下文和地标）
            - action_type: 动作类型
            - target_landmark: 目标地标
            - spatial_relation: 空间关系
            - scene_transition: 场景转换描述
            - completion_condition: 完成条件
            
            如果分解失败，返回包含原始指令的单元素列表
        """
        if not self.api_available:
            # API不可用，返回原始指令
            return [{
                "sub_id": 1,
                "sub_instruction": instruction,
                "action_type": "navigate",
                "target_landmark": "destination",
                "spatial_relation": "",
                "scene_transition": "",
                "completion_condition": "reach the final destination"
            }]
        
        try:
            result = self._call_llm(instruction)
            
            # 验证返回结果
            if 'sub_instructions' not in result or not result['sub_instructions']:
                print("⚠️  LLM返回结果格式错误，使用原始指令")
                return self._fallback_to_original(instruction)
            
            sub_instructions = result['sub_instructions']
            print(f"✅ 指令分解成功：{len(sub_instructions)} 个子指令")
            
            return sub_instructions
            
        except Exception as e:
            print(f"⚠️  指令分解失败: {e}")
            return self._fallback_to_original(instruction)
    
    
    def _call_llm(self, instruction: str) -> Dict[str, Any]:
        """调用LLM API进行指令分解"""
        api_key = self.config['openrouter']['api_key']
        model = self.config['openrouter']['default_model']
        
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
            "temperature": self.config['openrouter'].get('temperature', 0.3),
            "max_tokens": self.config['openrouter'].get('max_tokens', 2000),
            "response_format": {"type": "json_object"}
        }
        
        response = requests.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers=headers,
            json=data,
            timeout=self.config['openrouter'].get('timeout', 30)
        )
        
        response.raise_for_status()
        result = response.json()
        
        # 解析返回的JSON
        llm_output = result['choices'][0]['message']['content']
        
        try:
            parsed_result = json.loads(llm_output)
            return parsed_result
        except json.JSONDecodeError as e:
            raise ValueError(f"LLM返回的不是有效的JSON: {e}")
    
    
    def _fallback_to_original(self, instruction: str) -> List[Dict[str, Any]]:
        """降级方案：返回原始指令"""
        return [{
            "sub_id": 1,
            "sub_instruction": instruction,
            "action_type": "navigate",
            "target_landmark": "destination",
            "spatial_relation": "",
            "scene_transition": "",
            "completion_condition": "reach the final destination"
        }]
    
    
    def is_available(self) -> bool:
        """检查分解器是否可用"""
        return self.api_available


# ============================================================================
# 🔧 便捷函数
# ============================================================================

def decompose_instruction(instruction: str, config_path: Optional[str] = None) -> List[Dict[str, Any]]:
    """
    便捷函数：分解导航指令
    
    Args:
        instruction: 原始导航指令
        config_path: API配置文件路径（可选）
        
    Returns:
        子指令列表
    """
    decomposer = InstructionDecomposer(config_path)
    return decomposer.decompose(instruction)


# ============================================================================
# 🧪 测试代码
# ============================================================================

if __name__ == "__main__":
    # 测试指令分解器
    test_instruction = "Go up stairs and turn left to stairs. Stop between stairs and large double doors."
    
    print("="*80)
    print("🧪 测试指令分解器")
    print("="*80)
    print(f"\n原始指令: {test_instruction}\n")
    
    # 创建分解器
    decomposer = InstructionDecomposer()
    
    if decomposer.is_available():
        print("✅ 分解器可用，开始分解...\n")
        sub_instructions = decomposer.decompose(test_instruction)
        
        print(f"共分解为 {len(sub_instructions)} 个子指令:\n")
        for sub in sub_instructions:
            print(f"[子指令 {sub['sub_id']}]")
            print(f"  ▸ 动作: {sub['sub_instruction']}")
            print(f"  ▸ 类型: {sub['action_type']}")
            print(f"  ▸ 地标: {sub['target_landmark']}")
            print(f"  ▸ 空间关系: {sub.get('spatial_relation', 'N/A')}")
            print(f"  ▸ 完成条件: {sub.get('completion_condition', 'N/A')}")
            print()
    else:
        print("❌ 分解器不可用（配置文件缺失或API密钥无效）")
        print("   将使用原始指令作为降级方案")
        sub_instructions = decomposer.decompose(test_instruction)
        print(f"\n降级结果: {sub_instructions}")
    
    print("="*80)
