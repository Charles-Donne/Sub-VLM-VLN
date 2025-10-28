"""
LLM规划与思考模块
负责子任务生成、完成验证和导航规划
"""
import json
import requests
import base64
from typing import Dict, List, Tuple, Optional
from llm_config import LLMConfig


class SubTask:
    """子任务数据结构"""
    
    def __init__(self, description: str, planning_hints: str, completion_criteria: str):
        """
        Args:
            description: 子任务描述
            planning_hints: 规划提示（辅助思考）
            completion_criteria: 完成判别标准
        """
        self.description = description
        self.planning_hints = planning_hints
        self.completion_criteria = completion_criteria
    
    def to_dict(self):
        """转换为字典"""
        return {
            "description": self.description,
            "planning_hints": self.planning_hints,
            "completion_criteria": self.completion_criteria
        }
    
    def __repr__(self):
        return f"SubTask(description='{self.description[:50]}...')"


class LLMPlanner:
    """LLM规划器 - 负责子任务生成和验证"""
    
    def __init__(self, config_path="llm_config.yaml"):
        """
        初始化规划器
        
        Args:
            config_path: LLM配置文件路径
        """
        self.config = LLMConfig(config_path)
        print(f"✓ LLM规划器初始化完成: {self.config}")
    
    def encode_image_base64(self, image_path: str) -> str:
        """
        将图像编码为base64
        
        Args:
            image_path: 图像文件路径
            
        Returns:
            base64编码的图像字符串
        """
        with open(image_path, "rb") as f:
            return base64.b64encode(f.read()).decode('utf-8')
    
    def _build_initial_planning_prompt(self, instruction: str, direction_names: List[str]) -> str:
        """
        构建初始规划prompt（任务开始时）
        
        Args:
            instruction: 完整导航指令
            direction_names: 方向名称列表（对应8张图片）
            
        Returns:
            prompt文本
        """
        prompt = f"""你是一个视觉-语言导航（VLN）任务的规划助手。你的任务是帮助智能体完成室内导航任务。

# 当前任务
完整导航指令: {instruction}

# 观察信息
我会提供8个方向的观察图像（从前方开始顺时针）：
{', '.join(direction_names)}

# 你的任务
基于完整指令和当前观察，生成第一个子任务。子任务应该是可完成的小步骤。

# 输出格式（JSON）
请严格按照以下JSON格式输出，不要添加任何其他文字：
{{
    "subtask_description": "具体的子任务描述，例如：向前走到走廊尽头的门口",
    "planning_hints": "完成子任务的规划提示，例如：保持直行，注意观察前方是否有门或墙壁。预计需要5-8步前进",
    "completion_criteria": "判断子任务完成的标准，例如：前方可见门框或墙壁距离小于2米",
    "reasoning": "你的推理过程，说明为什么选择这个子任务"
}}

注意事项：
1. 子任务要具体、可执行
2. 规划提示要包含方向、距离、标志物等关键信息
3. 完成标准要明确、可观察
4. 考虑智能体的实际能力：只能前进、左转、右转、停止
"""
        return prompt
    
    def _build_verification_prompt(self, 
                                   instruction: str,
                                   subtask: SubTask,
                                   direction_names: List[str]) -> str:
        """
        构建验证prompt（检查子任务是否完成）
        
        Args:
            instruction: 完整导航指令
            subtask: 当前子任务
            direction_names: 方向名称列表
            
        Returns:
            prompt文本
        """
        prompt = f"""你是一个视觉-语言导航（VLN）任务的验证助手。你需要判断当前子任务是否已完成。

# 任务背景
完整导航指令: {instruction}

# 当前子任务
- 描述: {subtask.description}
- 完成标准: {subtask.completion_criteria}
- 规划提示: {subtask.planning_hints}

# 当前观察
我会提供8个方向的观察图像（从前方开始顺时针）：
{', '.join(direction_names)}

# 你的任务
1. 判断当前子任务是否已完成
2. 如果已完成，生成下一个子任务
3. 如果未完成，给出继续完成的建议

# 输出格式（JSON）
请严格按照以下JSON格式输出，不要添加任何其他文字：
{{
    "is_completed": true/false,
    "completion_analysis": "分析子任务完成情况的详细说明",
    "next_subtask": {{
        "subtask_description": "下一个子任务描述（如果当前已完成）",
        "planning_hints": "规划提示",
        "completion_criteria": "完成标准"
    }},
    "continuation_advice": "如果未完成，给出继续完成的建议（如果已完成则为null）"
}}

注意事项：
1. 仔细对比观察图像和完成标准
2. 如果生成下一个子任务，要确保它与总体指令一致
3. 建议要具体、可操作
"""
        return prompt
    
    def _build_task_completion_prompt(self,
                                     instruction: str,
                                     direction_names: List[str]) -> str:
        """
        构建任务完成检查prompt
        
        Args:
            instruction: 完整导航指令
            direction_names: 方向名称列表
            
        Returns:
            prompt文本
        """
        prompt = f"""你是一个视觉-语言导航（VLN）任务的验证助手。你需要判断整个导航任务是否已完成。

# 完整导航指令
{instruction}

# 当前观察
我会提供8个方向的观察图像（从前方开始顺时针）：
{', '.join(direction_names)}

# 你的任务
判断智能体是否已经到达指令描述的目标位置。

# 输出格式（JSON）
请严格按照以下JSON格式输出，不要添加任何其他文字：
{{
    "task_completed": true/false,
    "confidence": 0.0-1.0,
    "analysis": "详细分析为什么认为任务已完成或未完成",
    "recommendation": "如果未完成，建议下一步做什么"
}}

注意事项：
1. 仔细对比当前观察和指令中描述的目标位置
2. 给出你的置信度评分
3. 分析要基于视觉证据
"""
        return prompt
    
    def _call_llm_api(self, 
                     prompt: str, 
                     image_paths: List[str]) -> Optional[Dict]:
        """
        调用LLM API
        
        Args:
            prompt: 文本prompt
            image_paths: 图像文件路径列表
            
        Returns:
            API响应的JSON数据，失败返回None
        """
        try:
            # 构建消息内容
            content = [{"type": "text", "text": prompt}]
            
            # 添加图像
            for img_path in image_paths:
                img_base64 = self.encode_image_base64(img_path)
                content.append({
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{img_base64}"
                    }
                })
            
            # 构建请求
            payload = {
                "model": self.config.model,
                "messages": [
                    {
                        "role": "user",
                        "content": content
                    }
                ],
                "temperature": self.config.temperature,
                "max_tokens": self.config.max_tokens
            }
            
            # 发送请求
            print(f"\n🤖 正在调用LLM API ({self.config.model})...")
            response = requests.post(
                f"{self.config.base_url}/chat/completions",
                headers=self.config.get_headers(),
                json=payload,
                timeout=self.config.timeout
            )
            
            response.raise_for_status()
            
            # 解析响应
            result = response.json()
            content_text = result['choices'][0]['message']['content']
            
            # 尝试解析JSON
            # 移除可能的markdown代码块标记
            content_text = content_text.strip()
            if content_text.startswith("```json"):
                content_text = content_text[7:]
            if content_text.startswith("```"):
                content_text = content_text[3:]
            if content_text.endswith("```"):
                content_text = content_text[:-3]
            content_text = content_text.strip()
            
            parsed_json = json.loads(content_text)
            print("✓ LLM响应解析成功")
            
            return parsed_json
            
        except requests.exceptions.RequestException as e:
            print(f"✗ API请求失败: {e}")
            return None
        except json.JSONDecodeError as e:
            print(f"✗ JSON解析失败: {e}")
            print(f"原始响应: {content_text[:500]}...")
            return None
        except Exception as e:
            print(f"✗ 未知错误: {e}")
            return None
    
    def generate_initial_subtask(self,
                                instruction: str,
                                observation_images: List[str],
                                direction_names: List[str]) -> Optional[SubTask]:
        """
        生成初始子任务（任务开始时）
        
        Args:
            instruction: 完整导航指令
            observation_images: 8个方向的图像路径列表
            direction_names: 方向名称列表
            
        Returns:
            SubTask对象，失败返回None
        """
        prompt = self._build_initial_planning_prompt(instruction, direction_names)
        
        response = self._call_llm_api(prompt, observation_images)
        
        if response is None:
            return None
        
        try:
            subtask = SubTask(
                description=response['subtask_description'],
                planning_hints=response['planning_hints'],
                completion_criteria=response['completion_criteria']
            )
            
            print(f"\n📋 生成的子任务:")
            print(f"  描述: {subtask.description}")
            print(f"  提示: {subtask.planning_hints}")
            print(f"  标准: {subtask.completion_criteria}")
            if 'reasoning' in response:
                print(f"  推理: {response['reasoning']}")
            
            return subtask
            
        except KeyError as e:
            print(f"✗ 响应缺少必要字段: {e}")
            return None
    
    def verify_and_plan_next(self,
                            instruction: str,
                            current_subtask: SubTask,
                            observation_images: List[str],
                            direction_names: List[str]) -> Tuple[bool, Optional[SubTask], Optional[str]]:
        """
        验证当前子任务并规划下一个
        
        Args:
            instruction: 完整导航指令
            current_subtask: 当前子任务
            observation_images: 8个方向的图像路径列表
            direction_names: 方向名称列表
            
        Returns:
            (is_completed, next_subtask, advice)
            - is_completed: 当前子任务是否完成
            - next_subtask: 下一个子任务（如果当前已完成）
            - advice: 继续完成的建议（如果未完成）
        """
        prompt = self._build_verification_prompt(
            instruction, current_subtask, direction_names
        )
        
        response = self._call_llm_api(prompt, observation_images)
        
        if response is None:
            return False, None, "API调用失败，无法验证"
        
        try:
            is_completed = response['is_completed']
            analysis = response['completion_analysis']
            
            print(f"\n🔍 子任务验证结果:")
            print(f"  完成状态: {'✓ 已完成' if is_completed else '✗ 未完成'}")
            print(f"  分析: {analysis}")
            
            if is_completed:
                next_data = response['next_subtask']
                next_subtask = SubTask(
                    description=next_data['subtask_description'],
                    planning_hints=next_data['planning_hints'],
                    completion_criteria=next_data['completion_criteria']
                )
                
                print(f"\n📋 下一个子任务:")
                print(f"  描述: {next_subtask.description}")
                print(f"  提示: {next_subtask.planning_hints}")
                print(f"  标准: {next_subtask.completion_criteria}")
                
                return True, next_subtask, None
            else:
                advice = response.get('continuation_advice', '继续按计划执行')
                print(f"  建议: {advice}")
                return False, None, advice
                
        except KeyError as e:
            print(f"✗ 响应缺少必要字段: {e}")
            return False, None, "响应格式错误"
    
    def check_task_completion(self,
                             instruction: str,
                             observation_images: List[str],
                             direction_names: List[str]) -> Tuple[bool, float, str]:
        """
        检查整个任务是否完成
        
        Args:
            instruction: 完整导航指令
            observation_images: 8个方向的图像路径列表
            direction_names: 方向名称列表
            
        Returns:
            (is_completed, confidence, analysis)
        """
        prompt = self._build_task_completion_prompt(instruction, direction_names)
        
        response = self._call_llm_api(prompt, observation_images)
        
        if response is None:
            return False, 0.0, "API调用失败"
        
        try:
            is_completed = response['task_completed']
            confidence = response['confidence']
            analysis = response['analysis']
            
            print(f"\n🎯 任务完成检查:")
            print(f"  状态: {'✓ 已完成' if is_completed else '✗ 未完成'}")
            print(f"  置信度: {confidence:.2%}")
            print(f"  分析: {analysis}")
            
            if not is_completed and 'recommendation' in response:
                print(f"  建议: {response['recommendation']}")
            
            return is_completed, confidence, analysis
            
        except KeyError as e:
            print(f"✗ 响应缺少必要字段: {e}")
            return False, 0.0, "响应格式错误"
