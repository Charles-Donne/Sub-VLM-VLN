"""
LLM Planning and Reasoning Module
Responsible for subtask generation, completion verification, and navigation planning

模块架构:
1. 初始子任务生成模块 (generate_initial_subtask)
   - 在任务开始时分析环境,生成第一个子任务
   - 输出: response字典(包含subtask_destination, subtask_instruction, planning_hints, completion_criteria)
   
2. 验证+再规划模块 (verify_and_replan)
   - 接收上一步的子任务字典(目的地、指令、完成约束条件)
   - 验证是否完成子任务
   - 如果完成: 生成下一个子任务
   - 如果未完成: 修改当前子任务指令,保持目的地不变
   - 输出: (response字典, is_completed)
   
3. 全局任务完成检查 (check_task_completion)
   - 检查整个导航任务是否完成
   - 输出: (response字典, is_completed, confidence, analysis)

使用流程:
Step 1: generate_initial_subtask() -> response字典
Step 2: 执行动作...
Step 3: verify_and_replan() -> (response字典, is_completed)
Step 4a: 如果completed=True -> response是新的子任务,回到Step 2
Step 4b: 如果completed=False -> response是修改后的当前子任务,回到Step 2
"""
import json
import requests
import base64
import os
from typing import Dict, List, Tuple, Optional
from Sub_vlm.llm_config import LLMConfig
from Sub_vlm.prompts import (
    get_initial_planning_prompt,
    get_verification_replanning_prompt,
    get_task_completion_prompt
)


class LLMPlanner:
    """LLM规划器 - 负责子任务生成和验证"""
    
    def __init__(self, config_path="llm_config.yaml", action_space: str = None):
        """
        初始化规划器
        
        Args:
            config_path: LLM配置文件路径
            action_space: 动作空间描述,如 "MOVE_FORWARD (0.25m), TURN_LEFT (45°), TURN_RIGHT (45°), STOP"
        """
        self.config = LLMConfig(config_path)
        self.action_space = action_space or "MOVE_FORWARD, TURN_LEFT, TURN_RIGHT, STOP"
        print(f"✓ LLM Planner initialized: {self.config}")
        print(f"✓ Action space: {self.action_space}")
    
    def encode_image_base64(self, image_path: str) -> str:
        """编码图像为base64"""
        with open(image_path, "rb") as f:
            return base64.b64encode(f.read()).decode('utf-8')
    
    def _build_initial_planning_prompt(self, instruction: str, direction_names: List[str]) -> str:
        """构建初始规划prompt"""
        return get_initial_planning_prompt(instruction, direction_names, self.action_space)
    
    def _build_verification_replanning_prompt(self, instruction: str, current_subtask: Dict, direction_names: List[str]) -> str:
        """构建验证+再规划prompt"""
        return get_verification_replanning_prompt(
            instruction,
            current_subtask['subtask_destination'],
            current_subtask['subtask_instruction'],
            current_subtask['completion_criteria'],
            direction_names,
            self.action_space
        )
    
    def _build_task_completion_prompt(self, instruction: str, direction_names: List[str]) -> str:
        """构建任务完成检查prompt"""
        return get_task_completion_prompt(instruction, direction_names)
    
    def _call_llm_api(self, prompt: str, image_paths: List[str]) -> Optional[Dict]:
        """
        调用LLM API
        
        Args:
            prompt: 文本提示
            image_paths: 图像文件路径列表
            
        Returns:
            API响应的JSON数据,失败时返回None
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
                "messages": [{"role": "user", "content": content}],
                "temperature": self.config.temperature,
                "max_tokens": self.config.max_tokens
            }
            
            # 发送请求
            print(f"\n🤖 Calling LLM API ({self.config.model})...")
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
            
            # 清理JSON标记
            content_text = content_text.strip()
            if content_text.startswith("```json"):
                content_text = content_text[7:]
            if content_text.startswith("```"):
                content_text = content_text[3:]
            if content_text.endswith("```"):
                content_text = content_text[:-3]
            content_text = content_text.strip()
            
            # 解析JSON
            try:
                parsed_json = json.loads(content_text)
            except json.JSONDecodeError as e:
                print(f"⚠️ Initial JSON parsing failed: {e}")
                print(f"📝 Attempting to fix JSON format...")
                
                # 尝试提取完整的JSON对象
                brace_count = 0
                json_end = -1
                for i, char in enumerate(content_text):
                    if char == '{':
                        brace_count += 1
                    elif char == '}':
                        brace_count -= 1
                        if brace_count == 0:
                            json_end = i + 1
                            break
                
                if json_end > 0:
                    content_text = content_text[:json_end]
                    try:
                        parsed_json = json.loads(content_text)
                        print("✓ JSON repair successful")
                    except json.JSONDecodeError:
                        print(f"✗ JSON repair failed")
                        print(f"Raw response: {content_text[:500]}...")
                        return None
                else:
                    print(f"✗ Cannot find complete JSON object")
                    print(f"Raw response: {content_text[:500]}...")
                    return None
            
            print("✓ LLM response parsed successfully")
            return parsed_json
            
        except requests.exceptions.RequestException as e:
            print(f"✗ API request failed: {e}")
            return None
        except Exception as e:
            print(f"✗ Unknown error: {e}")
            return None
    
    def generate_initial_subtask(self,
                                instruction: str,
                                observation_images: List[str],
                                direction_names: List[str]) -> Optional[Dict]:
        """
        生成初始子任务(任务开始时)
        
        Args:
            instruction: 完整导航指令
            observation_images: 8方向图像路径列表
            direction_names: 方向名称列表
            
        Returns:
            response_dict: 完整的LLM响应JSON字典,失败时返回None
        """
        prompt = self._build_initial_planning_prompt(instruction, direction_names)
        response = self._call_llm_api(prompt, observation_images)
        
        if response is None:
            return None
        
        try:
            # 验证必需字段
            required_fields = ['subtask_destination', 'subtask_instruction', 'planning_hints', 'completion_criteria']
            missing_fields = [field for field in required_fields if field not in response]
            
            if missing_fields:
                print(f"✗ Response missing required fields: {', '.join(missing_fields)}")
                print(f"✗ Actual fields received: {list(response.keys())}")
                return None
            
            return response
            
        except Exception as e:
            print(f"✗ Subtask parsing failed: {e}")
            return None
    
    def verify_and_replan(self,
                         instruction: str,
                         current_subtask: Dict,
                         observation_images: List[str],
                         direction_names: List[str]) -> Tuple[Optional[Dict], bool]:
        """
        验证+再规划模块 - 验证子任务完成并规划下一步
        
        此模块负责:
        1. 验证当前子任务是否完成(基于完成约束条件)
        2. 如果完成: 生成下一个子任务
        3. 如果未完成: 修改当前子任务指令,保持目的地不变
        
        Args:
            instruction: 完整导航指令(全局任务)
            current_subtask: 当前子任务字典(包含目的地、指令、完成约束条件)
            observation_images: 8方向图像路径列表
            direction_names: 方向名称列表
            
        Returns:
            (response_dict, is_completed)
            - response_dict: 完整的LLM响应JSON字典,失败时返回None
            - is_completed: 是否完成当前子任务
        """
        prompt = self._build_verification_replanning_prompt(instruction, current_subtask, direction_names)
        response = self._call_llm_api(prompt, observation_images)
        
        if response is None:
            return None, False
        
        try:
            # 验证必需字段
            required_fields = ['is_completed', 'subtask_destination', 'subtask_instruction', 
                             'planning_hints', 'completion_criteria']
            missing_fields = [field for field in required_fields if field not in response]
            
            if missing_fields:
                print(f"✗ Response missing required fields: {', '.join(missing_fields)}")
                print(f"✗ Actual fields received: {list(response.keys())}")
                return None, False
            
            # 提取验证结果
            is_completed = response['is_completed']
            
            return response, is_completed
                
        except Exception as e:
            print(f"✗ Verification and replanning failed: {e}")
            return None, False
    
    def check_task_completion(self,
                             instruction: str,
                             observation_images: List[str],
                             direction_names: List[str]) -> Tuple[Optional[Dict], bool, float, str]:
        """
        检查整个任务是否完成
        
        Args:
            instruction: 完整导航指令
            observation_images: 8方向图像路径列表
            direction_names: 方向名称列表
            
        Returns:
            (response_dict, is_completed, confidence, analysis)
            - response_dict: 完整的LLM响应JSON字典
            - is_completed: 是否完成任务
            - confidence: 完成的置信度(0-1)
            - analysis: 分析说明
        """
        prompt = self._build_task_completion_prompt(instruction, direction_names)
        response = self._call_llm_api(prompt, observation_images)
        
        if response is None:
            return None, False, 0.0, "API call failed"
        
        try:
            # 验证必需字段
            required_fields = ['task_completed', 'confidence', 'analysis']
            missing_fields = [field for field in required_fields if field not in response]
            
            if missing_fields:
                print(f"✗ Response missing required fields: {', '.join(missing_fields)}")
                print(f"✗ Actual fields received: {list(response.keys())}")
                return response, False, 0.0, f"Response format error: missing {', '.join(missing_fields)}"
            
            is_completed = response['task_completed']
            confidence = float(response['confidence'])
            analysis = response['analysis']
            
            # 验证confidence范围
            if not (0.0 <= confidence <= 1.0):
                print(f"⚠️ Confidence out of range: {confidence}, clamping to [0.0, 1.0]")
                confidence = max(0.0, min(1.0, confidence))
            
            return response, is_completed, confidence, analysis
            
        except (KeyError, ValueError, TypeError) as e:
            print(f"✗ Field parsing error: {e}")
            print(f"✗ Actual fields received: {list(response.keys()) if response else 'None'}")
            return response, False, 0.0, f"Field parsing error: {e}"
        except Exception as e:
            print(f"✗ Task check failed: {e}")
            return response, False, 0.0, f"Processing exception: {e}"
