"""
LLM Planning and Reasoning Module
Responsible for subtask generation, completion verification, and navigation planning
"""
import json
import requests
import base64
import os
from datetime import datetime
from typing import Dict, List, Tuple, Optional
from Sub_vlm.llm_config import LLMConfig
from Sub_vlm.prompts import (
    get_initial_planning_prompt,
    get_verification_prompt,
    get_task_completion_prompt
)


class SubTask:
    """Subtask data structure"""
    
    def __init__(self, description: str, planning_hints: str, completion_criteria: str):
        """
        Args:
            description: Subtask description
            planning_hints: Planning hints
            completion_criteria: Completion criteria
        """
        self.description = description
        self.planning_hints = planning_hints
        self.completion_criteria = completion_criteria
    
    def to_dict(self):
        """Convert to dictionary"""
        return {
            "description": self.description,
            "planning_hints": self.planning_hints,
            "completion_criteria": self.completion_criteria
        }
    
    def __repr__(self):
        return f"SubTask(description='{self.description[:50]}...')"


class LLMPlanner:
    """LLM Planner - Responsible for subtask generation and verification"""
    
    def __init__(self, config_path="llm_config.yaml", log_dir="navigation_logs"):
        """
        Initialize planner
        
        Args:
            config_path: LLM configuration file path
            log_dir: Directory to save navigation logs and LLM responses
        """
        self.config = LLMConfig(config_path)
        self.log_dir = log_dir
        
        # Create log directory if it doesn't exist
        os.makedirs(log_dir, exist_ok=True)
        
        # Episode tracking
        self.current_episode_id = None
        self.episode_log = {
            "subtasks": [],
            "verifications": [],
            "completions": []
        }
        
        # Token usage tracking
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0
        self.total_tokens = 0
        self.api_call_count = 0
        
        print(f"✓ LLM Planner initialized: {self.config}")
        print(f"✓ Logs will be saved to: {log_dir}/")
    
    def start_episode(self, episode_id: str, instruction: str):
        """
        Start tracking a new episode
        
        Args:
            episode_id: Episode identifier
            instruction: Navigation instruction
        """
        self.current_episode_id = episode_id
        self.episode_log = {
            "episode_id": episode_id,
            "instruction": instruction,
            "start_time": datetime.now().isoformat(),
            "subtasks": [],
            "verifications": [],
            "completions": [],
            "token_usage": {
                "total_prompt_tokens": 0,
                "total_completion_tokens": 0,
                "total_tokens": 0,
                "api_calls": 0
            }
        }
        
        # Reset token counters
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0
        self.total_tokens = 0
        self.api_call_count = 0
        
        print(f"\n{'='*60}")
        print(f"📝 Started Episode: {episode_id}")
        print(f"📍 Instruction: {instruction}")
        print(f"{'='*60}\n")
    
    def save_episode_log(self, status: str = "completed"):
        """
        Save episode log to JSON file
        
        Args:
            status: Episode status ('completed', 'failed', 'timeout', etc.)
        """
        if self.current_episode_id is None:
            print("⚠️ Warning: No active episode to save")
            return
        
        # Update final info
        self.episode_log["end_time"] = datetime.now().isoformat()
        self.episode_log["status"] = status
        self.episode_log["token_usage"] = {
            "total_prompt_tokens": self.total_prompt_tokens,
            "total_completion_tokens": self.total_completion_tokens,
            "total_tokens": self.total_tokens,
            "api_calls": self.api_call_count
        }
        
        # Save to file
        log_file = os.path.join(self.log_dir, f"episode_{self.current_episode_id}.json")
        with open(log_file, 'w', encoding='utf-8') as f:
            json.dump(self.episode_log, f, indent=2, ensure_ascii=False)
        
        print(f"\n💾 Episode log saved: {log_file}")
        print(f"   - Total Subtasks: {len(self.episode_log['subtasks'])}")
        print(f"   - Total Verifications: {len(self.episode_log['verifications'])}")
        print(f"   - Total Completions: {len(self.episode_log['completions'])}")
        print(f"   - API Calls: {self.api_call_count}")
        print(f"   - Total Tokens: {self.total_tokens:,}")
        
        return log_file
    
    def encode_image_base64(self, image_path: str) -> str:
        """
        Encode image to base64
        
        Args:
            image_path: Image file path
            
        Returns:
            Base64 encoded image string
        """
        with open(image_path, "rb") as f:
            return base64.b64encode(f.read()).decode('utf-8')
    
    def _build_initial_planning_prompt(self, instruction: str, direction_names: List[str]) -> str:
        """
        Build initial planning prompt (at task start)
        
        Args:
            instruction: Complete navigation instruction
            direction_names: List of direction names (corresponding to 8 images)
            
        Returns:
            Prompt text
        """
        return get_initial_planning_prompt(instruction, direction_names)
    
    def _build_verification_prompt(self, 
                                   instruction: str,
                                   subtask: SubTask,
                                   direction_names: List[str]) -> str:
        """
        Build verification prompt (check if subtask is completed)
        
        Args:
            instruction: Complete navigation instruction
            subtask: Current subtask
            direction_names: List of direction names
            
        Returns:
            Prompt text
        """
        return get_verification_prompt(
            instruction,
            subtask.description,
            subtask.completion_criteria,
            subtask.planning_hints,
            direction_names
        )
    
    def _build_task_completion_prompt(self,
                                     instruction: str,
                                     direction_names: List[str]) -> str:
        """
        Build task completion check prompt
        
        Args:
            instruction: Complete navigation instruction
            direction_names: List of direction names
            
        Returns:
            Prompt text
        """
        return get_task_completion_prompt(instruction, direction_names)
    
    def _call_llm_api(self, 
                     prompt: str, 
                     image_paths: List[str]) -> Optional[Dict]:
        """
        Call LLM API
        
        Args:
            prompt: Text prompt
            image_paths: List of image file paths
            
        Returns:
            API response JSON data, None if failed
        """
        try:
            # Build message content
            content = [{"type": "text", "text": prompt}]
            
            # Add images
            for img_path in image_paths:
                img_base64 = self.encode_image_base64(img_path)
                content.append({
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{img_base64}"
                    }
                })
            
            # Build request
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
            
            # Send request
            print(f"\n🤖 Calling LLM API ({self.config.model})...")
            response = requests.post(
                f"{self.config.base_url}/chat/completions",
                headers=self.config.get_headers(),
                json=payload,
                timeout=self.config.timeout
            )
            
            response.raise_for_status()
            
            # Parse response
            result = response.json()
            content_text = result['choices'][0]['message']['content']
            
            # Track token usage
            if 'usage' in result:
                usage = result['usage']
                prompt_tokens = usage.get('prompt_tokens', 0)
                completion_tokens = usage.get('completion_tokens', 0)
                total = usage.get('total_tokens', 0)
                
                self.total_prompt_tokens += prompt_tokens
                self.total_completion_tokens += completion_tokens
                self.total_tokens += total
                self.api_call_count += 1
                
                print(f"📊 Token usage: {prompt_tokens} prompt + {completion_tokens} completion = {total} total")
            
            # Try to parse JSON
            # Remove possible markdown code block markers
            content_text = content_text.strip()
            if content_text.startswith("```json"):
                content_text = content_text[7:]
            if content_text.startswith("```"):
                content_text = content_text[3:]
            if content_text.endswith("```"):
                content_text = content_text[:-3]
            content_text = content_text.strip()
            
            # Try to parse JSON
            try:
                parsed_json = json.loads(content_text)
            except json.JSONDecodeError as e:
                # If parsing fails, try to extract the first complete JSON object
                print(f"⚠️ Initial JSON parsing failed: {e}")
                print(f"📝 Attempting to fix JSON format...")
                
                # Try to find the first complete JSON object
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
                                direction_names: List[str]) -> Optional[SubTask]:
        """
        Generate initial subtask (at task start)
        
        Args:
            instruction: Complete navigation instruction
            observation_images: List of 8 directional image paths
            direction_names: List of direction names
            
        Returns:
            SubTask object, None if failed
        """
        prompt = self._build_initial_planning_prompt(instruction, direction_names)
        
        response = self._call_llm_api(prompt, observation_images)
        
        if response is None:
            return None
        
        try:
            # Validate required fields
            required_fields = ['subtask_instruction', 'planning_hints', 'completion_criteria']
            missing_fields = [field for field in required_fields if field not in response]
            
            if missing_fields:
                print(f"✗ Response missing required fields: {', '.join(missing_fields)}")
                print(f"✗ Actual fields received: {list(response.keys())}")
                return None
            
            subtask = SubTask(
                description=response['subtask_instruction'],
                planning_hints=response['planning_hints'],
                completion_criteria=response['completion_criteria']
            )
            
            print(f"\n📋 Generated Subtask:")
            print(f"  Current Location: {response.get('current_location', 'N/A')}")
            print(f"  Instruction Sequence: {response.get('instruction_sequence', 'N/A')}")
            print(f"  Subtask Destination: {response.get('subtask_destination', 'N/A')}")
            print(f"  Subtask Instruction: {subtask.description}")
            print(f"  Planning Hints: {subtask.planning_hints[:100]}...")
            print(f"  Completion Criteria: {subtask.completion_criteria[:100]}...")
            if 'reasoning' in response:
                print(f"  Reasoning: {response['reasoning'][:100]}...")
            
            # Save to episode log
            if self.current_episode_id:
                subtask_log = {
                    "step": len(self.episode_log['subtasks']) + 1,
                    "timestamp": datetime.now().isoformat(),
                    "type": "initial_subtask",
                    "llm_response": response,  # Save complete LLM response
                    "subtask": subtask.to_dict()
                }
                self.episode_log['subtasks'].append(subtask_log)
                
                # Also save individual subtask file
                subtask_file = os.path.join(
                    self.log_dir, 
                    f"episode_{self.current_episode_id}_subtask_{len(self.episode_log['subtasks'])}.json"
                )
                with open(subtask_file, 'w', encoding='utf-8') as f:
                    json.dump(subtask_log, f, indent=2, ensure_ascii=False)
                print(f"  💾 Subtask saved to: {subtask_file}")
            
            return subtask
            
        except KeyError as e:
            print(f"✗ Field access error: {e}")
            print(f"✗ Actual fields received: {list(response.keys()) if response else 'None'}")
            return None
        except Exception as e:
            print(f"✗ Subtask creation failed: {e}")
            return None
    
    def verify_and_plan_next(self,
                            instruction: str,
                            current_subtask: SubTask,
                            observation_images: List[str],
                            direction_names: List[str]) -> Tuple[bool, Optional[SubTask], Optional[str]]:
        """
        Verify current subtask and plan next one
        
        Args:
            instruction: Complete navigation instruction
            current_subtask: Current subtask
            observation_images: List of 8 directional image paths
            direction_names: List of direction names
            
        Returns:
            (is_completed, next_subtask, advice)
            - is_completed: Whether current subtask is completed
            - next_subtask: Next subtask (if current is completed)
            - advice: Continuation advice (if not completed)
        """
        prompt = self._build_verification_prompt(
            instruction, current_subtask, direction_names
        )
        
        response = self._call_llm_api(prompt, observation_images)
        
        if response is None:
            return False, None, "API call failed, unable to verify"
        
        try:
            # Validate required fields
            if 'is_completed' not in response:
                print(f"✗ Response missing 'is_completed' field")
                print(f"✗ Actual fields received: {list(response.keys())}")
                return False, None, "Response format error: missing is_completed field"
            
            is_completed = response['is_completed']
            analysis = response.get('completion_analysis', 'No analysis provided')
            
            print(f"\n🔍 Subtask Verification Result:")
            print(f"  Completion Status: {'✓ Completed' if is_completed else '✗ Not Completed'}")
            print(f"  Analysis: {analysis[:150]}...")
            
            # Save to episode log
            verification_log = {
                "step": len(self.episode_log['verifications']) + 1 if self.current_episode_id else 0,
                "timestamp": datetime.now().isoformat(),
                "verified_subtask": current_subtask.to_dict(),
                "llm_response": response,  # Save complete LLM response
                "is_completed": is_completed
            }
            
            if is_completed:
                # Validate next_subtask field
                if 'next_subtask' not in response:
                    print(f"✗ Completed but missing 'next_subtask' field")
                    return False, None, "Response format error: completed but no next subtask"
                
                next_data = response['next_subtask']
                required_subtask_fields = ['subtask_instruction', 'planning_hints', 'completion_criteria']
                missing_fields = [field for field in required_subtask_fields if field not in next_data]
                
                if missing_fields:
                    print(f"✗ next_subtask missing fields: {', '.join(missing_fields)}")
                    return False, None, f"next_subtask format error: missing {', '.join(missing_fields)}"
                
                next_subtask = SubTask(
                    description=next_data['subtask_instruction'],
                    planning_hints=next_data['planning_hints'],
                    completion_criteria=next_data['completion_criteria']
                )
                
                print(f"\n📋 Next Subtask:")
                print(f"  Description: {next_subtask.description}")
                print(f"  Hints: {next_subtask.planning_hints[:100]}...")
                print(f"  Criteria: {next_subtask.completion_criteria[:100]}...")
                
                verification_log['next_subtask'] = next_subtask.to_dict()
                
                if self.current_episode_id:
                    self.episode_log['verifications'].append(verification_log)
                    
                    # Save individual verification file
                    verification_file = os.path.join(
                        self.log_dir,
                        f"episode_{self.current_episode_id}_verification_{len(self.episode_log['verifications'])}.json"
                    )
                    with open(verification_file, 'w', encoding='utf-8') as f:
                        json.dump(verification_log, f, indent=2, ensure_ascii=False)
                    print(f"  💾 Verification saved to: {verification_file}")
                
                return True, next_subtask, None
            else:
                advice = response.get('continuation_advice', 'Continue as planned')
                print(f"  Advice: {advice[:150]}...")
                
                verification_log['continuation_advice'] = advice
                
                if self.current_episode_id:
                    self.episode_log['verifications'].append(verification_log)
                    
                    # Save individual verification file
                    verification_file = os.path.join(
                        self.log_dir,
                        f"episode_{self.current_episode_id}_verification_{len(self.episode_log['verifications'])}.json"
                    )
                    with open(verification_file, 'w', encoding='utf-8') as f:
                        json.dump(verification_log, f, indent=2, ensure_ascii=False)
                    print(f"  💾 Verification saved to: {verification_file}")
                
                return False, None, advice
                
        except KeyError as e:
            print(f"✗ Field access error: {e}")
            print(f"✗ Actual fields received: {list(response.keys()) if response else 'None'}")
            return False, None, f"Field access error: {e}"
        except Exception as e:
            print(f"✗ Verification processing failed: {e}")
            return False, None, f"Processing exception: {e}"
    
    def check_task_completion(self,
                             instruction: str,
                             observation_images: List[str],
                             direction_names: List[str]) -> Tuple[bool, float, str]:
        """
        Check if the entire task is completed
        
        Args:
            instruction: Complete navigation instruction
            observation_images: List of 8 directional image paths
            direction_names: List of direction names
            
        Returns:
            (is_completed, confidence, analysis)
        """
        prompt = self._build_task_completion_prompt(instruction, direction_names)
        
        response = self._call_llm_api(prompt, observation_images)
        
        if response is None:
            return False, 0.0, "API call failed"
        
        try:
            # Validate required fields
            required_fields = ['task_completed', 'confidence', 'analysis']
            missing_fields = [field for field in required_fields if field not in response]
            
            if missing_fields:
                print(f"✗ Response missing required fields: {', '.join(missing_fields)}")
                print(f"✗ Actual fields received: {list(response.keys())}")
                return False, 0.0, f"Response format error: missing {', '.join(missing_fields)}"
            
            is_completed = response['task_completed']
            confidence = float(response['confidence'])  # Ensure conversion to float
            analysis = response['analysis']
            
            # Validate confidence range
            if not (0.0 <= confidence <= 1.0):
                print(f"⚠️ Confidence out of range: {confidence}, will clamp to [0.0, 1.0]")
                confidence = max(0.0, min(1.0, confidence))
            
            print(f"\n🎯 Task Completion Check:")
            print(f"  Status: {'✓ Completed' if is_completed else '✗ Not Completed'}")
            print(f"  Confidence: {confidence:.2%}")
            print(f"  Analysis: {analysis[:150]}...")
            
            if not is_completed and 'recommendation' in response and response['recommendation']:
                print(f"  Recommendation: {response['recommendation'][:150]}...")
            
            # Save to episode log
            if self.current_episode_id:
                completion_log = {
                    "step": len(self.episode_log['completions']) + 1,
                    "timestamp": datetime.now().isoformat(),
                    "llm_response": response,  # Save complete LLM response
                    "task_completed": is_completed,
                    "confidence": confidence
                }
                self.episode_log['completions'].append(completion_log)
                
                # Save individual completion check file
                completion_file = os.path.join(
                    self.log_dir,
                    f"episode_{self.current_episode_id}_completion_{len(self.episode_log['completions'])}.json"
                )
                with open(completion_file, 'w', encoding='utf-8') as f:
                    json.dump(completion_log, f, indent=2, ensure_ascii=False)
                print(f"  💾 Completion check saved to: {completion_file}")
            
            return is_completed, confidence, analysis
            
        except (KeyError, ValueError, TypeError) as e:
            print(f"✗ Field parsing error: {e}")
            print(f"✗ Actual fields received: {list(response.keys()) if response else 'None'}")
            return False, 0.0, f"Field parsing error: {e}"
        except Exception as e:
            print(f"✗ Task check failed: {e}")
            return False, 0.0, f"Processing exception: {e}"
