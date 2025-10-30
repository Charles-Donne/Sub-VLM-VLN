"""
LLM辅助的人工控制程序
结合大模型规划和人工执行的导航系统
"""
import os
import sys
import json
import cv2
from typing import Dict, List, Tuple, Optional
from habitat import Env
from habitat.utils.visualizations import maps

from VLN_CE.vlnce_baselines.config.default import get_config

from Sub_vlm.thinking import LLMPlanner, SubTask
from Sub_vlm.observation_collector import ObservationCollector


class LLMAssistedController:
    """LLM辅助控制器"""
    
    def __init__(self, output_dir: str, llm_config_path: str = "Sub_vlm/llm_config.yaml"):
        """
        初始化
        
        Args:
            output_dir: 输出目录
            llm_config_path: LLM配置文件路径
        """
        self.output_dir = output_dir
        self.step_count = 0
        self.subtask_count = 0
        self.episode_id = None
        
        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)
        
        # 初始化LLM规划器（带保存目录）
        llm_save_dir = os.path.join(output_dir, "llm_outputs")
        self.planner = LLMPlanner(llm_config_path, save_dir=llm_save_dir)
        
        # 当前子任务
        self.current_subtask = None
        
        # 当前子任务的动作序列
        self.current_subtask_actions = []
    
    def reset(self, episode_id: str, instruction: str):
        """
        重置（开始新episode）
        
        Args:
            episode_id: episode ID
            instruction: 导航指令
        """
        self.episode_id = episode_id
        self.step_count = 0
        self.subtask_count = 0
        self.current_subtask = None
        self.current_subtask_actions = []
        
        # 为当前episode创建文件夹
        self.episode_dir = os.path.join(self.output_dir, f"episode_{episode_id}")
        os.makedirs(self.episode_dir, exist_ok=True)
        
        # 创建子文件夹
        self.observations_dir = os.path.join(self.episode_dir, "observations")
        self.subtasks_dir = os.path.join(self.episode_dir, "subtasks")
        self.compass_dir = os.path.join(self.episode_dir, "compass_views")
        
        os.makedirs(self.observations_dir, exist_ok=True)
        os.makedirs(self.subtasks_dir, exist_ok=True)
        os.makedirs(self.compass_dir, exist_ok=True)
        
        print(f"\n{'='*80}")
        print(f"Episode Started: {episode_id}")
        print(f"Output Directory: {self.episode_dir}")
        print(f"{'='*80}\n")
    
    def observe_environment(self, observations: Dict, phase: str = "initial") -> Tuple[List[str], List[str]]:
        """
        观察环境（收集8方向图像）
        
        Args:
            observations: 环境观测
            phase: 阶段标识（initial/subtask_N/verification_N）
            
        Returns:
            (image_paths, direction_names)
        """
        print(f"\n{'='*60}")
        print(f"🔍 Observing Environment - {phase}")
        print(f"{'='*60}")
        
        # 创建观察收集器
        obs_dir = os.path.join(self.observations_dir, phase)
        collector = ObservationCollector(obs_dir)
        
        # 收集8方向图像
        image_paths, direction_names = collector.collect_8_directions(
            observations, 
            save_prefix=f"{phase}_step{self.step_count}"
        )
        
        # 创建罗盘可视化
        compass_path = os.path.join(self.compass_dir, f"{phase}_step{self.step_count}_compass.jpg")
        collector.create_compass_visualization(observations, compass_path)
        
        print(collector.get_direction_summary(observations))
        
        return image_paths, direction_names
    
    def generate_initial_subtask(self, instruction: str, observations: Dict) -> SubTask:
        """
        生成初始子任务
        
        Args:
            instruction: 完整导航指令
            observations: 初始观察
            
        Returns:
            SubTask对象
        """
        print(f"\n{'*'*80}")
        print("🤖 LLM Planning - Generate Initial Subtask")
        print(f"{'*'*80}")
        print(f"Instruction: {instruction}\n")
        
        # 收集观察
        image_paths, direction_names = self.observe_environment(observations, "initial")
        
        # 调用LLM生成子任务（自动保存完整输出）
        subtask = self.planner.generate_initial_subtask(
            instruction,
            image_paths,
            direction_names,
            save_filename=f"episode_{self.episode_id}_subtask_1.json"
        )
        
        if subtask is None:
            print("✗ Subtask generation failed, using default")
            subtask = SubTask(
                description="Explore forward and find the first landmark",
                planning_hints="Keep moving forward, observe surroundings",
                completion_criteria="See the first object or location mentioned"
            )
        
        # 保存子任务
        self.current_subtask = subtask
        self.subtask_count += 1
        self.current_subtask_actions = []
        
        # 保存子任务基本信息
        self._save_subtask_immediate(subtask)
        
        return subtask
    
    def _save_subtask_immediate(self, subtask: SubTask):
        """
        立即保存子任务信息（生成时保存）
        
        Args:
            subtask: 子任务对象
        """
        subtask_data = {
            "subtask_id": self.subtask_count,
            "generated_at_step": self.step_count,
            "description": subtask.description,
            "planning_hints": subtask.planning_hints,
            "completion_criteria": subtask.completion_criteria,
            "actions": []  # 动作列表，在子任务完成时更新
        }
        
        filepath = os.path.join(self.subtasks_dir, f"subtask_{self.subtask_count}.json")
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(subtask_data, f, indent=2, ensure_ascii=False)
        
        print(f"💾 Subtask saved: {filepath}")
    
    def verify_subtask_completion(self, instruction: str, observations: Dict) -> Tuple[bool, Optional[SubTask]]:
        """
        验证子任务完成情况
        
        Args:
            instruction: 完整导航指令
            observations: 当前观察
            
        Returns:
            (is_completed, next_subtask)
        """
        print(f"\n{'*'*80}")
        print("🤖 LLM Verification - Check Subtask Completion")
        print(f"{'*'*80}")
        print(f"Current Subtask: {self.current_subtask.description}\n")
        
        # 收集观察
        phase = f"verification_subtask{self.subtask_count}"
        image_paths, direction_names = self.observe_environment(observations, phase)
        
        # 调用LLM验证（自动保存完整输出）
        is_completed, next_subtask, advice = self.planner.verify_and_plan_next(
            instruction,
            self.current_subtask,
            image_paths,
            direction_names,
            save_filename=f"episode_{self.episode_id}_verification_{self.subtask_count}.json"
        )
        
        if is_completed and next_subtask:
            # 保存当前子任务的所有动作
            self._save_subtask_actions_on_completion()
            
            # 更新当前子任务
            self.current_subtask = next_subtask
            self.subtask_count += 1
            self.current_subtask_actions = []
            
            # 保存新子任务
            self._save_subtask_immediate(next_subtask)
            
        return is_completed, next_subtask
    
    def _save_subtask_actions_on_completion(self):
        """
        子任务完成时，更新该子任务文件，添加所有动作信息
        """
        if self.subtask_count == 0:
            return
        
        filepath = os.path.join(self.subtasks_dir, f"subtask_{self.subtask_count}.json")
        
        # 读取现有文件
        if os.path.exists(filepath):
            with open(filepath, "r", encoding="utf-8") as f:
                subtask_data = json.load(f)
            
            # 更新动作列表和完成信息
            subtask_data["actions"] = self.current_subtask_actions
            subtask_data["completed"] = True
            subtask_data["completion_step"] = self.step_count
            subtask_data["total_actions"] = len(self.current_subtask_actions)
            
            # 重新保存
            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(subtask_data, f, indent=2, ensure_ascii=False)
            
            print(f"✓ Subtask {self.subtask_count} actions saved: {len(self.current_subtask_actions)} actions")
    
    
    def display_current_subtask(self):
        """显示当前子任务信息"""
        if self.current_subtask is None:
            print("\n⚠️  No active subtask")
            return
        
        print(f"\n{'='*80}")
        print(f"📋 Current Subtask #{self.subtask_count}")
        print(f"{'='*80}")
        print(f"Instruction: {self.current_subtask.description}")
        print(f"\nPlanning Hints:")
        print(f"  {self.current_subtask.planning_hints}")
        print(f"\nCompletion Criteria:")
        print(f"  {self.current_subtask.completion_criteria}")
        print(f"Actions in this subtask: {len(self.current_subtask_actions)}")
        print(f"{'='*80}\n")
    
    def record_step_action(self, action_name: str, action_id: int, info: Dict):
        """
        记录步骤动作到当前子任务（不立即保存）
        
        Args:
            action_name: 动作名称
            action_id: 动作ID
            info: 环境信息
        """
        action_data = {
            "step": self.step_count,
            "action": {
                "name": action_name,
                "id": action_id
            },
            "metrics": {
                "distance_to_goal": info.get("distance_to_goal", -1),
                "path_length": info.get("path_length", 0)
            }
        }
        
        self.current_subtask_actions.append(action_data)
        print(f"  → Action recorded: {action_name} (total: {len(self.current_subtask_actions)})")


def run_llm_assisted_control(config_path: str, 
                             output_dir: str = "./llm_control_output",
                             llm_config_path: str = "llm_config.yaml"):
    """
    运行LLM辅助的人工控制程序
    
    Args:
        config_path: Habitat配置文件路径
        output_dir: 输出目录
        llm_config_path: LLM配置文件路径
    """
    print("="*80)
    print("LLM-Assisted Habitat Navigation Control")
    print("="*80)
    
    # 加载配置
    print(f"\n1. Loading Configuration")
    print(f"   - Habitat Config: {config_path}")
    print(f"   - LLM Config: {llm_config_path}")
    
    if not os.path.exists(config_path):
        print(f"✗ Error: Habitat config not found: {config_path}")
        return
    
    config = get_config(config_path)
    
    # 初始化环境
    print("\n2. Initializing Habitat Environment...")
    try:
        env = Env(config.TASK_CONFIG)
        print(f"   ✓ Environment initialized")
        print(f"   - Available Episodes: {len(env.episodes)}")
        
        if len(env.episodes) == 0:
            print("   ⚠️  Warning: No episodes found")
            return
            
    except Exception as e:
        print(f"   ✗ Initialization failed: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # 初始化控制器
    print("\n3. Initializing LLM Controller...")
    controller = LLMAssistedController(output_dir, llm_config_path)
    
    # 获取动作参数
    forward_step_size = config.TASK_CONFIG.SIMULATOR.FORWARD_STEP_SIZE
    turn_angle = config.TASK_CONFIG.SIMULATOR.TURN_ANGLE
    
    # 动作字典
    action_dict = {
        "0": (f"STOP", 0),
        "1": (f"MOVE_FORWARD ({forward_step_size}m)", 1),
        "2": (f"TURN_LEFT ({turn_angle}°)", 2),
        "3": (f"TURN_RIGHT ({turn_angle}°)", 3)
    }
    
    print(f"\n   Action Parameters:")
    print(f"   - Forward Step: {forward_step_size}m")
    print(f"   - Turn Angle: {turn_angle}°")
    
    # 选择episode
    print(f"\n4. Select Episode")
    episode_index = 0
    selected_episode = env.episodes[episode_index]
    
    env._current_episode = selected_episode
    observations = env.reset()
    
    episode_id = env.current_episode.episode_id
    instruction = observations["instruction"]["text"]
    
    # 重置控制器
    controller.reset(episode_id, instruction)
    
    # 显示episode信息
    print(f"\nEpisode ID: {episode_id}")
    print(f"Scene: {env.current_episode.scene_id}")
    print(f"Instruction: {instruction}")
    print(f"Initial Distance: {env.get_metrics()['distance_to_goal']:.2f}m")
    
    # 生成初始子任务
    subtask = controller.generate_initial_subtask(instruction, observations)
    controller.display_current_subtask()
    
    input("\n[Press Enter to start...]")
    
    # 主循环
    while not env.episode_over:
        
        # 显示当前状态
        info = env.get_metrics()
        print(f"\n{'-'*80}")
        print(f"Step {controller.step_count}")
        print(f"Distance to Goal: {info['distance_to_goal']:.2f}m")
        print(f"Path Length: {info['path_length']:.2f}m")
        print(f"{'-'*80}")
        
        # 显示当前子任务
        controller.display_current_subtask()
        
        # 显示动作选项
        print("\nAvailable Actions:")
        for key, (name, _) in action_dict.items():
            print(f"  {key}: {name}")
        print("  c: Complete subtask, request LLM verification")
        print("  q: End episode")
        print("  exit: Exit program")
        
        # 获取用户输入
        user_input = input("\nEnter action: ").strip().lower()
        
        # 处理特殊命令
        if user_input == "exit":
            print("\nExiting...")
            return
        
        if user_input == "q":
            print("\nEnding episode...")
            observations = env.step({"action": 0})
            controller.step_count += 1
            break
        
        # 完成子任务验证
        if user_input == "c":
            print("\nRequesting LLM verification...")
            
            is_completed, next_subtask = controller.verify_subtask_completion(
                instruction, observations
            )
            
            if is_completed:
                print("\n✓ Subtask completed!")
                if next_subtask:
                    print("\n→ Next subtask generated")
                    controller.display_current_subtask()
                else:
                    print("\n→ Approaching goal, continue navigation")
            else:
                print("\n✗ Subtask not completed yet")
            
            input("\n[Press Enter to continue...]")
            continue
        
        # 验证动作输入
        if user_input not in action_dict:
            print("⚠️  Invalid input")
            continue
        
        # 获取动作
        action_name, action_id = action_dict[user_input]
        
        # 记录动作（不立即保存）
        controller.record_step_action(action_name, action_id, info)
        
        # 执行动作
        print(f"\nExecuting: {action_name}")
        observations = env.step({"action": action_id})
        controller.step_count += 1
        
        # 检查STOP
        if action_id == 0:
            print("\n✓ STOP executed, episode ended")
            break
    
    # Episode结束
    final_metrics = env.get_metrics()
    
    # 保存最后一个子任务的动作
    if len(controller.current_subtask_actions) > 0:
        controller._save_subtask_actions_on_completion()
        print("\n✓ Final subtask actions saved")
    
    # 显示结果
    print(f"\n{'='*80}")
    print(f"Episode {episode_id} Results")
    print(f"{'='*80}")
    print(f"Total Steps: {controller.step_count}")
    print(f"Total Subtasks: {controller.subtask_count}")
    print(f"Final Distance: {final_metrics['distance_to_goal']:.2f}m")
    print(f"Success: {'Yes' if final_metrics['success'] else 'No'}")
    print(f"SPL: {final_metrics['spl']:.4f}")
    print(f"Path Length: {final_metrics['path_length']:.2f}m")
    print(f"\n✓ All results saved to: {controller.episode_dir}")


if __name__ == "__main__":
    print("\nLLM-Assisted Habitat Navigation Control")
    print("Combining LLM Planning with Manual Execution\n")
    
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python llm_manual_control.py <habitat_config> [output_dir] [llm_config]")
        print("\nExample:")
        print("  python llm_manual_control.py ../VLN_CE/habitat_extensions/config/vlnce_task_enhanced.yaml")
        print("  python llm_manual_control.py ../VLN_CE/habitat_extensions/config/vlnce_task_enhanced.yaml ./output llm_config.yaml")
        print("")
    else:
        habitat_config = sys.argv[1]
        output_dir = sys.argv[2] if len(sys.argv) > 2 else "./llm_control_output"
        llm_config = sys.argv[3] if len(sys.argv) > 3 else "./Sub_vlm/llm_config.yaml"
        
        run_llm_assisted_control(habitat_config, output_dir, llm_config)
