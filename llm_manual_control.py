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
        """初始化控制器"""
        self.output_dir = output_dir
        self.step_count = 0
        self.subtask_count = 0
        self.episode_id = None
        self.instruction = None
        
        os.makedirs(output_dir, exist_ok=True)
        self.planner = LLMPlanner(llm_config_path)
        
        self.current_subtask = None
        self.current_subtask_file = None
    
    def reset(self, episode_id: str, instruction: str):
        """重置episode"""
        self.episode_id = episode_id
        self.instruction = instruction
        self.step_count = 0
        self.subtask_count = 0
        self.current_subtask = None
        self.current_subtask_file = None
        
        # Episode目录
        self.episode_dir = os.path.join(self.output_dir, f"episode_{episode_id}")
        self.observations_dir = os.path.join(self.episode_dir, "observations")
        self.subtasks_dir = os.path.join(self.episode_dir, "subtasks")
        
        os.makedirs(self.observations_dir, exist_ok=True)
        os.makedirs(self.subtasks_dir, exist_ok=True)
    
    def observe_environment(self, observations: Dict, phase: str) -> Tuple[List[str], List[str]]:
        """收集8方向图像"""
        obs_dir = os.path.join(self.observations_dir, phase)
        collector = ObservationCollector(obs_dir)
        
        image_paths, direction_names = collector.collect_8_directions(
            observations, 
            save_prefix=f"{phase}_step{self.step_count}"
        )
        
        return image_paths, direction_names
    
    def save_first_person_view(self, observations: Dict, phase: str):
        """保存第一人称视角（front方向RGB）"""
        obs_dir = os.path.join(self.observations_dir, phase)
        os.makedirs(obs_dir, exist_ok=True)
        
        if "rgb" in observations:
            filename = f"step{self.step_count}_first_person.jpg"
            filepath = os.path.join(obs_dir, filename)
            cv2.imwrite(filepath, cv2.cvtColor(observations["rgb"], cv2.COLOR_RGB2BGR))
    
    def generate_initial_subtask(self, observations: Dict) -> SubTask:
        """生成初始子任务"""
        print(f"\n{'*'*80}")
        print("🤖 Generating Initial Subtask")
        print(f"{'*'*80}")
        
        # 收集观察
        image_paths, direction_names = self.observe_environment(observations, "initial")
        
        # 调用LLM
        response, subtask = self.planner.generate_initial_subtask(
            self.instruction,
            image_paths,
            direction_names
        )
        
        if not response or not subtask:
            print("✗ LLM call failed")
            return None
        
        # 保存子任务
        self.current_subtask = subtask
        self.subtask_count += 1
        
        # 创建子任务文件
        subtask_name = "initial_subtask"
        self._create_subtask_file(subtask_name, response, subtask)
        
        # 打印关键信息
        print(f"\n✅ Initial Subtask Generated")
        print(f"📋 Destination: {subtask.destination}")
        print(f"📋 Instruction: {subtask.instruction}")
        
        return subtask
    
    def _create_subtask_file(self, subtask_name: str, response: Dict, subtask: SubTask):
        """创建子任务文件"""
        subtask_data = {
            "global_instruction": self.instruction,
            "subtask_id": self.subtask_count,
            "subtask_name": subtask_name,
            "generated_at_step": self.step_count,
            "llm_response": response,
            "subtask": subtask.to_dict(),
            "actions": []
        }
        
        filepath = os.path.join(self.subtasks_dir, f"{subtask_name}.json")
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(subtask_data, f, indent=2, ensure_ascii=False)
        
        self.current_subtask_file = filepath
        print(f"💾 Subtask saved: {filepath}")
    
    def verify_and_replan(self, observations: Dict) -> Tuple[bool, Optional[SubTask]]:
        """验证+再规划模块"""
        print(f"\n{'*'*80}")
        print("🤖 Verification + Replanning")
        print(f"{'*'*80}")
        
        # 收集观察
        phase = f"verify_replan_{self.subtask_count}"
        image_paths, direction_names = self.observe_environment(observations, phase)
        
        # 调用LLM
        response, is_completed, next_subtask = self.planner.verify_and_replan(
            self.instruction,
            self.current_subtask,
            image_paths,
            direction_names
        )
        
        if not response:
            print("✗ LLM call failed")
            return False, None
        
        # 打印结果
        if is_completed and next_subtask:
            print(f"\n✅ Subtask Completed - Next Subtask Generated")
            print(f"📋 New Destination: {next_subtask.destination}")
            print(f"📋 New Instruction: {next_subtask.instruction}")
            
            # 保存新子任务
            self.current_subtask = next_subtask
            self.subtask_count += 1
            subtask_name = f"subtask_{self.subtask_count}"
            self._create_subtask_file(subtask_name, response, next_subtask)
            
        elif not is_completed and next_subtask:
            print(f"\n🔄 Subtask Not Completed - Instruction Refined")
            print(f"📋 Refined Instruction: {next_subtask.instruction}")
            
            # 更新当前子任务
            self.current_subtask = next_subtask
            self._update_subtask_file(response, next_subtask)
        
        return is_completed, next_subtask
    
    def _update_subtask_file(self, response: Dict, subtask: SubTask):
        """更新子任务文件（指令refinement）"""
        if not self.current_subtask_file or not os.path.exists(self.current_subtask_file):
            return
        
        with open(self.current_subtask_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        data["llm_response"] = response
        data["subtask"] = subtask.to_dict()
        
        with open(self.current_subtask_file, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
    
    def record_action(self, action_name: str, action_id: int, info: Dict):
        """记录动作并立即保存"""
        if not self.current_subtask_file:
            return
        
        action_data = {
            "step": self.step_count,
            "action_name": action_name,
            "action_id": action_id,
            "distance_to_goal": info.get("distance_to_goal", -1)
        }
        
        # 读取文件
        with open(self.current_subtask_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        # 添加动作
        data["actions"].append(action_data)
        
        # 立即保存
        with open(self.current_subtask_file, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
    
    
    def display_current_subtask(self):
        """显示当前子任务"""
        if not self.current_subtask:
            print("\n⚠️  No active subtask")
            return
        
        print(f"\n{'='*60}")
        print(f"📋 Current Subtask #{self.subtask_count}")
        print(f"Destination: {self.current_subtask.destination}")
        print(f"Instruction: {self.current_subtask.instruction}")
        print(f"{'='*60}\n")


def run_llm_assisted_control(config_path: str, 
                             output_dir: str = "./llm_control_output",
                             llm_config_path: str = "llm_config.yaml"):
    """运行LLM辅助导航控制"""
    print("="*60)
    print("LLM-Assisted Navigation Control")
    print("="*60)
    
    # 加载配置
    if not os.path.exists(config_path):
        print(f"✗ Config not found: {config_path}")
        return
    
    config = get_config(config_path)
    
    # 初始化环境
    try:
        env = Env(config.TASK_CONFIG)
        print(f"✓ Environment initialized ({len(env.episodes)} episodes)")
    except Exception as e:
        print(f"✗ Initialization failed: {e}")
        return
    
    # 初始化控制器
    controller = LLMAssistedController(output_dir, llm_config_path)
    
    # 动作参数
    forward_step = config.TASK_CONFIG.SIMULATOR.FORWARD_STEP_SIZE
    turn_angle = config.TASK_CONFIG.SIMULATOR.TURN_ANGLE
    
    action_dict = {
        "0": (f"STOP", 0),
        "1": (f"MOVE_FORWARD ({forward_step}m)", 1),
        "2": (f"TURN_LEFT ({turn_angle}°)", 2),
        "3": (f"TURN_RIGHT ({turn_angle}°)", 3)
    }
    
    # 选择episode
    episode_index = 0
    env._current_episode = env.episodes[episode_index]
    observations = env.reset()
    
    episode_id = env.current_episode.episode_id
    instruction = observations["instruction"]["text"]
    
    # 重置控制器
    controller.reset(episode_id, instruction)
    
    print(f"\n{'='*60}")
    print(f"Episode: {episode_id}")
    print(f"Instruction: {instruction}")
    print(f"Initial Distance: {env.get_metrics()['distance_to_goal']:.2f}m")
    print(f"{'='*60}")
    
    # 生成初始子任务
    subtask = controller.generate_initial_subtask(observations)
    if not subtask:
        print("✗ Failed to generate initial subtask")
        return
    
    controller.display_current_subtask()
    input("\n[Press Enter to start...]")
    
    # 主循环
    while not env.episode_over:
        info = env.get_metrics()
        
        print(f"\n{'-'*60}")
        print(f"Step {controller.step_count} | Distance: {info['distance_to_goal']:.2f}m")
        print(f"{'-'*60}")
        
        controller.display_current_subtask()
        
        # 动作选项
        print("Actions: 0=STOP 1=FORWARD 2=LEFT 3=RIGHT | c=Verify | q=Quit")
        user_input = input("Enter: ").strip().lower()
        
        if user_input == "exit":
            return
        
        if user_input == "q":
            observations = env.step({"action": 0})
            controller.step_count += 1
            break
        
        # 验证+再规划
        if user_input == "c":
            is_completed, next_subtask = controller.verify_and_replan(observations)
            
            if is_completed and next_subtask:
                controller.display_current_subtask()
            
            input("\n[Press Enter to continue...]")
            continue
        
        # 执行动作
        if user_input not in action_dict:
            print("⚠️  Invalid input")
            continue
        
        action_name, action_id = action_dict[user_input]
        
        # 保存执行动作前的第一人称视角
        current_phase = "initial" if controller.subtask_count == 1 else f"verify_replan_{controller.subtask_count}"
        controller.save_first_person_view(observations, current_phase)
        
        # 记录动作
        controller.record_action(action_name, action_id, info)
        
        # 执行
        observations = env.step({"action": action_id})
        controller.step_count += 1
        
        if action_id == 0:
            break
    
    # 结果
    final_metrics = env.get_metrics()
    print(f"\n{'='*60}")
    print(f"Episode Complete")
    print(f"Steps: {controller.step_count} | Subtasks: {controller.subtask_count}")
    print(f"Success: {final_metrics['success']} | SPL: {final_metrics['spl']:.4f}")
    print(f"Output: {controller.episode_dir}")
    print(f"{'='*60}")


if __name__ == "__main__":
    # 默认值
    default_habitat_config = "VLN_CE/habitat_extensions/config/vlnce_task.yaml"
    default_output_dir = "/root/autodl-tmp/manual-habitat"
    default_llm_config = "Sub_vlm/llm_config.yaml"
    
    # 解析参数
    habitat_config = sys.argv[1] if len(sys.argv) > 1 else default_habitat_config
    output_dir = sys.argv[2] if len(sys.argv) > 2 else default_output_dir
    llm_config = sys.argv[3] if len(sys.argv) > 3 else default_llm_config
    
    run_llm_assisted_control(habitat_config, output_dir, llm_config)
