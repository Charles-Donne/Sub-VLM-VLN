"""
LLM辅助的人工控制程序
结合大模型规划和人工执行的导航系统
"""
import os
import sys
import json
import cv2
from typing import Dict, List, Tuple, Optional
from habitat import Env, make_dataset
from habitat.utils.visualizations import maps

from VLN_CE.vlnce_baselines.config.default import get_config

from Sub_vlm.thinking import LLMPlanner
from Sub_vlm.observation_collector import ObservationCollector


class LLMAssistedController:
    """LLM辅助控制器"""
    
    def __init__(self, output_dir: str, llm_config_path: str = "Sub_vlm/llm_config.yaml", action_space: str = None):
        """
        初始化控制器
        
        Args:
            output_dir: 输出目录
            llm_config_path: LLM配置文件路径
            action_space: 动作空间描述
        """
        self.output_dir = output_dir
        self.step_count = 0
        self.subtask_count = 0
        self.episode_id = None
        self.instruction = None
        
        os.makedirs(output_dir, exist_ok=True)
        self.planner = LLMPlanner(llm_config_path, action_space)
        
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
        
        # 创建地图可视化收集器
        self.map_collector = ObservationCollector(self.observations_dir)
        self.map_collector.setup_maps_dir(self.episode_dir)
    
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
    
    def save_step_with_map(self, observations: Dict, info: Dict):
        """保存包含地图的可视化"""
        if not self.map_collector:
            return
        
        current_subtask_text = None
        if self.current_subtask:
            current_subtask_text = self.current_subtask['subtask_instruction']
        
        self.map_collector.save_step_visualization(
            observations=observations,
            info=info,
            step=self.step_count,
            instruction=self.instruction,
            current_subtask=current_subtask_text,
            distance=info.get("distance_to_goal", 0.0)
        )
    
    def generate_initial_subtask(self, observations: Dict) -> Optional[Dict]:
        """生成初始子任务"""
        print(f"\n{'*'*80}")
        print("🤖 Generating Initial Subtask")
        print(f"{'*'*80}")
        
        # 收集观察
        image_paths, direction_names = self.observe_environment(observations, "initial")
        
        # 调用LLM
        response = self.planner.generate_initial_subtask(
            self.instruction,
            image_paths,
            direction_names
        )
        
        if not response:
            print("✗ LLM call failed")
            return None
        
        # 保存子任务
        self.current_subtask = response
        self.subtask_count += 1
        
        # 创建子任务文件
        subtask_name = "initial_subtask"
        self._create_subtask_file(subtask_name, response)
        
        # 打印生成结果
        print(f"\n✅ ===== Initial Subtask Generated =====")
        print(f"🌍 Global Instruction: {self.instruction}")
        print(f"🎯 Subtask Destination: {response['subtask_destination']}")
        print(f"📋 Subtask Instruction: {response['subtask_instruction']}")
        print(f"💡 Planning Hints: {response['planning_hints']}")
        print(f"✓ Completion Criteria: {response['completion_criteria']}")
        print(f"✅ ======================================\n")
        
        return response
    
    def _create_subtask_file(self, subtask_name: str, response: Dict):
        """创建子任务文件"""
        subtask_data = {
            "global_instruction": self.instruction,
            "subtask_id": self.subtask_count,
            "subtask_name": subtask_name,
            "generated_at_step": self.step_count,
            "llm_response": response,
            "actions": []
        }
        
        filepath = os.path.join(self.subtasks_dir, f"{subtask_name}.json")
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(subtask_data, f, indent=2, ensure_ascii=False)
        
        self.current_subtask_file = filepath
        print(f"💾 Subtask saved: {filepath}")
    
    def verify_and_replan(self, observations: Dict) -> Tuple[bool, Optional[Dict]]:
        """验证+再规划模块"""
        print(f"\n{'*'*80}")
        print("🤖 Verification + Replanning")
        print(f"{'*'*80}")
        
        # 先增加子任务计数
        self.subtask_count += 1
        
        # 收集观察
        phase = f"verify_replan_{self.subtask_count}"
        image_paths, direction_names = self.observe_environment(observations, phase)
        
        # 调用LLM
        response, is_completed = self.planner.verify_and_replan(
            self.instruction,
            self.current_subtask,
            image_paths,
            direction_names
        )
        
        if not response:
            print("✗ LLM call failed")
            self.subtask_count -= 1  # 失败时回退
            return False, None
        
        # 无论成功与否，都保存为新子任务
        subtask_name = f"subtask_{self.subtask_count}"
        
        if is_completed:
            # 子任务完成，保存新子任务
            self.current_subtask = response
            self._create_subtask_file(subtask_name, response)
            
            print(f"\n✅ ===== Subtask #{self.subtask_count-1} COMPLETED =====")
            print(f"🌍 Global Instruction: {self.instruction}")
            print(f"🎯 Next Subtask Destination: {response['subtask_destination']}")
            print(f"📋 Next Subtask Instruction: {response['subtask_instruction']}")
            print(f"💡 Planning Hints: {response['planning_hints']}")
            print(f"✓ Completion Criteria: {response['completion_criteria']}")
            print(f"✅ =============================================\n")
            
        else:
            # 子任务未完成，保存refined子任务
            self.current_subtask = response
            self._create_subtask_file(subtask_name, response)
            
            print(f"\n🔄 ===== Subtask #{self.subtask_count-1} NOT COMPLETED =====")
            print(f"🌍 Global Instruction: {self.instruction}")
            print(f"🎯 Target Destination: {response['subtask_destination']}")
            print(f"📋 Refined Instruction: {response['subtask_instruction']}")
            print(f"💡 Planning Hints: {response['planning_hints']}")
            print(f"✓ Completion Criteria: {response['completion_criteria']}")
            print(f"🔄 =============================================\n")
        
        return is_completed, response
    
    def _update_subtask_file(self, response: Dict):
        """更新子任务文件（指令refinement）"""
        if not self.current_subtask_file or not os.path.exists(self.current_subtask_file):
            return
        
        with open(self.current_subtask_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        data["llm_response"] = response
        
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
        print(f"📋 Subtask #{self.subtask_count}")
        print(f"📍 Current: {self.current_subtask['subtask_destination']}")
        print(f"📋 Instruction: {self.current_subtask['subtask_instruction']}")
        print(f"💡 Hints: {self.current_subtask['planning_hints']}")
        print(f"✓ Criteria: {self.current_subtask['completion_criteria']}")
        print(f"{'='*60}\n")


def run_llm_assisted_control(config_path: str, 
                             output_dir: str = "./llm_control_output",
                             llm_config_path: str = "llm_config.yaml",
                             episode_index: int = 0):
    """运行LLM辅助导航控制"""
    print("="*60)
    print("LLM-Assisted Navigation Control")
    print("="*60)
    
    # 加载配置
    if not os.path.exists(config_path):
        print(f"✗ Config not found: {config_path}")
        return
    
    config = get_config(config_path)
    
    # 启用地图测量
    if "TOP_DOWN_MAP_VLNCE" not in config.TASK_CONFIG.TASK.MEASUREMENTS:
        config.TASK_CONFIG.TASK.MEASUREMENTS.append("TOP_DOWN_MAP_VLNCE")
    
    # 加载数据集
    print("Loading dataset...")
    dataset = make_dataset(
        id_dataset=config.TASK_CONFIG.DATASET.TYPE,
        config=config.TASK_CONFIG.DATASET
    )
    print(f"✓ Dataset loaded ({len(dataset.episodes)} episodes)")
    
    # 选择episode
    if episode_index < 0 or episode_index >= len(dataset.episodes):
        print(f"✗ Invalid episode index: {episode_index} (available: 0-{len(dataset.episodes)-1})")
        return
    
    # 只保留选定的episode
    selected_episode = dataset.episodes[episode_index]
    dataset.episodes = [selected_episode]
    
    print(f"Selected Episode {episode_index}: ID={selected_episode.episode_id}")
    
    # 初始化环境（使用筛选后的dataset）
    try:
        env = Env(config.TASK_CONFIG, dataset)
        print(f"✓ Environment initialized")
    except Exception as e:
        print(f"✗ Initialization failed: {e}")
        return
    
    # 初始化控制器
    forward_step = config.TASK_CONFIG.SIMULATOR.FORWARD_STEP_SIZE
    turn_angle = config.TASK_CONFIG.SIMULATOR.TURN_ANGLE
    
    # 构建动作空间描述
    action_space = f"MOVE_FORWARD ({forward_step}m), TURN_LEFT ({turn_angle}°), TURN_RIGHT ({turn_angle}°), STOP"
    
    controller = LLMAssistedController(output_dir, llm_config_path, action_space)
    
    # 动作参数
    
    action_dict = {
        "0": (f"STOP", 0),
        "1": (f"MOVE_FORWARD ({forward_step}m)", 1),
        "2": (f"TURN_LEFT ({turn_angle}°)", 2),
        "3": (f"TURN_RIGHT ({turn_angle}°)", 3)
    }
    
    # 重置环境获取初始观察
    observations = env.reset()
    
    # 获取episode信息
    episode_id = env.current_episode.episode_id
    instruction = observations["instruction"]["text"]
    
    # 重置控制器
    controller.reset(episode_id, instruction)
    
    print(f"\n{'='*60}")
    print(f"Episode Index: {episode_index}")
    print(f"Episode ID: {episode_id}")
    print(f"Instruction: {instruction}")
    print(f"Initial Distance: {env.get_metrics()['distance_to_goal']:.2f}m")
    print(f"{'='*60}")
    
    # 生成初始子任务
    subtask = controller.generate_initial_subtask(observations)
    if not subtask:
        print("✗ Failed to generate initial subtask")
        return
    
    input("\n[Press Enter to start...]")
    
    # 主循环
    while not env.episode_over:
        info = env.get_metrics()
        
        print(f"\n{'-'*60}")
        print(f"Step {controller.step_count} | Distance: {info['distance_to_goal']:.2f}m")
        print(f"{'-'*60}")
        
        # 保存当前观察（在用户选择动作之前）
        current_phase = "initial" if controller.subtask_count == 1 else f"verify_replan_{controller.subtask_count}"
        controller.save_first_person_view(observations, current_phase)
        
        # 保存地图可视化
        controller.save_step_with_map(observations, info)
        
        # 动作选项
        print("\nAvailable Actions:")
        print(f"  0 = STOP")
        print(f"  1 = MOVE_FORWARD ({forward_step}m)")
        print(f"  2 = TURN_LEFT ({turn_angle}°)")
        print(f"  3 = TURN_RIGHT ({turn_angle}°)")
        print(f"  c = Verify & Replan")
        print(f"  q = Quit")
        user_input = input("\nEnter action: ").strip().lower()
        
        if user_input == "exit":
            return
        
        if user_input == "q":
            observations = env.step({"action": 0})
            controller.step_count += 1
            break
        
        # 验证+再规划
        if user_input == "c":
            is_completed, response = controller.verify_and_replan(observations)
            input("\n[Press Enter to continue...]")
            continue
        
        # 执行动作
        if user_input not in action_dict:
            print("⚠️  Invalid input")
            continue
        
        action_name, action_id = action_dict[user_input]
        
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
    
    # 保存结果到JSON
    result_data = {
        "episode_index": episode_index,
        "episode_id": episode_id,
        "instruction": instruction,
        "steps": controller.step_count,
        "subtasks": controller.subtask_count,
        "metrics": {
            "distance_to_goal": final_metrics.get("distance_to_goal", -1),
            "success": final_metrics.get("success", False),
            "spl": final_metrics.get("spl", 0.0),
            "path_length": final_metrics.get("path_length", 0.0),
            "oracle_success": final_metrics.get("oracle_success", False)
        }
    }
    
    result_file = os.path.join(controller.episode_dir, "result.json")
    with open(result_file, "w", encoding="utf-8") as f:
        json.dump(result_data, f, indent=2, ensure_ascii=False)
    print(f"✓ Result saved: {result_file}")
    
    # 生成GIF
    if controller.map_collector:
        print("Generating navigation GIF...")
        gif_path = controller.map_collector.save_gif()
        if gif_path:
            print(f"GIF: {gif_path}")
    
    print(f"{'='*60}")


if __name__ == "__main__":
    # 默认值
    default_habitat_config = "VLN_CE/vlnce_baselines/config/r2r_baselines/navid_r2r.yaml"
    default_output_dir = "/root/autodl-tmp/manual-habitat"
    default_llm_config = "Sub_vlm/llm_config.yaml"
    default_episode_index = 0
    
    # 解析参数 - episode_index 放在第一位
    episode_index = int(sys.argv[1]) if len(sys.argv) > 1 else default_episode_index
    habitat_config = sys.argv[2] if len(sys.argv) > 2 else default_habitat_config
    output_dir = sys.argv[3] if len(sys.argv) > 3 else default_output_dir
    llm_config = sys.argv[4] if len(sys.argv) > 4 else default_llm_config
    
    run_llm_assisted_control(habitat_config, output_dir, llm_config, episode_index)
