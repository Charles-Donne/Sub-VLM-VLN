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

# 添加父目录到路径以导入VLN_CE模块
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from VLN_CE.vlnce_baselines.config.default import get_config

from thinking import LLMPlanner, SubTask
from observation_collector import ObservationCollector


class LLMAssistedController:
    """LLM辅助控制器"""
    
    def __init__(self, output_dir: str, llm_config_path: str = "llm_config.yaml"):
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
        
        # 初始化LLM规划器
        self.planner = LLMPlanner(llm_config_path)
        
        # 当前子任务
        self.current_subtask = None
        
        # 子任务历史
        self.subtask_history = []
    
    def reset(self, episode_id: str):
        """
        重置（开始新episode）
        
        Args:
            episode_id: episode ID
        """
        self.episode_id = episode_id
        self.step_count = 0
        self.subtask_count = 0
        self.current_subtask = None
        self.subtask_history = []
        
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
        print(f"Episode {episode_id} 已开始")
        print(f"输出目录: {self.episode_dir}")
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
        print(f"🔍 观察环境 - {phase}")
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
        
        # 打印摘要
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
        print("🤖 LLM规划 - 生成初始子任务")
        print(f"{'*'*80}")
        print(f"完整指令: {instruction}\n")
        
        # 收集观察
        image_paths, direction_names = self.observe_environment(observations, "initial")
        
        # 调用LLM生成子任务
        subtask = self.planner.generate_initial_subtask(
            instruction,
            image_paths,
            direction_names
        )
        
        if subtask is None:
            print("✗ 子任务生成失败，使用默认子任务")
            subtask = SubTask(
                description="向前探索并寻找指令中提到的第一个标志物",
                planning_hints="保持直行，注意观察周围环境",
                completion_criteria="看到指令中提到的第一个物体或位置"
            )
        
        # 保存子任务
        self.current_subtask = subtask
        self.subtask_count += 1
        self._save_subtask(subtask, "initial")
        
        return subtask
    
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
        print("🤖 LLM验证 - 检查子任务完成情况")
        print(f"{'*'*80}")
        print(f"当前子任务: {self.current_subtask.description}\n")
        
        # 收集观察
        phase = f"verification_subtask{self.subtask_count}"
        image_paths, direction_names = self.observe_environment(observations, phase)
        
        # 调用LLM验证
        is_completed, next_subtask, advice = self.planner.verify_and_plan_next(
            instruction,
            self.current_subtask,
            image_paths,
            direction_names
        )
        
        if is_completed and next_subtask:
            # 保存旧子任务到历史
            self.subtask_history.append({
                "subtask_id": self.subtask_count,
                "subtask": self.current_subtask.to_dict(),
                "completed": True,
                "completion_step": self.step_count
            })
            
            # 更新当前子任务
            self.current_subtask = next_subtask
            self.subtask_count += 1
            self._save_subtask(next_subtask, f"subtask{self.subtask_count}")
            
        return is_completed, next_subtask
    
    def _save_subtask(self, subtask: SubTask, phase: str):
        """
        保存子任务信息
        
        Args:
            subtask: 子任务对象
            phase: 阶段标识
        """
        subtask_data = {
            "subtask_id": self.subtask_count,
            "phase": phase,
            "step": self.step_count,
            **subtask.to_dict()
        }
        
        filepath = os.path.join(self.subtasks_dir, f"subtask_{self.subtask_count}_{phase}.json")
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(subtask_data, f, indent=4, ensure_ascii=False)
        
        print(f"✓ 子任务已保存: {filepath}")
    
    def display_current_subtask(self):
        """显示当前子任务信息"""
        if self.current_subtask is None:
            print("\n⚠️  当前没有活动的子任务")
            return
        
        print(f"\n{'='*80}")
        print(f"📋 当前子任务 #{self.subtask_count}")
        print(f"{'='*80}")
        print(f"描述: {self.current_subtask.description}")
        print(f"\n规划提示:")
        print(f"  {self.current_subtask.planning_hints}")
        print(f"\n完成标准:")
        print(f"  {self.current_subtask.completion_criteria}")
        print(f"{'='*80}\n")
    
    def save_step_action(self, action_name: str, action_id: int, info: Dict):
        """
        保存步骤动作信息
        
        Args:
            action_name: 动作名称
            action_id: 动作ID
            info: 环境信息
        """
        step_data = {
            "step": self.step_count,
            "subtask_id": self.subtask_count,
            "action": {
                "name": action_name,
                "id": action_id
            },
            "metrics": {
                "distance_to_goal": info.get("distance_to_goal", -1),
                "path_length": info.get("path_length", 0)
            }
        }
        
        filepath = os.path.join(self.episode_dir, f"step_{self.step_count:04d}_action.json")
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(step_data, f, indent=4, ensure_ascii=False)
    
    def save_episode_summary(self, instruction: str, final_metrics: Dict):
        """
        保存episode汇总
        
        Args:
            instruction: 导航指令
            final_metrics: 最终指标
        """
        summary = {
            "episode_id": self.episode_id,
            "instruction": instruction,
            "total_steps": self.step_count,
            "total_subtasks": self.subtask_count,
            "subtask_history": self.subtask_history,
            "final_metrics": {
                "distance_to_goal": final_metrics.get("distance_to_goal", -1),
                "success": final_metrics.get("success", 0),
                "spl": final_metrics.get("spl", 0),
                "path_length": final_metrics.get("path_length", 0),
                "oracle_success": final_metrics.get("oracle_success", 0)
            }
        }
        
        filepath = os.path.join(self.episode_dir, "episode_summary.json")
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=4, ensure_ascii=False)
        
        print(f"✓ Episode汇总已保存: {filepath}")


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
    print("LLM辅助的Habitat导航控制程序")
    print("="*80)
    
    # 加载配置
    print(f"\n1. 加载配置")
    print(f"   - Habitat配置: {config_path}")
    print(f"   - LLM配置: {llm_config_path}")
    
    if not os.path.exists(config_path):
        print(f"✗ 错误：Habitat配置文件不存在: {config_path}")
        return
    
    if not os.path.exists(llm_config_path):
        print(f"✗ 错误：LLM配置文件不存在: {llm_config_path}")
        print(f"   请从 llm_config.yaml.template 创建配置文件")
        return
    
    config = get_config(config_path)
    
    # 初始化环境
    print("\n2. 初始化Habitat环境...")
    try:
        env = Env(config.TASK_CONFIG)
        print(f"   ✓ 环境初始化成功")
        print(f"   - 可用Episodes: {len(env.episodes)}")
        
        if len(env.episodes) == 0:
            print("   ⚠️  警告: 没有找到任何episodes")
            return
            
    except Exception as e:
        print(f"   ✗ 环境初始化失败: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # 初始化控制器
    print("\n3. 初始化LLM辅助控制器...")
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
    
    print(f"\n   动作参数:")
    print(f"   - 前进步长: {forward_step_size}m")
    print(f"   - 转向角度: {turn_angle}°")
    
    # 选择episode（简化版，使用第一个）
    print(f"\n4. 选择Episode")
    episode_index = 0
    selected_episode = env.episodes[episode_index]
    
    env._current_episode = selected_episode
    observations = env.reset()
    
    episode_id = env.current_episode.episode_id
    instruction = observations["instruction"]["text"]
    
    # 重置控制器
    controller.reset(episode_id)
    
    # 显示episode信息
    print(f"\nEpisode ID: {episode_id}")
    print(f"场景: {env.current_episode.scene_id}")
    print(f"指令: {instruction}")
    print(f"初始距离: {env.get_metrics()['distance_to_goal']:.2f}m")
    
    # ========== 阶段1: 生成初始子任务 ==========
    subtask = controller.generate_initial_subtask(instruction, observations)
    controller.display_current_subtask()
    
    input("\n按回车键开始执行子任务...")
    
    # ========== 主循环: 子任务执行 ==========
    while not env.episode_over:
        
        # 显示当前状态
        info = env.get_metrics()
        print(f"\n{'-'*80}")
        print(f"Step {controller.step_count}")
        print(f"当前距离目标: {info['distance_to_goal']:.2f}m")
        print(f"已行走路径: {info['path_length']:.2f}m")
        print(f"{'-'*80}")
        
        # 显示当前子任务
        controller.display_current_subtask()
        
        # 显示动作选项
        print("\n可用操作:")
        for key, (name, _) in action_dict.items():
            print(f"  {key}: {name}")
        print("  c: 完成当前子任务，请求LLM验证")
        print("  q: 结束episode")
        print("  exit: 退出程序")
        
        # 获取用户输入
        user_input = input("\n请输入操作: ").strip().lower()
        
        # 处理特殊命令
        if user_input == "exit":
            print("\n退出程序...")
            return
        
        if user_input == "q":
            print("\n结束episode...")
            observations = env.step({"action": 0})
            controller.step_count += 1
            break
        
        # 完成子任务 - 请求LLM验证
        if user_input == "c":
            print("\n正在请求LLM验证子任务完成情况...")
            
            is_completed, next_subtask = controller.verify_subtask_completion(
                instruction, observations
            )
            
            if is_completed:
                print("\n✓ 子任务已完成！")
                if next_subtask:
                    print("\n已生成下一个子任务")
                    controller.display_current_subtask()
                else:
                    print("\n可能已接近目标，请继续手动导航或检查任务完成情况")
            else:
                print("\n✗ 子任务尚未完成，请继续执行")
            
            input("\n按回车键继续...")
            continue
        
        # 验证动作输入
        if user_input not in action_dict:
            print("⚠️  无效输入，请重新输入")
            continue
        
        # 获取动作
        action_name, action_id = action_dict[user_input]
        
        # 保存动作
        controller.save_step_action(action_name, action_id, info)
        
        # 执行动作
        print(f"\n执行动作: {action_name}")
        observations = env.step({"action": action_id})
        controller.step_count += 1
        
        # 检查STOP
        if action_id == 0:
            print("\n✓ 已执行STOP，episode结束")
            break
    
    # Episode结束
    final_metrics = env.get_metrics()
    
    # 保存汇总
    controller.save_episode_summary(instruction, final_metrics)
    
    # 显示结果
    print(f"\n{'='*80}")
    print(f"Episode {episode_id} 结果")
    print(f"{'='*80}")
    print(f"总步数: {controller.step_count}")
    print(f"子任务数: {controller.subtask_count}")
    print(f"最终距离: {final_metrics['distance_to_goal']:.2f}m")
    print(f"成功: {'是' if final_metrics['success'] else '否'}")
    print(f"SPL: {final_metrics['spl']:.4f}")
    print(f"路径长度: {final_metrics['path_length']:.2f}m")
    print(f"\n✓ 所有结果保存在: {controller.episode_dir}")


if __name__ == "__main__":
    print("\nLLM辅助的Habitat导航控制程序")
    print("结合大模型规划和人工执行\n")
    
    if len(sys.argv) < 2:
        print("使用方法:")
        print("  python llm_manual_control.py <habitat_config> [output_dir] [llm_config]")
        print("\n示例:")
        print("  python llm_manual_control.py ../VLN_CE/habitat_extensions/config/vlnce_task_enhanced.yaml")
        print("  python llm_manual_control.py ../VLN_CE/habitat_extensions/config/vlnce_task_enhanced.yaml ./output llm_config.yaml")
        print("")
    else:
        habitat_config = sys.argv[1]
        output_dir = sys.argv[2] if len(sys.argv) > 2 else "./llm_control_output"
        llm_config = sys.argv[3] if len(sys.argv) > 3 else "llm_config.yaml"
        
        run_llm_assisted_control(habitat_config, output_dir, llm_config)
