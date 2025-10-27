"""
Habitat环境人工控制程序
纯手动控制，每步保存观测和地图，方便人工决策
"""
import os
import json
import cv2
import numpy as np
from habitat import Env
from habitat.utils.visualizations import maps
from VLN_CE.vlnce_baselines.config.default import get_config


class ManualController:
    """人工控制器"""
    
    def __init__(self, output_dir="./manual_control_output"):
        """
        初始化
        
        Args:
            output_dir: 输出目录
        """
        self.output_dir = output_dir
        self.step_count = 0
        self.episode_id = None
        
        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)
    
    def reset(self, episode_id):
        """
        重置（开始新episode）
        
        Args:
            episode_id: episode ID
        """
        self.episode_id = episode_id
        self.step_count = 0
        
        # 为当前episode创建文件夹
        self.episode_dir = os.path.join(self.output_dir, f"episode_{episode_id}")
        os.makedirs(self.episode_dir, exist_ok=True)
        
        # 创建子文件夹
        self.rgb_dir = os.path.join(self.episode_dir, "rgb")
        self.map_dir = os.path.join(self.episode_dir, "map")
        self.combined_dir = os.path.join(self.episode_dir, "combined")
        
        os.makedirs(self.rgb_dir, exist_ok=True)
        os.makedirs(self.map_dir, exist_ok=True)
        os.makedirs(self.combined_dir, exist_ok=True)
        
        print(f"\n{'='*80}")
        print(f"Episode {episode_id} 已开始")
        print(f"输出目录: {self.episode_dir}")
        print(f"{'='*80}\n")
    
    def save_observation(self, observations, info):
        """
        保存当前观测和地图
        
        Args:
            observations: 环境观测
            info: 环境信息
        """
        rgb = observations["rgb"]
        instruction = observations["instruction"]["text"]
        distance = info.get("distance_to_goal", -1)
        
        # 1. 保存RGB图像
        rgb_path = os.path.join(self.rgb_dir, f"step_{self.step_count:04d}.jpg")
        cv2.imwrite(rgb_path, cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
        
        # 2. 生成并保存地图
        top_down_map = maps.colorize_draw_agent_and_fit_to_height(
            info["top_down_map_vlnce"], 
            rgb.shape[0]
        )
        map_path = os.path.join(self.map_dir, f"step_{self.step_count:04d}.jpg")
        cv2.imwrite(map_path, cv2.cvtColor(top_down_map, cv2.COLOR_RGB2BGR))
        
        # 3. 生成并保存组合图（左：RGB，右：地图，底部：文本信息）
        combined = self._create_combined_view(rgb, top_down_map, instruction, distance)
        combined_path = os.path.join(self.combined_dir, f"step_{self.step_count:04d}.jpg")
        cv2.imwrite(combined_path, cv2.cvtColor(combined, cv2.COLOR_RGB2BGR))
        
        print(f"✓ 已保存观测 (Step {self.step_count})")
        print(f"  - RGB: {rgb_path}")
        print(f"  - 地图: {map_path}")
        print(f"  - 组合: {combined_path}")
    
    def _create_combined_view(self, rgb, top_down_map, instruction, distance):
        """
        创建组合视图（RGB + 地图 + 文本）
        
        Args:
            rgb: RGB图像
            top_down_map: 俯视图
            instruction: 指令文本
            distance: 到目标距离
            
        Returns:
            组合图像
        """
        # 左右拼接RGB和地图
        combined = np.concatenate((rgb, top_down_map), axis=1)
        
        # 添加底部文本区域
        h, w = combined.shape[:2]
        text_height = 120
        final_img = np.zeros((h + text_height, w, 3), dtype=np.uint8)
        final_img.fill(255)  # 白色背景
        final_img[:h, :w] = combined
        
        # 添加文本
        font = cv2.FONT_HERSHEY_SIMPLEX
        y_pos = h + 20
        
        # Step信息
        cv2.putText(final_img, f"Step: {self.step_count}", (10, y_pos), 
                   font, 0.6, (0, 0, 0), 1)
        
        # 距离信息
        y_pos += 25
        cv2.putText(final_img, f"Distance to Goal: {distance:.2f}m", (10, y_pos), 
                   font, 0.6, (0, 0, 255), 2)
        
        # 指令（可能换行）
        y_pos += 25
        max_width = w - 20
        words = instruction.split()
        line = ""
        for word in words:
            test_line = f"{line} {word}" if line else word
            (text_width, _), _ = cv2.getTextSize(test_line, font, 0.5, 1)
            if text_width > max_width:
                cv2.putText(final_img, line, (10, y_pos), font, 0.5, (0, 0, 0), 1)
                line = word
                y_pos += 20
            else:
                line = test_line
        if line:
            cv2.putText(final_img, line, (10, y_pos), font, 0.5, (0, 0, 0), 1)
        
        return final_img
    
    def save_step_info(self, action_name, action_id, info):
        """
        保存当前步骤的详细信息（JSON）
        
        Args:
            action_name: 动作名称
            action_id: 动作ID
            info: 环境信息
        """
        step_info = {
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
        
        info_path = os.path.join(self.episode_dir, f"step_{self.step_count:04d}_info.json")
        with open(info_path, "w", encoding="utf-8") as f:
            json.dump(step_info, f, indent=4, ensure_ascii=False)


def select_episode(env, current_index=0):
    """
    让用户选择要运行的episode
    
    Args:
        env: Habitat环境
        current_index: 当前episode索引
        
    Returns:
        (selected_episode, next_index) 或 (None, None) 如果退出
    """
    print(f"\n{'-'*80}")
    print("📋 Episode选择:")
    print(f"  1. 按顺序运行下一个 (当前索引: {current_index})")
    print(f"  2. 指定Episode索引 (0-{len(env.episodes) - 1})")
    print(f"  3. 指定Episode ID")
    print(f"  4. 随机选择")
    print(f"  5. 列出所有Episodes (查看ID和信息)")
    print(f"  q. 退出程序")
    
    choice = input("\n请选择 (1-5/q): ").strip().lower()
    
    if choice == "q":
        return None, None
    
    # 1. 按顺序
    if choice == "1":
        if current_index >= len(env.episodes):
            print(f"\n⚠️  已运行完所有{len(env.episodes)}个episodes！")
            continue_input = input("是否从头开始? (y/n): ").strip().lower()
            if continue_input == "y":
                return env.episodes[0], 1
            return None, None
        return env.episodes[current_index], current_index + 1
    
    # 2. 指定索引
    elif choice == "2":
        try:
            idx = int(input(f"请输入Episode索引 (0-{len(env.episodes) - 1}): ").strip())
            if 0 <= idx < len(env.episodes):
                return env.episodes[idx], idx + 1
            else:
                print(f"⚠️  索引超出范围 (0-{len(env.episodes) - 1})")
                return select_episode(env, current_index)
        except ValueError:
            print("⚠️  请输入有效数字")
            return select_episode(env, current_index)
    
    # 3. 指定ID
    elif choice == "3":
        ep_id = input("请输入Episode ID: ").strip()
        for idx, ep in enumerate(env.episodes):
            if str(ep.episode_id) == ep_id:
                print(f"✓ 找到Episode (索引: {idx})")
                return ep, idx + 1
        print(f"⚠️  未找到Episode ID: {ep_id}")
        return select_episode(env, current_index)
    
    # 4. 随机
    elif choice == "4":
        import random
        idx = random.randint(0, len(env.episodes) - 1)
        print(f"🎲 随机选择了Episode索引: {idx}, ID: {env.episodes[idx].episode_id}")
        return env.episodes[idx], idx + 1
    
    # 5. 列出所有
    elif choice == "5":
        print(f"\n可用的Episodes (共{len(env.episodes)}个):")
        print("-"*80)
        
        # 显示前20个
        display_count = min(20, len(env.episodes))
        for i in range(display_count):
            ep = env.episodes[i]
            scene_name = ep.scene_id.split('/')[-1] if '/' in ep.scene_id else ep.scene_id
            print(f"  [{i:3d}] ID: {ep.episode_id:15s} | 场景: {scene_name}")
        
        if len(env.episodes) > display_count:
            print(f"  ... (还有 {len(env.episodes) - display_count} 个episodes)")
        
        print("-"*80)
        return select_episode(env, current_index)
    
    else:
        print("⚠️  无效选择")
        return select_episode(env, current_index)


def run_manual_control(config_path: str, output_dir: str = "./manual_control_output"):
    """
    运行人工控制程序
    
    Args:
        config_path: Habitat配置文件路径
        output_dir: 输出目录
    """
    print("="*80)
    print("Habitat环境人工控制程序")
    print("="*80)
    
    # 加载配置
    print(f"\n1. 加载配置: {config_path}")
    if not os.path.exists(config_path):
        print(f"错误：配置文件不存在: {config_path}")
        return
    
    config = get_config(config_path)
    
    # 初始化环境
    print("2. 初始化Habitat环境...")
    try:
        env = Env(config.TASK_CONFIG)
        print(f"   ✓ 环境初始化成功")
        print(f"   - 可用Episodes: {len(env.episodes)}")
    except Exception as e:
        print(f"   ✗ 环境初始化失败: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # 初始化控制器
    controller = ManualController(output_dir)
    
    # 动作字典
    action_dict = {
        "0": ("STOP (停止)", 0),
        "1": ("MOVE_FORWARD (前进 0.25m)", 1),
        "2": ("TURN_LEFT (左转 30度)", 2),
        "3": ("TURN_RIGHT (右转 30度)", 3)
    }
    
    # 所有episode的结果
    all_results = []
    
    print(f"\n3. Episode管理")
    print(f"   总数: {len(env.episodes)}")
    
    # 主循环
    episode_index = 0  # 当前episode索引
    
    while True:
        # 让用户选择episode
        selected_episode, episode_index = select_episode(env, episode_index)
        
        if selected_episode is None:
            print("\n退出程序...")
            break
        
        # 设置环境到选定的episode
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
        print(f"\n初始距离: {env.get_metrics()['distance_to_goal']:.2f}m")
        
        # Episode循环
        while not env.episode_over:
            # 获取当前信息
            info = env.get_metrics()
            
            # 保存当前观测
            controller.save_observation(observations, info)
            
            # 显示当前状态
            print(f"\n{'-'*80}")
            print(f"Step {controller.step_count}")
            print(f"当前距离目标: {info['distance_to_goal']:.2f}m")
            print(f"已行走路径: {info['path_length']:.2f}m")
            print(f"{'-'*80}")
            
            # 显示动作选项
            print("\n可用动作:")
            for key, (name, _) in action_dict.items():
                print(f"  {key}: {name}")
            print("  q: 结束当前episode并查看结果")
            print("  exit: 退出程序")
            
            # 获取用户输入
            user_input = input("\n请输入动作编号: ").strip().lower()
            
            # 处理退出命令
            if user_input == "exit":
                print("\n退出程序...")
                return
            
            if user_input == "q":
                print("\n强制结束当前episode...")
                # 执行STOP动作
                observations = env.step({"action": 0})
                controller.save_step_info("STOP (强制)", 0, info)
                controller.step_count += 1
                break
            
            # 验证输入
            if user_input not in action_dict:
                print("⚠️  无效输入，请输入 0-3, q 或 exit")
                continue
            
            # 获取动作
            action_name, action_id = action_dict[user_input]
            
            # 保存步骤信息
            controller.save_step_info(action_name, action_id, info)
            
            # 执行动作
            print(f"\n执行动作: {action_name}")
            observations = env.step({"action": action_id})
            controller.step_count += 1
            
            # 检查是否结束
            if action_id == 0:
                print("\n✓ 已执行STOP，episode结束")
                break
        
        # Episode结束，收集结果
        final_metrics = env.get_metrics()
        
        # 最后一步的观测
        controller.save_observation(observations, final_metrics)
        
        # 保存episode结果
        result = {
            "episode_id": episode_id,
            "scene_id": env.current_episode.scene_id,
            "instruction": instruction,
            "total_steps": controller.step_count,
            "final_metrics": {
                "distance_to_goal": final_metrics.get("distance_to_goal", -1),
                "success": final_metrics.get("success", 0),
                "spl": final_metrics.get("spl", 0),
                "path_length": final_metrics.get("path_length", 0),
                "oracle_success": final_metrics.get("oracle_success", 0)
            }
        }
        all_results.append(result)
        
        # 显示结果
        print(f"\n{'='*80}")
        print(f"Episode {episode_id} 结果")
        print(f"{'='*80}")
        print(f"总步数: {result['total_steps']}")
        print(f"最终距离: {result['final_metrics']['distance_to_goal']:.2f}m")
        print(f"成功: {'是' if result['final_metrics']['success'] else '否'} (< 3m)")
        print(f"SPL: {result['final_metrics']['spl']:.4f}")
        print(f"路径长度: {result['final_metrics']['path_length']:.2f}m")
        print(f"Oracle成功: {'是' if result['final_metrics']['oracle_success'] else '否'}")
        
        # 保存episode汇总
        result_path = os.path.join(controller.episode_dir, "episode_result.json")
        with open(result_path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=4, ensure_ascii=False)
        print(f"\n结果已保存: {result_path}")
        
        # 生成视频
        print("\n正在生成历史帧视频...")
        video_path = generate_video(controller.combined_dir, controller.episode_dir, episode_id)
        print(f"✓ 视频已生成: {video_path}")
        
        # Episode完成，回到选择菜单
        input("\n按回车键继续...")
    
    # 保存总体汇总
    if all_results:
        print(f"\n{'='*80}")
        print("总体评估汇总")
        print(f"{'='*80}")
        
        summary = {
            "total_episodes": len(all_results),
            "episodes": all_results,
            "average_metrics": {
                "avg_distance_to_goal": np.mean([r["final_metrics"]["distance_to_goal"] for r in all_results]),
                "avg_success_rate": np.mean([r["final_metrics"]["success"] for r in all_results]),
                "avg_spl": np.mean([r["final_metrics"]["spl"] for r in all_results]),
                "avg_path_length": np.mean([r["final_metrics"]["path_length"] for r in all_results])
            }
        }
        
        summary_path = os.path.join(output_dir, "overall_summary.json")
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=4, ensure_ascii=False)
        
        print(f"\n完成Episodes数: {summary['total_episodes']}")
        print(f"平均最终距离: {summary['average_metrics']['avg_distance_to_goal']:.2f}m")
        print(f"平均成功率: {summary['average_metrics']['avg_success_rate']:.2%}")
        print(f"平均SPL: {summary['average_metrics']['avg_spl']:.4f}")
        print(f"平均路径长度: {summary['average_metrics']['avg_path_length']:.2f}m")
        print(f"\n总体汇总已保存: {summary_path}")
    
    print(f"\n✓ 所有结果保存在: {output_dir}")


def generate_video(frame_dir, output_dir, episode_id, fps=2):
    """
    从帧序列生成视频
    
    Args:
        frame_dir: 帧图像目录
        output_dir: 输出目录
        episode_id: episode ID
        fps: 帧率
        
    Returns:
        视频路径
    """
    # 获取所有帧
    frame_files = sorted([f for f in os.listdir(frame_dir) if f.endswith('.jpg')])
    
    if not frame_files:
        print("警告：没有找到帧图像")
        return None
    
    # 读取第一帧获取尺寸
    first_frame = cv2.imread(os.path.join(frame_dir, frame_files[0]))
    h, w = first_frame.shape[:2]
    
    # 创建视频写入器
    video_path = os.path.join(output_dir, f"episode_{episode_id}_history.mp4")
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(video_path, fourcc, fps, (w, h))
    
    # 写入所有帧
    for frame_file in frame_files:
        frame = cv2.imread(os.path.join(frame_dir, frame_file))
        video_writer.write(frame)
    
    video_writer.release()
    return video_path


if __name__ == "__main__":
    import sys
    
    print("\nHabitat环境人工控制程序")
    print("每步保存观测和地图，人工决策\n")
    
    if len(sys.argv) < 2:
        print("使用方法:")
        print("  python manual_control.py <config_path> [output_dir]")
        print("\n示例:")
        print("  python manual_control.py VLN_CE/habitat_extensions/config/vlnce_task.yaml")
        print("  python manual_control.py VLN_CE/habitat_extensions/config/vlnce_task.yaml ./my_output")
        print("")
    else:
        config_path = sys.argv[1]
        output_dir = sys.argv[2] if len(sys.argv) > 2 else "./manual_control_output"
        
        run_manual_control(config_path, output_dir)
