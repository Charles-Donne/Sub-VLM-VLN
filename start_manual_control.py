#!/usr/bin/env python3
"""
人工控制Habitat - 快速启动脚本
"""

import os
import sys

def print_banner():
    """打印欢迎信息"""
    print("""
╔══════════════════════════════════════════════════════════════╗
║                                                              ║
║          🎮 Habitat VLN 人工控制程序 🎮                     ║
║                                                              ║
║  完全手动控制 | 实时保存观测 | 自动生成视频                  ║
║                                                              ║
╚══════════════════════════════════════════════════════════════╝
    """)


def check_dependencies():
    """检查依赖"""
    print("📦 检查依赖...")
    missing = []
    
    try:
        import habitat
        print("  ✓ habitat")
    except ImportError:
        missing.append("habitat-sim habitat-lab")
    
    try:
        import cv2
        print("  ✓ opencv")
    except ImportError:
        missing.append("opencv-python")
    
    try:
        import numpy
        print("  ✓ numpy")
    except ImportError:
        missing.append("numpy")
    
    if missing:
        print(f"\n❌ 缺少依赖: {', '.join(missing)}")
        print(f"\n请安装: pip install {' '.join(missing)}")
        return False
    
    print("✓ 所有依赖已满足\n")
    return True


def find_config_files():
    """查找配置文件"""
    possible_paths = [
        "VLN_CE/habitat_extensions/config/vlnce_task.yaml",
        "VLN_CE/habitat_extensions/config/vlnce_task_navid_r2r.yaml",
        "VLN_CE/habitat_extensions/config/vlnce_task_navid_rxr.yaml",
        "habitat_extensions/config/vlnce_task.yaml"
    ]
    
    found = []
    for path in possible_paths:
        if os.path.exists(path):
            found.append(path)
    
    return found


def interactive_setup():
    """交互式配置"""
    print("🔧 配置设置\n")
    
    # 1. 查找配置文件
    print("1. 查找配置文件...")
    configs = find_config_files()
    
    if configs:
        print(f"   找到 {len(configs)} 个配置文件:")
        for i, config in enumerate(configs, 1):
            print(f"   {i}. {config}")
        
        print(f"   {len(configs)+1}. 手动输入路径")
        
        choice = input(f"\n选择配置文件 (1-{len(configs)+1}): ").strip()
        
        try:
            idx = int(choice)
            if 1 <= idx <= len(configs):
                config_path = configs[idx-1]
            else:
                config_path = input("请输入配置文件路径: ").strip()
        except ValueError:
            config_path = input("请输入配置文件路径: ").strip()
    else:
        print("   未找到配置文件")
        config_path = input("   请输入配置文件路径: ").strip()
    
    if not os.path.exists(config_path):
        print(f"\n❌ 配置文件不存在: {config_path}")
        return None, None
    
    print(f"   ✓ 使用配置: {config_path}\n")
    
    # 2. 输出目录
    print("2. 设置输出目录")
    default_output = "./manual_control_output"
    output_dir = input(f"   输出目录 (回车使用默认 '{default_output}'): ").strip()
    
    if not output_dir:
        output_dir = default_output
    
    print(f"   ✓ 输出目录: {output_dir}\n")
    
    return config_path, output_dir


def print_instructions():
    """打印使用说明"""
    print("="*70)
    print("📖 使用说明")
    print("="*70)
    print("""
动作选项:
  0 - STOP (停止，认为已到达目标)
  1 - MOVE_FORWARD (前进 0.25米)
  2 - TURN_LEFT (左转 30度)
  3 - TURN_RIGHT (右转 30度)
  q - 结束当前episode
  exit - 退出程序

提示:
  • 每步会自动保存RGB图像、地图和组合视图
  • 建议查看 combined/ 文件夹中的图像做决策
  • Episode结束后会自动生成历史视频
  • 可以随时输入 'exit' 退出程序

评估指标:
  • distance_to_goal: 到目标距离 (< 3米为成功)
  • success: 是否成功 (0或1)
  • SPL: 路径效率 (0-1，越高越好)
  • path_length: 总路径长度
    """)
    print("="*70)
    print()


def main():
    """主函数"""
    print_banner()
    
    # 检查依赖
    if not check_dependencies():
        sys.exit(1)
    
    # 交互式配置
    config_path, output_dir = interactive_setup()
    
    if not config_path:
        sys.exit(1)
    
    # 打印说明
    print_instructions()
    
    # 确认开始
    input("按回车键开始... ")
    print()
    
    # 导入并运行主程序
    try:
        from manual_control import run_manual_control
        run_manual_control(config_path, output_dir)
    except ImportError:
        print("❌ 找不到 manual_control.py")
        print("请确保 manual_control.py 在同一目录下")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\n\n⚠️  用户中断")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ 错误: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    # 检查命令行参数
    if len(sys.argv) > 1:
        if sys.argv[1] in ["-h", "--help", "help"]:
            print("使用方法:")
            print("  python start_manual_control.py              # 交互式配置")
            print("  python start_manual_control.py <config>     # 快速启动")
            print("  python start_manual_control.py <config> <output_dir>")
            print("\n示例:")
            print("  python start_manual_control.py VLN_CE/habitat_extensions/config/vlnce_task.yaml")
            sys.exit(0)
        
        # 直接启动
        config_path = sys.argv[1]
        output_dir = sys.argv[2] if len(sys.argv) > 2 else "./manual_control_output"
        
        if not os.path.exists(config_path):
            print(f"❌ 配置文件不存在: {config_path}")
            sys.exit(1)
        
        print_banner()
        print(f"📄 配置文件: {config_path}")
        print(f"📁 输出目录: {output_dir}\n")
        
        try:
            from manual_control import run_manual_control
            run_manual_control(config_path, output_dir)
        except Exception as e:
            print(f"❌ 错误: {e}")
            import traceback
            traceback.print_exc()
            sys.exit(1)
    else:
        # 交互式模式
        main()
