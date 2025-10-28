"""
观察收集模块
负责从环境中收集8方向观察图像并保存
"""
import os
import cv2
import numpy as np
from typing import List, Dict, Tuple


class ObservationCollector:
    """观察收集器 - 负责8方向图像采集"""
    
    # 8个方向的名称（从前方开始顺时针）
    DIRECTION_NAMES = [
        "前方 (0°)",
        "右前方 (45°)",
        "右方 (90°)",
        "右后方 (135°)",
        "后方 (180°)",
        "左后方 (225°)",
        "左方 (270°)",
        "左前方 (315°)"
    ]
    
    # 观察键名映射
    OBSERVATION_KEYS = [
        "rgb",              # 前方
        "rgb_front_right",  # 右前方
        "rgb_right",        # 右方
        "rgb_back_right",   # 右后方
        "rgb_back",         # 后方
        "rgb_back_left",    # 左后方
        "rgb_left",         # 左方
        "rgb_front_left"    # 左前方
    ]
    
    def __init__(self, output_dir: str):
        """
        初始化收集器
        
        Args:
            output_dir: 输出目录
        """
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
    
    def collect_8_directions(self, 
                            observations: Dict,
                            save_prefix: str = "observation") -> Tuple[List[str], List[str]]:
        """
        收集8个方向的观察图像并保存
        
        Args:
            observations: 环境观测字典
            save_prefix: 保存文件名前缀
            
        Returns:
            (image_paths, direction_names) - 图像路径列表和方向名称列表
        """
        image_paths = []
        collected_directions = []
        
        for i, (key, direction_name) in enumerate(zip(self.OBSERVATION_KEYS, self.DIRECTION_NAMES)):
            if key in observations:
                # 获取图像
                img = observations[key]
                
                # 保存路径
                filename = f"{save_prefix}_dir{i}_{key}.jpg"
                filepath = os.path.join(self.output_dir, filename)
                
                # 保存图像
                cv2.imwrite(filepath, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
                
                image_paths.append(filepath)
                collected_directions.append(direction_name)
        
        print(f"✓ 收集了 {len(image_paths)} 个方向的观察图像")
        
        return image_paths, collected_directions
    
    def create_compass_visualization(self,
                                     observations: Dict,
                                     save_path: str = None) -> np.ndarray:
        """
        创建罗盘式可视化（8方向环绕布局）
        
        Args:
            observations: 环境观测字典
            save_path: 保存路径（可选）
            
        Returns:
            可视化图像
        """
        # 检查是否有足够的观察
        available_views = [key for key in self.OBSERVATION_KEYS if key in observations]
        
        if len(available_views) < 2:
            print("⚠️  观察视角不足，无法创建罗盘可视化")
            return None
        
        # 获取图像尺寸
        sample_img = observations[available_views[0]]
        h, w = sample_img.shape[:2]
        
        # 创建3x3网格（中心放置罗盘图标或地图）
        # 布局：
        # ┌────────┬────────┬────────┐
        # │ 左前   │  前方  │ 右前   │
        # ├────────┼────────┼────────┤
        # │  左方  │ 中心   │  右方  │
        # ├────────┼────────┼────────┤
        # │ 左后   │  后方  │ 右后   │
        # └────────┴────────┴────────┘
        
        # 位置映射 (row, col)
        position_map = {
            0: (0, 1),  # 前方 -> 第一行中间
            1: (0, 2),  # 右前方 -> 第一行右边
            2: (1, 2),  # 右方 -> 第二行右边
            3: (2, 2),  # 右后方 -> 第三行右边
            4: (2, 1),  # 后方 -> 第三行中间
            5: (2, 0),  # 左后方 -> 第三行左边
            6: (1, 0),  # 左方 -> 第二行左边
            7: (0, 0)   # 左前方 -> 第一行左边
        }
        
        # 创建网格
        grid = [[None for _ in range(3)] for _ in range(3)]
        
        # 填充观察图像
        for i, key in enumerate(self.OBSERVATION_KEYS):
            if key in observations:
                row, col = position_map[i]
                grid[row][col] = observations[key]
        
        # 创建中心图像（罗盘或文字）
        center_img = np.zeros((h, w, 3), dtype=np.uint8)
        center_img.fill(50)  # 深灰色背景
        
        # 绘制罗盘方向
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(center_img, "N", (w//2 - 15, 40), font, 1.2, (255, 255, 255), 2)
        cv2.putText(center_img, "S", (w//2 - 15, h - 20), font, 1.2, (255, 255, 255), 2)
        cv2.putText(center_img, "E", (w - 40, h//2 + 10), font, 1.2, (255, 255, 255), 2)
        cv2.putText(center_img, "W", (20, h//2 + 10), font, 1.2, (255, 255, 255), 2)
        
        # 绘制圆圈
        cv2.circle(center_img, (w//2, h//2), min(w, h)//3, (100, 100, 100), 2)
        
        grid[1][1] = center_img
        
        # 填充空位
        for row in range(3):
            for col in range(3):
                if grid[row][col] is None:
                    empty_img = np.zeros((h, w, 3), dtype=np.uint8)
                    empty_img.fill(30)  # 更深的灰色
                    grid[row][col] = empty_img
        
        # 拼接行
        rows = []
        for row_imgs in grid:
            row = np.concatenate(row_imgs, axis=1)
            rows.append(row)
        
        # 拼接列
        result = np.vstack(rows)
        
        # 添加方向标签
        label_positions = [
            (w, 30),      # 前方
            (w*2, 30),    # 右前方
            (w*2, h + 30),  # 右方
            (w*2, h*2 + 30),  # 右后方
            (w, h*2 + 30),    # 后方
            (20, h*2 + 30),   # 左后方
            (20, h + 30),     # 左方
            (20, 30)          # 左前方
        ]
        
        for i, (x, y) in enumerate(label_positions):
            if i < len(self.DIRECTION_NAMES):
                # 简短标签
                label = self.DIRECTION_NAMES[i].split()[0]
                cv2.putText(result, label, (x + 10, y), 
                           font, 0.6, (255, 255, 0), 2)
        
        # 保存
        if save_path:
            cv2.imwrite(save_path, cv2.cvtColor(result, cv2.COLOR_RGB2BGR))
            print(f"✓ 罗盘可视化已保存: {save_path}")
        
        return result
    
    def get_direction_summary(self, observations: Dict) -> str:
        """
        生成方向观察摘要（文本）
        
        Args:
            observations: 环境观测字典
            
        Returns:
            摘要文本
        """
        summary_lines = ["观察摘要（8个方向）:"]
        
        for i, (key, direction) in enumerate(zip(self.OBSERVATION_KEYS, self.DIRECTION_NAMES)):
            if key in observations:
                summary_lines.append(f"  [{i+1}] {direction}: 图像已采集")
            else:
                summary_lines.append(f"  [{i+1}] {direction}: 未采集")
        
        return "\n".join(summary_lines)
