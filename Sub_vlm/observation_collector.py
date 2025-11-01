"""
观察收集模块
负责从环境中收集8方向观察图像并保存
"""
import os
import cv2
import numpy as np
from typing import List, Dict, Tuple, Optional
from habitat.utils.visualizations import maps

try:
    import imageio
    HAS_IMAGEIO = True
except ImportError:
    HAS_IMAGEIO = False


class ObservationCollector:
    """观察收集器 - 负责8方向图像采集和可视化"""
    
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
        self.maps_dir = None
        self.video_frames = []
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
                img = observations[key]
                filename = f"{save_prefix}_dir{i}_{key}.jpg"
                filepath = os.path.join(self.output_dir, filename)
                cv2.imwrite(filepath, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
                image_paths.append(filepath)
                collected_directions.append(direction_name)
        
        return image_paths, collected_directions
    
    def setup_maps_dir(self, episode_dir: str):
        """设置地图可视化目录"""
        self.maps_dir = os.path.join(episode_dir, "maps")
        os.makedirs(self.maps_dir, exist_ok=True)
        self.video_frames = []
    
    def save_step_visualization(self,
                               observations: Dict,
                               info: Dict,
                               step: int,
                               instruction: str,
                               current_subtask: str = None,
                               distance: float = 0.0) -> str:
        """
        保存单步可视化：左边第一人称视角 + 右边地图 + 底部文本信息
        
        Args:
            observations: 环境观测
            info: 环境指标
            step: 步数
            instruction: 全局指令
            current_subtask: 当前子任务指令
            distance: 到目标距离
            
        Returns:
            保存的图像路径
        """
        if not self.maps_dir or "rgb" not in observations:
            return None
        
        # 获取第一人称RGB
        rgb = observations["rgb"]
        
        # 获取地图
        if "top_down_map_vlnce" in info:
            top_down_map = maps.colorize_draw_agent_and_fit_to_height(
                info["top_down_map_vlnce"], rgb.shape[0]
            )
        else:
            # 如果没有地图，创建空白占位
            top_down_map = np.zeros_like(rgb)
        
        # 拼接：左边RGB + 右边地图
        combined = np.concatenate((rgb, top_down_map), axis=1)
        
        # 添加文本信息
        combined = self._add_text_overlay(
            combined, 
            instruction, 
            current_subtask, 
            step, 
            distance
        )
        
        # 保存
        filename = f"step{step:04d}_visualization.jpg"
        filepath = os.path.join(self.maps_dir, filename)
        cv2.imwrite(filepath, cv2.cvtColor(combined, cv2.COLOR_RGB2BGR))
        
        # 记录到视频帧列表
        self.video_frames.append(combined)
        
        return filepath
    
    def _add_text_overlay(self,
                         image: np.ndarray,
                         instruction: str,
                         current_subtask: Optional[str],
                         step: int,
                         distance: float) -> np.ndarray:
        """在图像底部添加文本信息"""
        img = image.copy()
        h, w = img.shape[:2]
        
        # 创建文本区域
        text_height = 120
        text_area = np.zeros((text_height, w, 3), dtype=np.uint8)
        text_area.fill(40)  # 深灰色背景
        
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        thickness = 1
        color = (255, 255, 255)
        
        y_offset = 25
        
        # 步数和距离
        metrics_text = f"Step: {step} | Distance: {distance:.2f}m"
        cv2.putText(text_area, metrics_text, (10, y_offset), font, font_scale, (0, 255, 255), thickness)
        y_offset += 30
        
        # 全局指令
        instruction_lines = self._wrap_text(instruction, w - 20, font, font_scale)
        for line in instruction_lines[:2]:  # 最多2行
            cv2.putText(text_area, line, (10, y_offset), font, font_scale, color, thickness)
            y_offset += 25
        
        # 当前子任务
        if current_subtask:
            y_offset += 5
            subtask_text = f"Subtask: {current_subtask}"
            subtask_lines = self._wrap_text(subtask_text, w - 20, font, font_scale)
            for line in subtask_lines[:1]:  # 最多1行
                cv2.putText(text_area, line, (10, y_offset), font, font_scale, (0, 255, 0), thickness)
        
        # 拼接文本区域到图像底部
        result = np.vstack([img, text_area])
        return result
    
    def _wrap_text(self, text: str, max_width: int, font, font_scale: float) -> List[str]:
        """文本换行"""
        words = text.split()
        lines = []
        current_line = ""
        
        for word in words:
            test_line = f"{current_line} {word}".strip()
            (text_width, _), _ = cv2.getTextSize(test_line, font, font_scale, 1)
            
            if text_width <= max_width:
                current_line = test_line
            else:
                if current_line:
                    lines.append(current_line)
                current_line = word
        
        if current_line:
            lines.append(current_line)
        
        return lines
    
    def save_video(self, output_path: str = None, fps: int = 2) -> Optional[str]:
        """
        将所有帧保存为视频
        
        Args:
            output_path: 输出路径（可选，默认在maps目录下）
            fps: 帧率
            
        Returns:
            视频路径
        """
        if not self.video_frames:
            print("⚠️  No frames to save")
            return None
        
        if output_path is None and self.maps_dir:
            output_path = os.path.join(self.maps_dir, "navigation_video.mp4")
        
        if not output_path:
            return None
        
        try:
            # 获取帧尺寸
            h, w = self.video_frames[0].shape[:2]
            
            # 尝试使用更兼容的编码器
            # 优先尝试 H264 (avc1)，如果失败则使用 mp4v
            fourcc_list = [
                cv2.VideoWriter_fourcc(*'avc1'),  # H.264
                cv2.VideoWriter_fourcc(*'mp4v'),  # MPEG-4
                cv2.VideoWriter_fourcc(*'XVID'),  # Xvid
            ]
            
            video_writer = None
            for fourcc in fourcc_list:
                video_writer = cv2.VideoWriter(output_path, fourcc, fps, (w, h))
                if video_writer.isOpened():
                    break
                video_writer.release()
                video_writer = None
            
            if not video_writer or not video_writer.isOpened():
                print("✗ Failed to create video writer")
                return None
            
            # 写入所有帧
            for frame in self.video_frames:
                # 确保帧是uint8类型
                if frame.dtype != np.uint8:
                    frame = frame.astype(np.uint8)
                video_writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
            
            video_writer.release()
            
            # 验证文件是否创建成功
            if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
                print(f"✓ Video saved: {output_path} ({len(self.video_frames)} frames, {fps} fps)")
                return output_path
            else:
                print(f"✗ Video file creation failed or empty")
                return None
                
        except Exception as e:
            print(f"✗ Error saving video: {e}")
            return None
    
    def save_gif(self, output_path: str = None, fps: int = 2, duration: float = None) -> Optional[str]:
        """
        将所有帧保存为GIF动画（备用方案）
        
        Args:
            output_path: 输出路径（可选，默认在maps目录下）
            fps: 帧率
            duration: 每帧持续时间（秒），优先于fps
            
        Returns:
            GIF路径
        """
        if not self.video_frames:
            print("⚠️  No frames to save")
            return None
        
        if not HAS_IMAGEIO:
            print("⚠️  imageio not installed, cannot create GIF")
            return None
        
        if output_path is None and self.maps_dir:
            output_path = os.path.join(self.maps_dir, "navigation.gif")
        
        if not output_path:
            return None
        
        try:
            # 转换帧为uint8 BGR格式
            frames_bgr = []
            for frame in self.video_frames:
                if frame.dtype != np.uint8:
                    frame = frame.astype(np.uint8)
                # imageio需要RGB格式
                frames_bgr.append(frame)
            
            # 计算每帧持续时间
            if duration is None:
                duration = 1.0 / fps
            
            # 保存GIF
            imageio.mimsave(output_path, frames_bgr, duration=duration, loop=0)
            
            if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
                print(f"✓ GIF saved: {output_path} ({len(self.video_frames)} frames)")
                return output_path
            else:
                print(f"✗ GIF file creation failed")
                return None
                
        except Exception as e:
            print(f"✗ Error saving GIF: {e}")
            return None

