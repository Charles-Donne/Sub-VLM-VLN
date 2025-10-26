"""
NaVid 智能体模块
实现基于视频-语言模型的视觉语言导航智能体
支持增量式视觉token复用，提高评估效率
"""

import json
from datetime import datetime
import numpy as np
from habitat import Env
from habitat.core.agent import Agent
from tqdm import trange
import os
import re
import torch
import cv2
import imageio
from habitat.utils.visualizations import maps
import random

from navid.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from navid.conversation import conv_templates, SeparatorStyle
from navid.model.builder import load_pretrained_model
from navid.mm_utils import tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria


def evaluate_agent(config, split_id, dataset, model_path, result_path) -> None:
    """
    评估NaVid智能体的导航性能
    
    Args:
        config: 实验配置对象
        split_id: 数据分块ID
        dataset: 评估数据集
        model_path: 模型权重路径
        result_path: 结果保存路径
    """
    # 创建结果目录
    os.makedirs(os.path.join(result_path, "log"), exist_ok=True)
    os.makedirs(os.path.join(result_path, "video"), exist_ok=True)
    
    # 管理已评估episode记录（支持断点续评）
    eval_record = os.path.join(result_path, "evaluated.txt")
    evaluated_ids = set()
    if os.path.exists(eval_record):
        with open(eval_record, "r") as f:
            evaluated_ids = {line.strip() for line in f.readlines()}
    
    print(f"已评估ID数量: {len(evaluated_ids)}")
    
    # 过滤未评估的episode
    unevaluated = [
        ep for ep in dataset.episodes 
        if str(ep.episode_id) not in evaluated_ids
    ]
    
    # 限制单次评估数量
    max_episodes = min(20, len(unevaluated))
    
    if not unevaluated:
        print("所有episode已完成评估！")
        return
    
    dataset.episodes = unevaluated[:max_episodes]
    evaluating_ids = [str(ep.episode_id) for ep in dataset.episodes]
    print(f"即将评估的ID: {evaluating_ids}")
    
    # 初始化环境和智能体
    env = Env(config.TASK_CONFIG, dataset)
    agent = NaVid_Agent(model_path, result_path)
    
    num_episodes = len(env.episodes)
    print(f"实际评估 {num_episodes} 个episode")
    
    # 早停参数
    EARLY_STOP_ROTATION = config.EVAL.EARLY_STOP_ROTATION
    EARLY_STOP_STEPS = config.EVAL.EARLY_STOP_STEPS
    
    # 评估指标
    # distance_to_goal: 停止时智能体与目标点的距离(米)，越小越好
    # success: 成功率，智能体是否在3米内停止(0或1)
    # spl: Success weighted by Path Length，成功率与路径效率的综合指标
    #      计算公式: success * (最短路径长度 / 实际路径长度)
    #      范围[0,1]，越高表示既成功又高效
    # path_length: 智能体实际行走的路径长度(米)
    # oracle_success: 预言成功率，整个轨迹中是否曾经到达过目标3米内(0或1)
    #                 用于评估智能体是否找到过目标但错过了停止
    target_key = {"distance_to_goal", "success", "spl", "path_length", "oracle_success"}

    # 用于统计所有episode的结果
    all_results = []
    
    count = 0
    progress = trange(num_episodes, desc=config.EVAL.IDENTIFICATION+"-{}".format(split_id))
    
    # 主评估循环
    for _ in progress:
        if count >= max_episodes:
            print(f"已达到最大episode数 ({max_episodes})，提前结束评估")
            break
            
        # 重置环境和智能体
        env.reset()
        agent.reset()
        
        # 执行一轮完整的episode评估（封装在agent类中）
        iter_step = agent.run_episode(env, EARLY_STOP_ROTATION, EARLY_STOP_STEPS)
            
        # 收集本次episode的评估指标（如距离目标、成功率等）
        info = env.get_metrics()
        result_dict = dict()
        # 只保留关心的评估指标
        result_dict = {k: info[k] for k in target_key if k in info}
        # 记录当前episode的唯一ID
        result_dict["id"] = env.current_episode.episode_id
        count += 1

        # 保存本次 episode 的评估指标到 log 目录
        log_dir = os.path.join(result_path, "log")
        os.makedirs(log_dir, exist_ok=True)
        stats_path = os.path.join(log_dir, f"stats_{env.current_episode.episode_id}.json")
        with open(stats_path, "w") as f:
            json.dump(result_dict, f, indent=4)
        
        # 将当前结果添加到汇总列表
        all_results.append(result_dict)

        # 动态更新进度条描述，显示当前进度
        progress.set_description(f"{config.EVAL.IDENTIFICATION}-{split_id} [ep {count}/{max_episodes}]")
    
    # 保存已评估episode ID（避免重复评估）
    new_evaluated_ids = {str(ep.episode_id) for ep in dataset.episodes}
    with open(eval_record, "a") as f:
        for ep_id in new_evaluated_ids:
            if ep_id not in evaluated_ids:
                f.write(ep_id + "\n")
                evaluated_ids.add(ep_id)
    
    # 计算并保存本次评估的汇总统计
    if all_results:
        # 轮次编号：通过 index 文件确保每次递增且不会覆盖
        index_file = os.path.join(result_path, "summary_index.txt")
        try:
            last_idx = int(open(index_file, "r").read().strip()) if os.path.exists(index_file) else 0
        except Exception:
            last_idx = 0
        run_idx = last_idx + 1
        with open(index_file, "w") as idx_f:
            idx_f.write(str(run_idx))

        # 汇总文件：统一写入一个 summary.txt，按轮次追加
        summary_file = os.path.join(result_path, "summary.txt")
        now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with open(summary_file, "a") as f:
            f.write("\n" + "#"*80 + "\n")
            f.write(f"NaVid 评估汇总报告 | 第 {run_idx} 次评估 | Split {split_id} | {now_str}\n")
            f.write("#"*80 + "\n\n")
            f.write(f"评估标识: {getattr(getattr(config, 'EVAL', object()), 'IDENTIFICATION', 'N/A')}\n")
            f.write(f"本轮评估episode数: {count}\n")
            f.write("\n")

            # 列出所有测试的episode
            f.write("测试的Episode列表:\n")
            for i, result in enumerate(all_results, 1):
                f.write(f"  {i}. Episode {result['id']}\n")
            f.write("\n")

            # 计算各指标的平均值
            f.write("评估指标汇总:\n")
            f.write("-"*40 + "\n")
            metrics_to_average = ["distance_to_goal", "success", "spl", "path_length", "oracle_success"]
            for metric in metrics_to_average:
                values = [r[metric] for r in all_results if metric in r]
                if values:
                    avg_value = sum(values) / len(values)
                    f.write(f"{metric:20s}: {avg_value:.4f}\n")
            f.write("\n")

        print(f"新增评估 {len(new_evaluated_ids)} 个episode")
        print(f"汇总报告已追加到: {summary_file} (第 {run_idx} 次评估)")



class NaVid_Agent(Agent):
    """
    NaVid智能体类
    基于视频-语言模型的导航智能体，支持：
    - 历史视觉token复用（提高推理效率）
    - 增量式图像处理
    - 动作序列缓存
    """
    
    def __init__(self, model_path, result_path, require_map=True):
        """
        初始化NaVid智能体
        
        Args:
            model_path: 模型权重路径
            result_path: 结果保存路径
            require_map: 是否生成可视化地图视频
        """
        print("Initialize NaVid")
        
        self.result_path = result_path
        self.require_map = require_map
        self.conv_mode = "vicuna_v1"
        
        # 创建输出目录
        os.makedirs(self.result_path, exist_ok=True)
        os.makedirs(os.path.join(self.result_path, "log"), exist_ok=True)
        os.makedirs(os.path.join(self.result_path, "video"), exist_ok=True)

        # 加载预训练模型
        self.model_name = get_model_name_from_path(model_path)
        self.tokenizer, self.model, self.image_processor, self.context_len = load_pretrained_model(
            model_path, None, get_model_name_from_path(model_path)
        )

        print("Initialization Complete")

        
        self.promt_template = "Imagine you are a robot programmed for navigation tasks. You have been given a video of historical observations and an image of the current observation <image>. Your assigned task is: '{}'. Analyze this series of images to decide your next move, which could involve turning left or right by a specific degree or moving forward a certain distance."

        self.history_rgb_tensor = None
        
        self.rgb_list = []
        self.topdown_map_list = []

        self.count_id = 0
        self.reset()


    def run_episode(self, env, early_stop_rotation=20, early_stop_steps=500):
        """
        执行一轮完整的episode评估循环
        
        Args:
            env: Habitat环境对象
            early_stop_rotation: 最大连续旋转次数（早停阈值）
            early_stop_steps: 最大步数（早停阈值）
            
        Returns:
            obs: 最后一步的观测
            iter_step: 总步数
        """
        # 连续旋转计数器（用于早停）
        continuse_rotation_count = 0
        last_dtg = 999
        iter_step = 0
        obs = None
        
        while not env.episode_over:
            info = env.get_metrics()
            
            # 检测是否持续原地旋转（通过距离变化判断）
            if info["distance_to_goal"] != last_dtg:
                last_dtg = info["distance_to_goal"]
                continuse_rotation_count = 0
            else:
                continuse_rotation_count += 1 
            
            # 获取智能体动作（使用当前观测obs）
            action = self.act(obs if obs is not None else env.reset(), info, env.current_episode.episode_id)
            
            # 早停条件：过多旋转或超过最大步数
            if continuse_rotation_count > early_stop_rotation or iter_step > early_stop_steps:
                action = {"action": 0}  # 强制停止动作

            iter_step += 1
            # 同步执行动作并等待完成，返回新观测
            # env.step()内部流程：
            # 1. 执行动作（前进/转向/停止）
            # 2. 等待物理模拟完成
            # 3. 更新机器人位置和朝向
            # 4. 从新位置渲染RGB图像、深度图等传感器数据
            # 5. 返回新的observation字典（包含rgb、instruction等）
            obs = env.step(action)  # 阻塞调用，确保动作完成后才继续
        
        return iter_step


    def process_images(self, rgb_list):
        """
        增量式图像处理：只处理新增图像，复用历史视觉token
        显著提升多步导航任务的推理效率
        
        Args:
            rgb_list: RGB图像列表
            
        Returns:
            包含完整视觉token的列表
        """
        # 确定需要处理的起始索引
        start_img_index = 0
        if self.history_rgb_tensor is not None:
            start_img_index = self.history_rgb_tensor.shape[0]
        
        # 只处理新增图像
        batch_image = np.asarray(rgb_list[start_img_index:])
        video = self.image_processor.preprocess(batch_image, return_tensors='pt')['pixel_values'].half().cuda()

        # 拼接历史和新增token
        if self.history_rgb_tensor is None:
            self.history_rgb_tensor = video
        else:
            self.history_rgb_tensor = torch.cat((self.history_rgb_tensor, video), dim=0)
        
        return [self.history_rgb_tensor]



    def predict_inference(self, prompt):
        """
        执行模型推理，生成导航决策
        
        Args:
            prompt: 包含任务指令的提示词
            
        Returns:
            模型输出的导航指令（如"turn left 30 degrees"）
        """
        question = prompt.replace(DEFAULT_IMAGE_TOKEN, '').replace('\n', '')
        qs = prompt

        # 定义特殊token
        VIDEO_START_SPECIAL_TOKEN = "<video_special>"
        VIDEO_END_SPECIAL_TOKEN = "</video_special>"
        IMAGE_START_TOKEN = "<image_special>"
        IMAGE_END_TOKEN = "</image_special>"
        NAVIGATION_SPECIAL_TOKEN = "[Navigation]"
        IAMGE_SEPARATOR = "<image_sep>"
        
        # token化特殊标记
        image_start_special_token = self.tokenizer(IMAGE_START_TOKEN, return_tensors="pt").input_ids[0][1:].cuda()
        image_end_special_token = self.tokenizer(IMAGE_END_TOKEN, return_tensors="pt").input_ids[0][1:].cuda()
        video_start_special_token = self.tokenizer(VIDEO_START_SPECIAL_TOKEN, return_tensors="pt").input_ids[0][1:].cuda()
        video_end_special_token = self.tokenizer(VIDEO_END_SPECIAL_TOKEN, return_tensors="pt").input_ids[0][1:].cuda()
        navigation_special_token = self.tokenizer(NAVIGATION_SPECIAL_TOKEN, return_tensors="pt").input_ids[0][1:].cuda()
        image_seperator = self.tokenizer(IAMGE_SEPARATOR, return_tensors="pt").input_ids[0][1:].cuda()

        # 构建提示词
        if self.model.config.mm_use_im_start_end:
            qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + qs.replace('<image>', '')
        else:
            qs = DEFAULT_IMAGE_TOKEN + '\n' + qs.replace('<image>', '')

        # 使用对话模板
        conv = conv_templates[self.conv_mode].copy()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        # token化并插入特殊标记
        token_prompt = tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').cuda()
        indices_to_replace = torch.where(token_prompt == -200)[0]
        new_list = []
        
        while indices_to_replace.numel() > 0:
            idx = indices_to_replace[0]
            new_list.append(token_prompt[:idx])
            new_list.append(video_start_special_token)
            new_list.append(image_seperator)
            new_list.append(token_prompt[idx:idx + 1])
            new_list.append(video_end_special_token)
            new_list.append(image_start_special_token)
            new_list.append(image_end_special_token)
            new_list.append(navigation_special_token)
            token_prompt = token_prompt[idx + 1:]
            indices_to_replace = torch.where(token_prompt == -200)[0]
            
        if token_prompt.numel() > 0:
            new_list.append(token_prompt)
        input_ids = torch.cat(new_list, dim=0).unsqueeze(0)

        # 停止条件
        stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
        keywords = [stop_str]
        stopping_criteria = KeywordsStoppingCriteria(keywords, self.tokenizer, input_ids)

        # 处理图像
        imgs = self.process_images(self.rgb_list)

        # 模型生成
        cur_prompt = question
        with torch.inference_mode():
            self.model.update_prompt([[cur_prompt]])
            output_ids = self.model.generate(
                input_ids,
                images=imgs,
                do_sample=True,
                temperature=0.2,
                max_new_tokens=1024,
                use_cache=True,
                stopping_criteria=[stopping_criteria]
            )

        # 解码输出
        input_token_len = input_ids.shape[1]
        n_diff_input_output = (input_ids != output_ids[:, :input_token_len]).sum().item()
        if n_diff_input_output > 0:
            print(f'[Warning] {n_diff_input_output} output_ids are not the same as the input_ids')
        
        outputs = self.tokenizer.batch_decode(output_ids[:, input_token_len:], skip_special_tokens=True)[0]
        outputs = outputs.strip()
        if outputs.endswith(stop_str):
            outputs = outputs[:-len(stop_str)]
        outputs = outputs.strip()

        return outputs




    def extract_result(self, output):
        """
        从模型输出中提取动作和参数
        
        Args:
            output: 模型输出文本
            
        Returns:
            (action_id, value): 动作ID和数值参数
                - 0: stop
                - 1: move forward
                - 2: turn left
                - 3: turn right
        """
        if "stop" in output:
            return 0, None
        elif "forward" in output:
            match = re.search(r'-?\d+', output)
            if match is None:
                return None, None
            return 1, float(match.group())
        elif "left" in output:
            match = re.search(r'-?\d+', output)
            if match is None:
                return None, None
            return 2, float(match.group())
        elif "right" in output:
            match = re.search(r'-?\d+', output)
            if match is None:
                return None, None
            return 3, float(match.group())

        return None, None


    def addtext(self, image, instuction, navigation):
        """
        在图像上添加指令和导航决策文本（用于可视化）
        
        Args:
            image: 原始图像
            instuction: 任务指令文本
            navigation: 导航决策文本
            
        Returns:
            添加文本后的图像
        """
        h, w = image.shape[:2]
        new_height = h + 150
        new_image = np.zeros((new_height, w, 3), np.uint8)
        new_image.fill(255)  
        new_image[:h, :w] = image

        font = cv2.FONT_HERSHEY_SIMPLEX
        textsize = cv2.getTextSize(instuction, font, 0.5, 2)[0]
        textY = h + (50 + textsize[1]) // 2
        y_line = textY + 0 * textsize[1]

        # 自动换行处理
        words = instuction.split(' ')
        x = 10
        line = ""

        for word in words:
            test_line = line + ' ' + word if line else word
            test_line_size, _ = cv2.getTextSize(test_line, font, 0.5, 2)

            if test_line_size[0] > image.shape[1] - x:
                cv2.putText(new_image, line, (x, y_line), font, 0.5, (0, 0, 0), 2)
                line = word
                y_line += textsize[1] + 5
            else:
                line = test_line

        if line:
            cv2.putText(new_image, line, (x, y_line), font, 0.5, (0, 0, 0), 2)

        # 添加导航决策
        y_line = y_line + 1 * textsize[1] + 10
        new_image = cv2.putText(new_image, navigation, (x, y_line), font, 0.5, (0, 0, 0), 2)

        return new_image



    def reset(self):
        """重置智能体状态，用于开始新的episode"""
        # 保存上一个episode的可视化视频
        if self.require_map:
            if len(self.topdown_map_list) != 0:
                output_video_path = os.path.join(self.result_path, "video", "{}.gif".format(self.episode_id))
                imageio.mimsave(output_video_path, self.topdown_map_list)

        # 清空状态
        self.history_rgb_tensor = None
        self.transformation_list = []
        self.rgb_list = []
        self.topdown_map_list = []
        self.count_id += 1
        self.pending_action_list = []


    def act(self, observations, info, episode_id):
        """
        执行单步动作决策
        
        Args:
            observations: 环境观测字典，由Habitat环境返回，包含：
                - "rgb": 当前视角的RGB图像 (numpy数组)
                - "instruction": 导航指令字典，包含：
                    - "text": 自然语言指令文本 (str)
                      例如："Go to the kitchen and find the refrigerator"
                      来源：VLN-CE数据集中的episode定义
                      每个episode在数据集中预先定义了instruction_text
            info: 环境信息（包含distance_to_goal等指标）
            episode_id: 当前episode ID
            
        Returns:
            动作字典 {"action": action_id}
        """
        self.episode_id = episode_id
        rgb = observations["rgb"]
        self.rgb_list.append(rgb)

        # 生成可视化地图
        if self.require_map:
            top_down_map = maps.colorize_draw_agent_and_fit_to_height(info["top_down_map_vlnce"], rgb.shape[0])
            output_im = np.concatenate((rgb, top_down_map), axis=1)

        # 【动作缓存机制】避免每步都调用耗时的VLM推理
        # 原因：模型输出高层指令（如"前进75cm"），需要拆分为多个底层动作（3个"前进25cm"）
        # 如果缓存中还有待执行动作，直接返回，无需重新推理
        if len(self.pending_action_list) != 0:
            temp_action = self.pending_action_list.pop(0)  # 弹出队列中的第一个动作
            
            if self.require_map:
                img = self.addtext(output_im, observations["instruction"]["text"], 
                                  "Pending action: {}".format(temp_action))
                self.topdown_map_list.append(img)
            
            return {"action": temp_action}  # 直接返回缓存动作，跳过模型推理

        # 【模型推理】缓存为空时，才调用耗时的VLM生成新决策
        # 1. 构建完整的提示词
        navigation_qs = self.promt_template.format(observations["instruction"]["text"])
        
        # 2. 调用VLM模型进行推理（耗时操作）
        navigation = self.predict_inference(navigation_qs)  # GPU推理，约2-3秒
        
        # 3. 可视化：在地图上叠加文本（用于生成GIF视频）
        if self.require_map:
            # 将instruction和模型输出的决策都标注在图像上
            # 例如图像底部显示：
            #   "Walk to the kitchen..."  (任务指令)
            #   "turn left 60 degrees"    (模型决策)
            img = self.addtext(output_im, observations["instruction"]["text"], navigation)
            self.topdown_map_list.append(img)  # 添加到视频帧列表

        # 解析动作
        action_index, num = self.extract_result(navigation[:-1])

        # 【高层到底层的动作分解】将模型输出的连续指令拆分为离散的原子动作
        # 例：模型输出 "move forward 75 meters" 
        #     → 拆分为 [1, 1, 1] 三个底层前进动作（每个25cm）
        #     → 依次执行，每次执行后环境会更新观测
        if action_index == 0:
            self.pending_action_list.append(0)  # 停止动作不需要拆分
        elif action_index == 1:
            # 前进：每次底层动作前进25cm，最多排队3步
            # 例：num=75 → int(75/25)=3 → [1,1,1]
            for _ in range(min(3, int(num / 25))):
                self.pending_action_list.append(1)
        elif action_index == 2:
            # 左转：每次底层动作转30度，最多排队3步
            # 例：num=90 → int(90/30)=3 → [2,2,2]
            for _ in range(min(3, int(num / 30))):
                self.pending_action_list.append(2)
        elif action_index == 3:
            # 右转：每次底层动作转30度，最多排队3步
            for _ in range(min(3, int(num / 30))):
                self.pending_action_list.append(3)
        
        # 容错：如果解析失败或无动作，随机选择一个探索动作
        if action_index is None or len(self.pending_action_list) == 0:
            self.pending_action_list.append(random.randint(1, 3))

        # 返回队列中的第一个动作，剩余动作留在缓存中
        return {"action": self.pending_action_list.pop(0)}

