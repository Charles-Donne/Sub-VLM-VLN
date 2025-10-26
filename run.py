#!/usr/bin/env python3
"""
NaVid-VLN-CE 评估脚本
用于在VLN-CE数据集上评估NaVid模型的导航性能
支持多GPU并行评估
"""
import numpy as np
import argparse
from habitat.datasets import make_dataset
from VLN_CE.vlnce_baselines.config.default import get_config
from navid_agent import evaluate_agent



def main():
    """主函数：解析命令行参数并启动评估"""
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--exp-config",
        type=str,
        required=True,
        help="path to config yaml containing info about experiment",
    )
    parser.add_argument(
        "--split-num",
        type=int,
        required=True,
        help="chunks of evluation"
    )
    
    parser.add_argument(
        "--split-id",
        type=int,
        required=True,
        help="chunks ID of evluation"

    )

    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
        help="location of model weights"

    )

    parser.add_argument(
        "--result-path",
        type=str,
        required=True,
        help="location to save results"

    )

    args = parser.parse_args()
    run_exp(**vars(args))


def run_exp(exp_config: str, split_num: str, split_id: str, model_path: str, result_path: str, opts=None) -> None:
    """Runs experiment given mode and config

    Args:
        exp_config: 实验配置文件路径
        split_num: 数据集分块总数
        split_id: 当前分块ID
        model_path: 模型权重文件路径
        result_path: 结果保存路径
        opts: 额外的配置选项列表
    """
    # 加载配置文件
    config = get_config(exp_config, opts)
    
    # 创建数据集并按episode ID排序
    dataset = make_dataset(id_dataset=config.TASK_CONFIG.DATASET.TYPE, config=config.TASK_CONFIG.DATASET)
    dataset.episodes.sort(key=lambda ep: ep.episode_id)
    
    # 设置随机种子并划分数据集
    np.random.seed(42)
    dataset_split = dataset.get_splits(split_num)[split_id]
    
    # 执行评估
    evaluate_agent(config, split_id, dataset_split, model_path, result_path)


    # # 检查分块是否不重叠（调试用）
    # test_cur_splits = [int(item.episode_id) for item in dataset_split.episodes]
    # with open(f"test_cur_splits_{split_id}.txt", "w") as f:
    #     for item in test_cur_splits:
    #         f.write(str(item) + "\n")
  



if __name__ == "__main__":
    main()
