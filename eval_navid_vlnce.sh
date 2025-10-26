#!/bin/bash

# 评估任务分块数量（通常等于GPU数量）
CHUNKS=1

# 模型权重路径
MODEL_PATH="/root/autodl-tmp/model_zoo/navid-7b-full-224-video-fps-1-grid-2-r2r-rxr-training-split" 


# R2R 数据集配置
CONFIG_PATH="VLN_CE/vlnce_baselines/config/r2r_baselines/navid_r2r.yaml"
SAVE_PATH="/root/autodl-tmp/result/navid/r2r" 

mkdir -p $SAVE_PATH

# RxR 数据集配置（已注释）
# CONFIG_PATH="/root/navid_ws/NaVid-VLN-CE/VLN_CE/vlnce_baselines/config/rxr_baselines/navid_rxr.yaml"
# SAVE_PATH="/root/navid_ws/NaVid-VLN-CE/results2/" 


# 并行评估：在不同GPU上运行各数据块
for IDX in $(seq 0 $((CHUNKS-1))); do
    # 打印当前GPU编号
    echo $(( IDX % 8 ))
    
    # 在指定GPU上运行评估任务（后台并行）
    CUDA_VISIBLE_DEVICES=$(( IDX % 8 )) python run.py \
    --exp-config $CONFIG_PATH \
    --split-num $CHUNKS \
    --split-id $IDX \
    --model-path $MODEL_PATH \
    --result-path $SAVE_PATH &
    
done

# 等待所有后台任务完成
wait

