#!/bin/bash

# Episode分析工具 - 超简化版

# 用法说明
if [[ "$1" == "-h" ]] || [[ "$1" == "--help" ]]; then
    echo "用法:"
    echo "  $0                    # 随机选1个episode"
    echo "  $0 -a                 # 随机选1个episode + LLM分析"
    echo "  $0 123                # 指定episode ID: 123"
    echo "  $0 123 -a             # 指定episode + LLM分析"
    exit 0
fi

# 解析参数
EPISODE_ID=""
ANALYZE=""

if [[ "$1" =~ ^[0-9]+$ ]]; then
    EPISODE_ID="--episode-id $1"
    shift
fi

if [[ "$1" == "-a" ]] || [[ "$2" == "-a" ]]; then
    ANALYZE="--analyze"
fi

# 运行
python3 analyze_episode.py $EPISODE_ID $ANALYZE
