#!/bin/bash

echo "正在停止所有评估进程..."
pids=$(ps aux | grep 'python navid/eval_navid_vlnce.py' | grep -v grep | awk '{print $2}')

if [ -z "$pids" ]; then
    echo "没有找到正在运行的评估进程。"
else
    kill $pids
    echo "已停止以下进程: $pids"
fi
