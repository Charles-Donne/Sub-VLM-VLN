#!/bin/bash
# 指令分解工具快捷脚本

if [[ "$1" == "-h" ]] || [[ "$1" == "--help" ]]; then
    echo "用法:"
    echo "  $0 \"<instruction>\"              # 分解指定的指令"
    echo "  $0                                # 使用示例指令"
    echo "  $0 \"<instruction>\" -o result.json  # 分解并保存结果"
    echo "  $0 \"<instruction>\" -s            # 简化输出"
    echo ""
    echo "示例:"
    echo "  $0 \"Walk to the kitchen and stop at the fridge\""
    echo "  $0 -o decomposition.json"
    exit 0
fi

# 运行Python脚本
python3 decompose_instruction.py "$@"
