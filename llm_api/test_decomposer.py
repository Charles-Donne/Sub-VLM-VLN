#!/usr/bin/env python3
"""
测试指令分解器独立模块
"""

from instruction_decomposer import InstructionDecomposer

def test_decomposer():
    """测试指令分解器功能"""
    
    # 测试指令
    test_instructions = [
        "Go up stairs and turn left to stairs. Stop between stairs and large double doors.",
        "Walk down the hallway and enter the bedroom on your left. Stop by the window.",
        "Go down the stairs and go into the door on your right. Wait by the toilet."
    ]
    
    print("="*80)
    print("🧪 测试指令分解器模块")
    print("="*80)
    
    # 初始化分解器
    decomposer = InstructionDecomposer()
    
    if not decomposer.is_available():
        print("\n❌ 分解器不可用")
        print("   请检查 llm_api/api_config.yaml 是否存在")
        return
    
    print("\n✅ 分解器初始化成功\n")
    
    # 测试每条指令
    for i, instruction in enumerate(test_instructions, 1):
        print(f"\n{'─'*80}")
        print(f"测试 {i}/{len(test_instructions)}")
        print(f"{'─'*80}")
        print(f"原始指令: {instruction}\n")
        
        # 调用分解器
        sub_instructions = decomposer.decompose(instruction)
        
        print(f"✅ 分解为 {len(sub_instructions)} 个子指令:\n")
        
        for sub in sub_instructions:
            print(f"[子指令 {sub['sub_id']}]")
            print(f"  ▸ 动作: {sub['sub_instruction']}")
            print(f"  ▸ 类型: {sub['action_type']}")
            print(f"  ▸ 地标: {sub['target_landmark']}")
            if sub.get('spatial_relation'):
                print(f"  ▸ 空间关系: {sub['spatial_relation']}")
            if sub.get('completion_condition'):
                print(f"  ▸ 完成条件: {sub['completion_condition']}")
            print()
    
    print("="*80)
    print("✅ 测试完成")
    print("="*80)


if __name__ == "__main__":
    test_decomposer()
