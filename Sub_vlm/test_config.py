#!/usr/bin/env python
"""
测试LLM配置和API连接
"""
import sys
import requests
from llm_config import LLMConfig


def test_config_load():
    """测试配置加载"""
    print("=" * 60)
    print("1. 测试配置文件加载")
    print("=" * 60)
    
    try:
        config = LLMConfig("llm_config.yaml")
        print("✅ 配置文件加载成功")
        print(f"   - API基础URL: {config.base_url}")
        print(f"   - 模型: {config.model}")
        print(f"   - 温度: {config.temperature}")
        print(f"   - 最大tokens: {config.max_tokens}")
        print(f"   - 超时: {config.timeout}秒")
        print(f"   - API密钥: {config.api_key[:20]}...")
        
        # 测试新增配置项
        if hasattr(config, 'enable_8_directions'):
            print(f"   - 8方向观察: {'启用' if config.enable_8_directions else '禁用'}")
            print(f"   - 罗盘视图: {'启用' if config.save_compass_view else '禁用'}")
            print(f"   - 自动验证: {'启用' if config.auto_verify else '禁用'}")
            print(f"   - 输出目录: {config.output_base_dir}")
        
        return config
    except FileNotFoundError as e:
        print(f"❌ 配置文件不存在: {e}")
        print("\n提示:")
        print("  运行 bash setup_llm_control.sh 创建配置文件")
        print("  或手动从 llm_config.yaml.template 复制")
        sys.exit(1)
    except ValueError as e:
        print(f"❌ 配置错误: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"❌ 加载失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


def test_api_connection(config):
    """测试API连接"""
    print("\n" + "=" * 60)
    print("2. 测试API连接")
    print("=" * 60)
    
    try:
        # 构建简单的测试请求
        payload = {
            "model": config.model,
            "messages": [
                {
                    "role": "user",
                    "content": "Hello! Please respond with 'API connection successful.'"
                }
            ],
            "max_tokens": 50,
            "temperature": 0.7
        }
        
        print(f"📡 正在连接 {config.base_url}...")
        print(f"   使用模型: {config.model}")
        
        response = requests.post(
            f"{config.base_url}/chat/completions",
            headers=config.get_headers(),
            json=payload,
            timeout=config.timeout
        )
        
        if response.status_code == 200:
            print("✅ API连接成功！")
            result = response.json()
            
            # 显示响应信息
            if 'choices' in result and len(result['choices']) > 0:
                message = result['choices'][0]['message']['content']
                print(f"   📨 LLM响应: {message}")
            
            # 显示使用统计
            if 'usage' in result:
                usage = result['usage']
                print(f"   📊 Token使用:")
                print(f"      - Prompt tokens: {usage.get('prompt_tokens', 0)}")
                print(f"      - Completion tokens: {usage.get('completion_tokens', 0)}")
                print(f"      - Total tokens: {usage.get('total_tokens', 0)}")
            
            return True
            
        elif response.status_code == 401:
            print("❌ 认证失败 (401)")
            print("   请检查API密钥是否正确")
            print(f"   当前密钥: {config.api_key[:20]}...")
            return False
            
        elif response.status_code == 429:
            print("❌ 请求过多 (429)")
            print("   请稍后再试或检查账户余额")
            return False
            
        else:
            print(f"❌ API请求失败 (状态码: {response.status_code})")
            print(f"   响应: {response.text[:200]}")
            return False
            
    except requests.exceptions.Timeout:
        print(f"❌ 请求超时 (>{config.timeout}秒)")
        print("   提示: 可以在配置文件中增加 timeout 值")
        return False
        
    except requests.exceptions.ConnectionError:
        print("❌ 网络连接失败")
        print("   请检查网络连接或防火墙设置")
        return False
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_vision_support(config):
    """测试视觉模型支持（可选）"""
    print("\n" + "=" * 60)
    print("3. 测试视觉模型支持")
    print("=" * 60)
    
    # 检查模型名称
    vision_models = [
        'claude-3',
        'gpt-4-vision',
        'gemini-pro-vision',
        'gpt-4o'
    ]
    
    has_vision = any(vm in config.model.lower() for vm in vision_models)
    
    if has_vision:
        print(f"✅ 当前模型支持视觉输入: {config.model}")
        print("   适用于导航任务的8方向图像分析")
    else:
        print(f"⚠️  当前模型可能不支持视觉输入: {config.model}")
        print("   推荐使用以下模型之一:")
        print("     - anthropic/claude-3-5-sonnet")
        print("     - openai/gpt-4-vision-preview")
        print("     - google/gemini-pro-vision")
    
    return has_vision


def main():
    """主测试流程"""
    print("\n🔧 LLM配置测试工具")
    print("━" * 60)
    
    # 测试1: 加载配置
    config = test_config_load()
    
    # 测试2: API连接
    api_ok = test_api_connection(config)
    
    # 测试3: 视觉支持
    vision_ok = test_vision_support(config)
    
    # 总结
    print("\n" + "=" * 60)
    print("📋 测试总结")
    print("=" * 60)
    print(f"配置加载: ✅")
    print(f"API连接: {'✅' if api_ok else '❌'}")
    print(f"视觉支持: {'✅' if vision_ok else '⚠️ '}")
    
    if api_ok and vision_ok:
        print("\n🎉 所有测试通过！系统已就绪")
        print("\n下一步:")
        print("  运行导航程序:")
        print("    python llm_manual_control.py \\")
        print("        ../VLN_CE/habitat_extensions/config/vlnce_task_enhanced.yaml")
        sys.exit(0)
    elif api_ok:
        print("\n⚠️  API连接正常，但建议使用支持视觉的模型")
        sys.exit(0)
    else:
        print("\n❌ 测试未通过，请检查配置和网络")
        sys.exit(1)


if __name__ == "__main__":
    main()
