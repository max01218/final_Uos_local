#!/usr/bin/env python3
"""
OPRO系统使用示例
演示不同的调用方法和实际应用场景
"""

import os
import subprocess
import time
from pathlib import Path

def run_command(cmd, description):
    """运行命令并显示结果"""
    print(f"\n{'='*60}")
    print(f"🔥 {description}")
    print(f"💻 命令: {cmd}")
    print(f"{'='*60}")
    
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    
    if result.returncode == 0:
        print(f"✅ 成功执行")
        if result.stdout:
            print(f"📤 输出:\n{result.stdout}")
    else:
        print(f"❌ 执行失败")
        if result.stderr:
            print(f"📤 错误:\n{result.stderr}")
    
    return result.returncode == 0

def main():
    """主要使用示例"""
    
    print("🧠 OPRO系统使用示例")
    print("=" * 60)
    
    # 示例1: 快速系统检查
    print("\n📋 示例1: 系统状态检查")
    run_command(
        "cd OPRO_Streamlined && python run_opro.py --mode info",
        "检查OPRO系统状态和GPU配置"
    )
    
    # 示例2: GPU状态检查
    print("\n📋 示例2: GPU性能检查") 
    run_command(
        "cd OPRO_Streamlined && python run_opro.py --gpu-check",
        "检查GPU可用性和推荐模型配置"
    )
    
    # 示例3: 快速测试优化
    print("\n📋 示例3: 快速测试优化 (推荐)")
    run_command(
        "cd OPRO_Streamlined && python run_opro.py --mode test-run",
        "运行2轮快速测试，验证系统工作状态"
    )
    
    # 示例4: 评估当前提示词
    print("\n📋 示例4: 评估当前提示词")
    run_command(
        "cd OPRO_Streamlined && python run_opro.py --mode evaluate --prompt-file prompts/optimized_prompt.txt",
        "评估当前使用的优化提示词性能"
    )
    
    # 示例5: 测试LLM模型
    print("\n📋 示例5: 测试LLM模型连接")
    run_command(
        "cd OPRO_Streamlined && python run_opro.py --mode test-model",
        "测试本地LLM模型是否可以正常加载和推理"
    )
    
    print("\n🎯 高级用法提示:")
    print("   完整优化: cd OPRO_Streamlined && python run_opro.py --mode optimize")
    print("   自动调度: python auto_opro_scheduler.py")
    print("   原始版本: cd ICD11_OPRO && python run_opro.py")
    
    print("\n📊 优化结果查看:")
    print("   当前提示词: OPRO_Streamlined/prompts/optimized_prompt.txt")  
    print("   优化历史: OPRO_Streamlined/prompts/optimization_history.json")
    print("   用户反馈: interactions.json")

def show_optimization_workflow():
    """展示完整的优化工作流程"""
    
    workflow_steps = [
        "1. 🔍 系统检查: python run_opro.py --mode info",
        "2. 🚀 GPU检查: python run_opro.py --gpu-check", 
        "3. 🧪 快速测试: python run_opro.py --mode test-run",
        "4. 📊 性能评估: python run_opro.py --mode evaluate --prompt-file prompts/optimized_prompt.txt",
        "5. 🏆 完整优化: python run_opro.py --mode optimize",
        "6. 🔄 自动调度: python auto_opro_scheduler.py (后台运行)"
    ]
    
    print("\n🔥 OPRO优化完整工作流程:")
    print("=" * 50)
    
    for step in workflow_steps:
        print(f"   {step}")
    
    print("\n💡 建议:")
    print("   • 首次使用先运行步骤1-3验证系统")
    print("   • 定期运行步骤4检查提示词性能")
    print("   • 收集足够反馈后运行步骤5优化")
    print("   • 生产环境启用步骤6自动调度")

def check_opro_status():
    """检查OPRO当前状态"""
    
    print("\n🔍 OPRO系统状态检查:")
    print("=" * 40)
    
    # 检查关键文件
    key_files = {
        "OPRO_Streamlined/prompts/optimized_prompt.txt": "当前优化提示词",
        "OPRO_Streamlined/prompts/optimization_history.json": "优化历史记录",
        "OPRO_Streamlined/config/config.json": "OPRO配置文件",
        "interactions.json": "用户交互反馈",
        "opro_scheduler_state.json": "调度器状态",
        "opro_scheduler.log": "调度器日志"
    }
    
    for file_path, description in key_files.items():
        if os.path.exists(file_path):
            file_size = os.path.getsize(file_path)
            mod_time = os.path.getmtime(file_path)
            mod_date = time.strftime("%Y-%m-%d %H:%M", time.localtime(mod_time))
            print(f"✅ {description}")
            print(f"   文件: {file_path}")
            print(f"   大小: {file_size} bytes, 修改: {mod_date}")
        else:
            print(f"❌ {description}")
            print(f"   文件: {file_path} (不存在)")
    
    # 检查当前提示词内容
    prompt_file = "OPRO_Streamlined/prompts/optimized_prompt.txt"
    if os.path.exists(prompt_file):
        with open(prompt_file, 'r', encoding='utf-8') as f:
            content = f.read()
        print(f"\n📄 当前优化提示词预览:")
        print(f"   长度: {len(content)} 字符")
        print(f"   内容: {content[:200]}...")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        if sys.argv[1] == "examples":
            main()
        elif sys.argv[1] == "workflow":
            show_optimization_workflow()
        elif sys.argv[1] == "status":
            check_opro_status()
        else:
            print("用法: python opro_usage_examples.py [examples|workflow|status]")
    else:
        print("🧠 OPRO使用指南")
        print("=" * 30)
        print("python opro_usage_examples.py examples  - 运行使用示例")
        print("python opro_usage_examples.py workflow  - 显示优化工作流程") 
        print("python opro_usage_examples.py status    - 检查系统状态")
        
        # 默认显示工作流程
        show_optimization_workflow()
        check_opro_status() 