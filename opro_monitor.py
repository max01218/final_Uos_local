#!/usr/bin/env python3
"""
OPRO性能监控脚本
监控优化效果、用户反馈、系统性能
"""

import json
import os
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import pandas as pd

def analyze_optimization_history():
    """分析优化历史"""
    history_file = "OPRO_Streamlined/prompts/optimization_history.json"
    
    if not os.path.exists(history_file):
        print("❌ 优化历史文件不存在")
        return
    
    with open(history_file, 'r', encoding='utf-8') as f:
        history = json.load(f)
    
    print("📊 OPRO优化历史分析")
    print("=" * 40)
    
    if 'optimization_runs' in history:
        runs = history['optimization_runs']
        print(f"总优化轮次: {len(runs)}")
        
        for i, run in enumerate(runs, 1):
            print(f"\n轮次 {i}:")
            print(f"  时间: {run.get('timestamp', 'N/A')}")
            print(f"  最终评分: {run.get('final_score', 'N/A')}")
            print(f"  改进幅度: {run.get('improvement', 'N/A')}")
            print(f"  迭代次数: {run.get('iterations', 'N/A')}")

def analyze_user_feedback():
    """分析用户反馈趋势"""
    interactions_file = "interactions.json"
    
    if not os.path.exists(interactions_file):
        print("❌ 交互文件不存在")
        return
    
    with open(interactions_file, 'r', encoding='utf-8') as f:
        interactions = json.load(f)
    
    print("\n📈 用户反馈分析")
    print("=" * 40)
    
    total = len(interactions)
    with_feedback = [i for i in interactions if i.get('user_feedback')]
    
    print(f"总交互数: {total}")
    print(f"有反馈数: {len(with_feedback)}")
    print(f"反馈率: {len(with_feedback)/total*100:.1f}%" if total > 0 else "无数据")
    
    if with_feedback:
        # 计算平均评分
        scores = []
        for interaction in with_feedback:
            feedback = interaction.get('user_feedback', {})
            if isinstance(feedback, dict):
                # 假设反馈包含满意度评分
                satisfaction = feedback.get('satisfaction')
                if satisfaction is not None:
                    scores.append(satisfaction)
        
        if scores:
            avg_score = sum(scores) / len(scores)
            print(f"平均满意度: {avg_score:.2f}")
            print(f"最高评分: {max(scores)}")
            print(f"最低评分: {min(scores)}")

def check_system_performance():
    """检查系统性能指标"""
    print("\n⚡ 系统性能检查")
    print("=" * 40)
    
    # 检查提示词文件
    prompt_file = "OPRO_Streamlined/prompts/optimized_prompt.txt"
    if os.path.exists(prompt_file):
        with open(prompt_file, 'r', encoding='utf-8') as f:
            content = f.read()
        print(f"当前提示词长度: {len(content)} 字符")
        
        # 分析提示词特征
        keywords = ['empathetic', 'professional', 'safety', 'guidelines', 'crisis']
        found_keywords = [kw for kw in keywords if kw.lower() in content.lower()]
        print(f"关键词覆盖: {len(found_keywords)}/{len(keywords)} ({', '.join(found_keywords)})")
    
    # 检查文件状态
    files_to_check = {
        "interactions.json": "用户交互记录",
        "opro_scheduler.log": "调度器日志",
        "opro_scheduler_state.json": "调度器状态"
    }
    
    for file_path, description in files_to_check.items():
        if os.path.exists(file_path):
            size = os.path.getsize(file_path)
            mtime = os.path.getmtime(file_path)
            last_modified = datetime.fromtimestamp(mtime)
            age = datetime.now() - last_modified
            
            print(f"{description}: {size} bytes (更新于 {age.days} 天前)")
        else:
            print(f"{description}: 文件不存在")

def generate_opro_report():
    """生成完整的OPRO报告"""
    print("🧠 OPRO系统完整报告")
    print("=" * 50)
    print(f"报告生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # 运行所有分析
    analyze_optimization_history()
    analyze_user_feedback() 
    check_system_performance()
    
    # 提供建议
    print("\n💡 优化建议")
    print("=" * 40)
    
    # 检查交互数量
    if os.path.exists("interactions.json"):
        with open("interactions.json", 'r') as f:
            interactions = json.load(f)
        
        if len(interactions) < 50:
            print("📈 建议: 收集更多用户交互 (当前: {}, 建议: ≥50)".format(len(interactions)))
        elif len(interactions) > 100:
            print("🚀 建议: 数据充足，可以运行完整OPRO优化")
    
    # 检查调度器状态
    if not os.path.exists("opro_scheduler.log"):
        print("🔄 建议: 启动自动调度器 (python auto_opro_scheduler.py)")
    
    # 检查最近优化时间
    history_file = "OPRO_Streamlined/prompts/optimization_history.json"
    if os.path.exists(history_file):
        mtime = os.path.getmtime(history_file)
        last_opt = datetime.fromtimestamp(mtime)
        age = datetime.now() - last_opt
        
        if age.days > 7:
            print(f"⏰ 建议: 上次优化已过 {age.days} 天，考虑重新优化")

def quick_status():
    """快速状态检查"""
    print("🔍 OPRO快速状态")
    print("=" * 30)
    
    # 关键文件检查
    key_files = [
        "OPRO_Streamlined/prompts/optimized_prompt.txt",
        "interactions.json",
        "opro_scheduler_state.json"
    ]
    
    all_good = True
    for file_path in key_files:
        if os.path.exists(file_path):
            print(f"✅ {file_path}")
        else:
            print(f"❌ {file_path}")
            all_good = False
    
    if all_good:
        print("\n🎉 系统状态良好！")
    else:
        print("\n⚠️  系统需要配置")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        if sys.argv[1] == "report":
            generate_opro_report()
        elif sys.argv[1] == "quick":
            quick_status()
        elif sys.argv[1] == "feedback":
            analyze_user_feedback()
        elif sys.argv[1] == "history":
            analyze_optimization_history()
        else:
            print("用法: python opro_monitor.py [report|quick|feedback|history]")
    else:
        quick_status() 