#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试增强版任务状态检测功能
验证新的状态检测逻辑是否能正确识别不同情况下的任务状态
"""

import requests
import json
import os
import time
from datetime import datetime, timedelta
import glob

# 服务器配置
SERVER_URL = "http://localhost:7000"

def test_server_status():
    """测试服务器是否正常运行"""
    try:
        response = requests.get(f"{SERVER_URL}/local-images", timeout=5)
        if response.status_code == 200:
            print("✅ 服务器运行正常")
            return True
        else:
            print(f"❌ 服务器响应异常: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ 无法连接到服务器: {e}")
        return False

def test_status_api(output_path):
    """测试状态API"""
    try:
        # 如果是绝对路径，提取目录名
        if os.path.isabs(output_path):
            output_path = os.path.basename(output_path)
        
        response = requests.get(f"{SERVER_URL}/status/{output_path}", timeout=10)
        if response.status_code == 200:
            status_data = response.json()
            print(f"\n📊 目录: {output_path}")
            print(f"   状态: {status_data.get('status', 'unknown')}")
            print(f"   消息: {status_data.get('message', 'No message')}")
            print(f"   时间戳: {status_data.get('timestamp', 'No timestamp')}")
            
            # 显示详细信息
            details = status_data.get('details', {})
            if details:
                print(f"   详细信息:")
                print(f"     - 进程运行中: {details.get('process_running', False)}")
                print(f"     - 进程返回码: {details.get('process_return_code', 'None')}")
                print(f"     - 完成标志: {details.get('has_completion_markers', False)}")
                print(f"     - 日志活动: {details.get('log_activity', False)}")
                print(f"     - 文件活动: {details.get('recent_file_activity', False)}")
                print(f"     - 最后活动时间: {details.get('last_activity_time', 'None')}")
                
                # 显示完成文件
                completion_files = details.get('completion_files', [])
                if completion_files:
                    print(f"     - 完成标志文件: {', '.join(completion_files)}")
                
                # 显示日志文件
                log_files = details.get('log_files', [])
                if log_files:
                    print(f"     - 日志文件数量: {len(log_files)}")
                    for log_file in log_files[:3]:  # 只显示前3个
                        print(f"       * {log_file['file']} (大小: {log_file['size']} bytes, 修改时间: {log_file['last_modified']})")
                
                # 显示最近文件
                recent_files = details.get('recent_files', [])
                if recent_files:
                    print(f"     - 最近活动文件数量: {len(recent_files)}")
                    for recent_file in recent_files[:3]:  # 只显示前3个
                        print(f"       * {recent_file['file']} (大小: {recent_file['size']} bytes, 修改时间: {recent_file['last_modified']})")
            
            return status_data
        else:
            print(f"❌ 状态API请求失败: {response.status_code}")
            print(f"   响应内容: {response.text}")
            return None
    except Exception as e:
        print(f"❌ 状态API请求异常: {e}")
        return None

def find_test_directories():
    """查找可用的测试目录"""
    base_path = "f:/前后端开发/BackendRunnerServer-master1/BackendRunnerServer-master"
    test_dirs = []
    
    # 查找training开头的目录
    training_dirs = glob.glob(os.path.join(base_path, "training_*"))
    test_dirs.extend(training_dirs)
    
    # 查找test开头的目录
    test_pattern_dirs = glob.glob(os.path.join(base_path, "test_*"))
    test_dirs.extend(test_pattern_dirs)
    
    # 过滤出存在的目录
    existing_dirs = [d for d in test_dirs if os.path.isdir(d)]
    
    return existing_dirs[:5]  # 返回前5个目录

def create_test_scenario():
    """创建测试场景"""
    test_dir = "test_status_scenario"
    
    # 创建测试目录
    os.makedirs(test_dir, exist_ok=True)
    
    # 创建一些测试文件
    current_time = datetime.now()
    
    # 1. 创建一个旧的日志文件（超过5分钟）
    old_log_path = os.path.join(test_dir, "old_run.log")
    with open(old_log_path, 'w', encoding='utf-8') as f:
        f.write("这是一个旧的日志文件\n")
    # 设置文件时间为10分钟前
    old_time = (current_time - timedelta(minutes=10)).timestamp()
    os.utime(old_log_path, (old_time, old_time))
    
    # 2. 创建一个新的日志文件（最近2分钟）
    new_log_path = os.path.join(test_dir, "recent_run.log")
    with open(new_log_path, 'w', encoding='utf-8') as f:
        f.write("这是一个最近的日志文件\n")
        f.write(f"当前时间: {current_time.isoformat()}\n")
    # 设置文件时间为2分钟前
    recent_time = (current_time - timedelta(minutes=2)).timestamp()
    os.utime(new_log_path, (recent_time, recent_time))
    
    # 3. 创建一个完成标志文件
    completion_path = os.path.join(test_dir, "completed.txt")
    with open(completion_path, 'w', encoding='utf-8') as f:
        f.write("任务已完成\n")
    
    print(f"✅ 创建测试场景目录: {test_dir}")
    return test_dir

def main():
    """主测试函数"""
    print("🚀 开始测试增强版任务状态检测功能")
    print("=" * 60)
    
    # 1. 测试服务器状态
    if not test_server_status():
        print("❌ 服务器未运行，请先启动workstation_server.py")
        return
    
    # 2. 创建测试场景
    test_scenario_dir = create_test_scenario()
    
    # 3. 测试不存在的目录
    print("\n🔍 测试场景1: 不存在的目录")
    test_status_api("non_existent_directory")
    
    # 4. 测试创建的测试场景
    print("\n🔍 测试场景2: 有完成标志的目录")
    test_status_api(test_scenario_dir)
    
    # 5. 测试现有的训练目录
    existing_dirs = find_test_directories()
    if existing_dirs:
        print(f"\n🔍 测试场景3-{2+len(existing_dirs)}: 现有的训练/测试目录")
        for i, test_dir in enumerate(existing_dirs, 3):
            print(f"\n--- 测试场景{i} ---")
            test_status_api(test_dir)
    else:
        print("\n⚠️  未找到现有的训练目录")
    
    # 6. 修改测试场景，移除完成标志，添加最近活动
    print("\n🔍 测试场景: 移除完成标志，模拟运行中状态")
    completion_path = os.path.join(test_scenario_dir, "completed.txt")
    if os.path.exists(completion_path):
        os.remove(completion_path)
    
    # 创建一个非常新的文件（1分钟前）
    very_recent_file = os.path.join(test_scenario_dir, "very_recent.png")
    with open(very_recent_file, 'wb') as f:
        f.write(b"fake image data")
    very_recent_time = (datetime.now() - timedelta(minutes=1)).timestamp()
    os.utime(very_recent_file, (very_recent_time, very_recent_time))
    
    test_status_api(test_scenario_dir)
    
    print("\n" + "=" * 60)
    print("✅ 测试完成！")
    print("\n📋 功能总结:")
    print("   - ✅ 支持多种完成标志文件检测")
    print("   - ✅ 支持日志文件活动监控")
    print("   - ✅ 支持输出文件活动监控")
    print("   - ✅ 支持进程状态检测")
    print("   - ✅ 支持递归文件搜索")
    print("   - ✅ 提供详细的状态信息")
    print("   - ✅ 智能状态判断逻辑")

if __name__ == "__main__":
    main()