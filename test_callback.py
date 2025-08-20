#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试回调机制的脚本
"""

import requests
import json
import os
import time
from datetime import datetime

# 配置
WORKSTATION_URL = "http://localhost:7000/run-algorithm/"
STATUS_URL = "http://localhost:7000/status/{}"
CONF_FILE_PATH = "dummy_config.conf"
OUTPUT_PATH = f"test_callback_{int(time.time())}"

def create_dummy_conf_file():
    """创建一个虚拟的配置文件"""
    conf_content = """[Settings]
param1=value1
param2=value2
param3=value3
"""
    with open(CONF_FILE_PATH, 'w') as f:
        f.write(conf_content)
    print(f"Created dummy config file: {CONF_FILE_PATH}")

def test_algorithm_with_default_callback():
    """测试使用默认回调URL的算法执行"""
    print("\n=== 测试默认回调URL ===")
    
    # 准备请求参数（不指定callback_url，使用默认值）
    run_params = {
        "output_path": OUTPUT_PATH,
        "env": "base",
        "cuda_devices": "0"
        # 注意：没有指定callback_url，应该使用默认值
    }
    
    # 准备文件
    files = {
        'conf_file': (CONF_FILE_PATH, open(CONF_FILE_PATH, 'rb'), 'application/octet-stream')
    }
    
    data = {
        'run_params': json.dumps(run_params)
    }
    
    print(f"Sending request to workstation server...")
    print(f"Output path: {OUTPUT_PATH}")
    print(f"Using default callback URL (should be http://localhost:8111/api/callback)")
    
    try:
        response = requests.post(WORKSTATION_URL, files=files, data=data, timeout=30)
        response.raise_for_status()
        
        print("Request successful!")
        print(f"Server response: {response.json()}")
        
        # 关闭文件
        files['conf_file'][1].close()
        
        return True
        
    except requests.RequestException as e:
        print(f"Request failed: {e}")
        if hasattr(e, 'response') and e.response is not None:
            print(f"Response status: {e.response.status_code}")
            print(f"Response text: {e.response.text}")
        return False
    except Exception as e:
        print(f"Unexpected error: {e}")
        return False

def test_status_polling():
    """测试状态轮询"""
    print("\n=== 测试状态轮询 ===")
    
    max_polls = 20  # 最多轮询20次
    poll_interval = 3  # 每3秒轮询一次
    
    for i in range(max_polls):
        try:
            status_url = STATUS_URL.format(OUTPUT_PATH)
            response = requests.get(status_url, timeout=10)
            response.raise_for_status()
            
            status_data = response.json()
            print(f"[{datetime.now().strftime('%H:%M:%S')}] Poll {i+1}: {status_data}")
            
            if status_data.get('status') == 'completed':
                print("✅ 任务已完成！")
                return True
            elif status_data.get('status') == 'not_found':
                print("❌ 任务未找到")
                return False
            
            time.sleep(poll_interval)
            
        except requests.RequestException as e:
            print(f"Status request failed: {e}")
            time.sleep(poll_interval)
    
    print("⏰ 轮询超时，任务可能仍在运行")
    return False

def check_output_files():
    """检查输出文件"""
    print("\n=== 检查输出文件 ===")
    
    if not os.path.exists(OUTPUT_PATH):
        print(f"❌ 输出目录不存在: {OUTPUT_PATH}")
        return False
    
    files = os.listdir(OUTPUT_PATH)
    print(f"📁 输出目录 {OUTPUT_PATH} 中的文件:")
    
    for file in files:
        file_path = os.path.join(OUTPUT_PATH, file)
        file_size = os.path.getsize(file_path)
        print(f"  - {file} ({file_size} bytes)")
    
    # 检查是否有final_status.log
    final_log = os.path.join(OUTPUT_PATH, "final_status.log")
    if os.path.exists(final_log):
        print("✅ 找到 final_status.log 文件")
        with open(final_log, 'r', encoding='utf-8') as f:
            content = f.read()
            print(f"📄 final_status.log 内容:\n{content}")
    else:
        print("❌ 未找到 final_status.log 文件")
    
    return len(files) > 0

def main():
    """主函数"""
    print("🚀 开始测试回调机制")
    print(f"测试时间: {datetime.now()}")
    
    # 创建配置文件
    create_dummy_conf_file()
    
    # 测试算法执行
    if test_algorithm_with_default_callback():
        # 测试状态轮询
        test_status_polling()
        
        # 检查输出文件
        check_output_files()
    
    # 清理
    if os.path.exists(CONF_FILE_PATH):
        os.remove(CONF_FILE_PATH)
        print(f"\n🧹 已清理配置文件: {CONF_FILE_PATH}")
    
    print("\n✨ 测试完成")

if __name__ == "__main__":
    main()