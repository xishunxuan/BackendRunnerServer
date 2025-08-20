#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
基于优化进度监控的任务完成状态检查器
通过定时监控debug.log中的优化进度更新来判断任务是否完成
当优化进度在指定时间内不再更新时，返回任务完成状态
"""

import os
import re
import time
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Callable
from dataclasses import dataclass

@dataclass
class ProgressInfo:
    """优化进度信息"""
    timestamp: datetime
    round_num: int
    epoch: Optional[int] = None
    loss: Optional[float] = None
    message: str = ""

class ProgressMonitor:
    """优化进度监控器"""
    
    def __init__(self, 
                 log_file_path: str = "debug.log",
                 check_interval: int = 30,  # 检查间隔（秒）
                 timeout_minutes: int = 10,  # 超时时间（分钟）
                 callback: Optional[Callable[[str, Dict], None]] = None):
        """
        初始化进度监控器
        
        Args:
            log_file_path: 日志文件路径
            check_interval: 检查间隔（秒）
            timeout_minutes: 无进度更新的超时时间（分钟）
            callback: 状态变化回调函数
        """
        self.log_file_path = log_file_path
        self.check_interval = check_interval
        self.timeout_minutes = timeout_minutes
        self.callback = callback
        
        self.is_monitoring = False
        self.monitor_thread = None
        self.last_progress_time = None
        self.progress_history: List[ProgressInfo] = []
        self.current_status = "unknown"
        
        # 优化进度匹配模式
        self.progress_patterns = [
            r'第(\d+)轮',  # 中文轮次
            r'round\s+(\d+)',  # 英文轮次
            r'epoch\s+(\d+).*?loss[:\s]+(\d+\.\d+)',  # epoch和loss
            r'optimization\s+round\s+(\d+)',  # 优化轮次
            r'训练.*?第(\d+)轮',  # 训练轮次
        ]
        
        # 完成标志模式
        self.completion_patterns = [
            r'optimization\s+completed',
            r'training\s+finished',
            r'优化完成',
            r'训练完成',
            r'任务完成',
            r'finished',
            r'completed',
            r'done'
        ]
        
        # 错误标志模式
        self.error_patterns = [
            r'ERROR',
            r'Exception',
            r'Traceback',
            r'错误',
            r'失败',
            r'failed'
        ]
    
    def _detect_encoding(self, file_path: str) -> str:
        """检测文件编码"""
        encodings = ['utf-8', 'gbk', 'gb2312', 'cp936', 'latin1']
        
        for encoding in encodings:
            try:
                with open(file_path, 'r', encoding=encoding) as f:
                    f.read(1024)  # 读取前1024字符测试
                return encoding
            except (UnicodeDecodeError, UnicodeError):
                continue
        
        return 'utf-8'  # 默认返回utf-8
    
    def _parse_log_line(self, line: str) -> Optional[ProgressInfo]:
        """解析日志行，提取进度信息"""
        # 提取时间戳
        timestamp_match = re.search(r'(\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2})', line)
        if not timestamp_match:
            return None
        
        try:
            timestamp = datetime.strptime(timestamp_match.group(1), '%Y-%m-%d %H:%M:%S')
        except ValueError:
            return None
        
        # 检查是否包含进度信息
        for pattern in self.progress_patterns:
            match = re.search(pattern, line, re.IGNORECASE)
            if match:
                groups = match.groups()
                if len(groups) >= 1:
                    try:
                        round_num = int(groups[0])
                        epoch = None
                        loss = None
                        
                        # 如果有epoch和loss信息
                        if len(groups) >= 2:
                            try:
                                if 'epoch' in pattern.lower():
                                    epoch = int(groups[0])
                                    loss = float(groups[1])
                                    round_num = epoch  # 使用epoch作为轮次
                            except (ValueError, IndexError):
                                pass
                        
                        return ProgressInfo(
                            timestamp=timestamp,
                            round_num=round_num,
                            epoch=epoch,
                            loss=loss,
                            message=line.strip()
                        )
                    except (ValueError, IndexError):
                        continue
        
        return None
    
    def _check_completion_markers(self, content: str) -> bool:
        """检查完成标志"""
        for pattern in self.completion_patterns:
            if re.search(pattern, content, re.IGNORECASE):
                return True
        return False
    
    def _check_error_markers(self, content: str) -> bool:
        """检查错误标志"""
        for pattern in self.error_patterns:
            if re.search(pattern, content, re.IGNORECASE):
                return True
        return False
    
    def _analyze_current_status(self) -> Tuple[str, Dict]:
        """分析当前状态"""
        if not os.path.exists(self.log_file_path):
            return "unknown", {"reason": "日志文件不存在"}
        
        try:
            encoding = self._detect_encoding(self.log_file_path)
            with open(self.log_file_path, 'r', encoding=encoding, errors='ignore') as f:
                content = f.read()
        except Exception as e:
            return "error", {"reason": f"读取日志文件失败: {str(e)}"}
        
        # 检查明确的完成标志
        if self._check_completion_markers(content):
            return "completed", {"reason": "发现完成标志"}
        
        # 检查错误标志
        if self._check_error_markers(content):
            # 如果有错误但也有最近的进度，可能还在运行
            if self.last_progress_time and \
               datetime.now() - self.last_progress_time < timedelta(minutes=self.timeout_minutes):
                return "running", {"reason": "有错误但进度仍在更新"}
            else:
                return "failed", {"reason": "发现错误且无最近进度更新"}
        
        # 基于进度更新时间判断
        if self.last_progress_time:
            time_since_last = datetime.now() - self.last_progress_time
            if time_since_last < timedelta(minutes=self.timeout_minutes):
                return "running", {
                    "reason": f"最后进度更新: {time_since_last.total_seconds():.0f}秒前",
                    "last_progress": self.last_progress_time.strftime('%Y-%m-%d %H:%M:%S')
                }
            else:
                return "completed", {
                    "reason": f"进度已停止更新 {time_since_last.total_seconds()/60:.1f} 分钟",
                    "last_progress": self.last_progress_time.strftime('%Y-%m-%d %H:%M:%S')
                }
        
        return "unknown", {"reason": "未发现进度信息"}
    
    def _scan_log_for_progress(self):
        """扫描日志文件获取最新进度"""
        if not os.path.exists(self.log_file_path):
            return
        
        try:
            encoding = self._detect_encoding(self.log_file_path)
            with open(self.log_file_path, 'r', encoding=encoding, errors='ignore') as f:
                lines = f.readlines()
        except Exception:
            return
        
        # 从后往前扫描，获取最新的进度信息
        latest_progress = None
        for line in reversed(lines[-1000:]):  # 只检查最后1000行
            progress_info = self._parse_log_line(line)
            if progress_info:
                if not latest_progress or progress_info.timestamp > latest_progress.timestamp:
                    latest_progress = progress_info
        
        if latest_progress:
            # 更新进度历史
            if not self.progress_history or \
               latest_progress.timestamp > self.progress_history[-1].timestamp:
                self.progress_history.append(latest_progress)
                self.last_progress_time = latest_progress.timestamp
                
                # 保持历史记录不超过100条
                if len(self.progress_history) > 100:
                    self.progress_history = self.progress_history[-100:]
    
    def _monitor_loop(self):
        """监控循环"""
        print(f"🔄 开始监控优化进度 (检查间隔: {self.check_interval}秒, 超时: {self.timeout_minutes}分钟)")
        
        while self.is_monitoring:
            try:
                # 扫描日志获取最新进度
                self._scan_log_for_progress()
                
                # 分析当前状态
                new_status, status_info = self._analyze_current_status()
                
                # 如果状态发生变化，调用回调函数
                if new_status != self.current_status:
                    old_status = self.current_status
                    self.current_status = new_status
                    
                    print(f"📊 状态变化: {old_status} -> {new_status}")
                    print(f"   原因: {status_info.get('reason', '未知')}")
                    
                    if self.callback:
                        self.callback(new_status, status_info)
                    
                    # 如果任务完成或失败，停止监控
                    if new_status in ['completed', 'failed']:
                        print(f"✅ 监控结束: 任务状态为 {new_status}")
                        break
                
                # 等待下次检查
                time.sleep(self.check_interval)
                
            except Exception as e:
                print(f"❌ 监控过程中出现错误: {str(e)}")
                time.sleep(self.check_interval)
        
        self.is_monitoring = False
        print("🛑 进度监控已停止")
    
    def start_monitoring(self):
        """开始监控"""
        if self.is_monitoring:
            print("⚠️ 监控已在运行中")
            return
        
        self.is_monitoring = True
        self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.monitor_thread.start()
    
    def stop_monitoring(self):
        """停止监控"""
        self.is_monitoring = False
        if self.monitor_thread and self.monitor_thread.is_alive():
            self.monitor_thread.join(timeout=5)
    
    def get_current_status(self) -> Tuple[str, Dict]:
        """获取当前状态"""
        return self._analyze_current_status()
    
    def get_progress_summary(self) -> Dict:
        """获取进度摘要"""
        if not self.progress_history:
            return {"total_rounds": 0, "latest_progress": None}
        
        latest = self.progress_history[-1]
        return {
            "total_rounds": len(self.progress_history),
            "latest_progress": {
                "timestamp": latest.timestamp.strftime('%Y-%m-%d %H:%M:%S'),
                "round_num": latest.round_num,
                "epoch": latest.epoch,
                "loss": latest.loss,
                "message": latest.message
            },
            "first_progress": {
                "timestamp": self.progress_history[0].timestamp.strftime('%Y-%m-%d %H:%M:%S'),
                "round_num": self.progress_history[0].round_num
            } if self.progress_history else None
        }

def demo_progress_monitor():
    """演示进度监控功能"""
    print("🚀 优化进度监控器演示")
    print("通过定时监控debug.log中的优化进度来判断任务完成状态")
    print("=" * 60)
    
    def status_callback(status: str, info: Dict):
        """状态变化回调函数"""
        print(f"\n📢 任务状态更新: {status}")
        print(f"   详情: {info}")
        if status in ['completed', 'failed']:
            print(f"\n🎯 最终状态: {status.upper()}")
    
    # 创建监控器
    monitor = ProgressMonitor(
        log_file_path="debug.log",
        check_interval=10,  # 10秒检查一次
        timeout_minutes=5,  # 5分钟无更新则认为完成
        callback=status_callback
    )
    
    # 获取初始状态
    status, info = monitor.get_current_status()
    print(f"📊 初始状态: {status}")
    print(f"   详情: {info}")
    
    # 获取进度摘要
    summary = monitor.get_progress_summary()
    print(f"\n📈 进度摘要:")
    print(f"   总轮次: {summary['total_rounds']}")
    if summary['latest_progress']:
        latest = summary['latest_progress']
        print(f"   最新进度: 第{latest['round_num']}轮 ({latest['timestamp']})")
        if latest['epoch'] and latest['loss']:
            print(f"   训练信息: Epoch {latest['epoch']}, Loss {latest['loss']}")
    
    print(f"\n🔄 开始实时监控 (按Ctrl+C停止)...")
    
    try:
        # 开始监控
        monitor.start_monitoring()
        
        # 主线程等待
        while monitor.is_monitoring:
            time.sleep(1)
            
    except KeyboardInterrupt:
        print("\n⏹️ 用户中断监控")
    finally:
        monitor.stop_monitoring()
        
        # 显示最终状态
        final_status, final_info = monitor.get_current_status()
        print(f"\n📊 最终状态: {final_status}")
        print(f"   详情: {final_info}")

if __name__ == "__main__":
    demo_progress_monitor()