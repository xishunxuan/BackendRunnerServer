import os
import json
import subprocess
import threading
import platform
import time
from datetime import datetime, timedelta
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import uvicorn
import requests
import shutil
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
import glob
from typing import List
import pathlib
from SerializeTool import deserialize_from_file_with_types, serialize_to_file_with_types
from ConfClass import OPTIMIZER_CONF, GLOBAL_CONF, TRAINER_CONF, TEST_SURROGATE_MODULE_CONF
from progress_monitor import ProgressMonitor
app = FastAPI()

# 全局变量来跟踪进程和监控状态
processes = {}
monitors = {}

class ConfigManager:
    """配置管理器，用于处理config.conf文件的转义和验证"""
    def __init__(self):
        self.args = None
        self.optim_conf = None
        self.global_conf = None
        self.trainer_conf = None
        self.test_surrogate_module_conf = None

    def init_conf(self, args=None, target_output_path=None):
        """使用反序列化的args初始化所有配置"""
        self.args = args
        self.optim_conf = OPTIMIZER_CONF(args)
        self.global_conf = GLOBAL_CONF(args)
        self.trainer_conf = TRAINER_CONF(args)
        self.test_surrogate_module_conf = TEST_SURROGATE_MODULE_CONF(args)
        
        # 序列化输出 - 使用传入的目标路径或args中的路径
        if target_output_path:
            output_path = pathlib.Path(target_output_path) / "OptimizerMainConfig.conf"
        else:
            output_path = pathlib.Path(args.output_path) / "OptimizerMainConfig.conf"
            
        serialize_to_file_with_types(output_path, args, 
                                   ignore_unsupported=True, 
                                   pretty_json=True)
        print(f"配置已序列化到: {output_path}")
        return output_path

def extract_form_from_config(config_file_path):
    """从配置文件中提取form部分"""
    try:
        import json
        with open(config_file_path, 'r', encoding='utf-8') as file:
            config_data = json.load(file)
        
        # 检查是否有form字段
        if 'form' in config_data:
            return config_data['form']
        else:
            # 如果没有form字段，返回整个配置
            return config_data
    except Exception as e:
        print(f"❌ 提取form部分失败: {str(e)}")
        raise e

def create_temp_form_file(form_data, output_path):
    """创建临时的form配置文件"""
    import json
    import os
    
    temp_file_path = os.path.join(output_path, "temp_form_config.json")
    
    with open(temp_file_path, 'w', encoding='utf-8') as file:
        json.dump(form_data, file, ensure_ascii=False, indent=2)
    
    return temp_file_path

def process_config_file(config_file_path, output_path):
    """处理配置文件，进行转义和验证"""
    try:
        print(f"正在处理配置文件: {config_file_path}")
        
        # 1. 提取form部分
        form_data = extract_form_from_config(config_file_path)
        print("✅ 成功提取form部分！")
        
        # 2. 创建临时form文件
        temp_form_file = create_temp_form_file(form_data, output_path)
        print(f"✅ 创建临时form文件: {temp_form_file}")
        
        # 3. 反序列化form配置
        args = deserialize_from_file_with_types(temp_form_file)
        print("✅ 配置文件反序列化成功！")
        
        # 4. 检查反序列化后的对象
        print(f"对象类型: {type(args)}")
        
        # 5. 初始化所有配置
        print("正在初始化配置...")
        config_manager = ConfigManager()
        processed_config_path = config_manager.init_conf(args, target_output_path=output_path)
        print("✅ 配置处理完成！")
        
        # 6. 清理临时文件
        try:
            os.remove(temp_form_file)
            print("✅ 清理临时文件完成")
        except:
            pass
        
        return str(processed_config_path)
        
    except FileNotFoundError:
        print(f"❌ 错误: 配置文件不存在 - {config_file_path}")
        raise HTTPException(status_code=400, detail=f"配置文件不存在: {config_file_path}")
    except Exception as e:
        print(f"❌ 配置文件处理失败: {str(e)}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"配置文件处理失败: {str(e)}")

class RunConfig(BaseModel):
    output_path: str
    conf_content: str
    server_ip: str
    server_port: int


class ImageListRequest(BaseModel):
    output_path: str


class ImageDownloadRequest(BaseModel):
    output_path: str
    image_path: str
import subprocess, os


def run_command_detached(cmd_list, work_dir=None, logfile_name="run_.log", env=None):
    # Windows flags
    DETACHED_PROCESS = 0x00000008
    CREATE_NEW_PROCESS_GROUP = 0x00000200
    flags = DETACHED_PROCESS | CREATE_NEW_PROCESS_GROUP

    work_dir = work_dir or os.getcwd()
    logfile = os.path.join(work_dir, logfile_name)
    # 以追加模式打开，避免 PIPE 堵塞
    fout = open(logfile, "a", encoding="utf-8", errors="ignore")
    # 注意：shell=False，cmd_list 必须是列表
    process = subprocess.Popen(
        cmd_list,
        cwd=work_dir,
        stdout=fout,
        stderr=fout,
        shell=False,
        creationflags=flags,
        close_fds=True,
        env=env  # <--- 在这里添加 env 参数
    )
    # 不要在这里 close(fout) —— 子进程还在写。可以把 fout 保存在全局字典以便 later close（可选）。
    return process, logfile

def run_command(command, output_path, callback_url, work_dir=None, shell=True):
    """
    在子进程中运行命令，并返回进程对象
    """
    print(f"Executing command: {command}")
    # 使用 shell=True 来正确处理复杂的 shell 命令
    process = subprocess.Popen(
        command,
        shell=shell,
        cwd=work_dir,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    processes[output_path] = process
    return process

def find_python_executable(env_name_or_path):
    """
    尝试找到指定 conda env 对应的 python.exe。
    env_name_or_path 可以是：
    - 直接的 python.exe 的绝对路径 -> 直接返回
    - conda 环境目录路径 -> 在该目录下查找 python.exe
    - conda 环境名称（如 pytorch112_py311） -> 在常见 conda 根目录下查找
    如果找不到，返回 None。
    """
    # 1) 如果传入的是明确的 python.exe 路径
    if env_name_or_path and os.path.isfile(env_name_or_path) and env_name_or_path.lower().endswith("python.exe"):
        return env_name_or_path

    # 2) 如果是目录，判断目录下是否有 python.exe
    if env_name_or_path and os.path.isdir(env_name_or_path):
        candidate = os.path.join(env_name_or_path, "python.exe")
        if os.path.isfile(candidate):
            return candidate

    # 3) 如果看起来是环境名，尝试几处常见 conda 安装位置
    userprofile = os.environ.get("USERPROFILE", "")
    candidates = []
    if userprofile:
        candidates += [
            os.path.join(userprofile, "anaconda3", "envs", env_name_or_path, "python.exe"),
            os.path.join(userprofile, "Miniconda3", "envs", env_name_or_path, "python.exe"),
            os.path.join(userprofile, ".conda", "envs", env_name_or_path, "python.exe"),
        ]
    # 4) 也尝试系统 PATH 上可能的 conda envs 路径（若 CONDA_PREFIX 提供）
    conda_prefix = os.environ.get("CONDA_PREFIX")
    if conda_prefix:
        candidates.append(os.path.join(conda_prefix, "envs", env_name_or_path, "python.exe"))

    # 5) 检查 candidates
    for p in candidates:
        if p and os.path.isfile(p):
            return p

    # 6) 还可以尝试在 PATH 中查找 python（但这不保证是目标 env）
    which_py = shutil.which("python")
    if which_py:
        # 返回 None 而不是冒然使用系统 python；只有在极端回退场景才使用
        return None

    return None

def monitor_process(process, output_path, callback_url, logfile=None):
    # 等待进程结束
    process.wait()
    rc = process.returncode
    print(f"Process for {output_path} finished with return code {rc}.")

    # 停止监控文件夹
    if output_path in monitors:
        monitors[output_path].stop()
        monitors[output_path].join()
        del monitors[output_path]

    # 读取 run_.log（如果存在），并把片段包含到回调中，方便远程排查
    log_excerpt = ""
    try:
        if logfile and os.path.exists(logfile):
            with open(logfile, "r", encoding="utf-8", errors="ignore") as f:
                # 读取最后 20000 字符或最后若干行以避免太大
                f.seek(0, os.SEEK_END)
                size = f.tell()
                start = max(0, size - 20000)
                f.seek(start)
                log_excerpt = f.read()
    except Exception as e:
        print(f"Error reading logfile {logfile}: {e}")

    status_data = {
        "status": "completed",
        "output_path": output_path,
        "return_code": rc,
        "log_excerpt": log_excerpt[-20000:] if log_excerpt else ""
    }
    try:
        requests.post(callback_url, json=status_data, timeout=10)
    except requests.RequestException as e:
        print(f"Error sending final status: {e}")

    # 清理 process dict
    if output_path in processes:
        del processes[output_path]


class FileChangeHandler(FileSystemEventHandler):
    def __init__(self, output_path, callback_url):
        self.output_path = output_path
        self.callback_url = callback_url

    def on_any_event(self, event):
        if event.is_directory:
            return

        # 我们只关心创建和修改事件
        if event.event_type in ['created', 'modified']:
            file_path = event.src_path
            if file_path.endswith(('.jpg', '.png', '.log')):
                print(f"Detected change in {file_path}. Sending to backend.")
                self.send_file_to_backend(file_path)

    def send_file_to_backend(self, file_path):
        try:
            with open(file_path, 'rb') as f:
                files = {'file': (os.path.basename(file_path), f)}
                data = {'output_path': self.output_path}
                response = requests.post(self.callback_url, files=files, data=data, timeout=10)
                response.raise_for_status()
                print(f"Successfully sent {file_path} to {self.callback_url}")
        except requests.RequestException as e:
            print(f"Error sending file {file_path} to backend: {e}")
        except FileNotFoundError:
            print(f"File not found: {file_path}. It might have been deleted before sending.")
        except Exception as e:
            print(f"An unexpected error occurred while sending {file_path}: {e}")


def start_file_monitoring(output_path, callback_url):
    """
    开始监控指定输出目录的文件夹
    """
    event_handler = FileChangeHandler(output_path, callback_url)
    observer = Observer()
    observer.schedule(event_handler, output_path, recursive=True)
    observer.start()
    monitors[output_path] = observer
    print(f"Started monitoring {output_path}")


@app.post("/run-algorithm/")
async def run_algorithm(
    conf_file: UploadFile = File(...),
    run_params: str = Form(...)
):
    try:
        params = json.loads(run_params)
        env = params.get("env")
        output_path = params.get("output_path")
        cuda_devices = params.get("cuda_devices", "0") # 默认为 "0"
        callback_url = params.get("callback_url", "http://vivighr.vip:8111/api/callback") # 默认回调URL

        if not output_path:
             raise HTTPException(status_code=400, detail="output_path is required.")
        
        # 1. 保存 .conf 文件
        conf_content = await conf_file.read()
        
        # 创建输出目录，如果它不存在
        os.makedirs(output_path, exist_ok=True)

        # 检查并删除已存在的debug.log文件
        debug_log_path = os.path.join(output_path, "debug.log")
        if os.path.exists(debug_log_path):
            try:
                # 尝试修改文件权限（Windows系统）
                import stat
                os.chmod(debug_log_path, stat.S_IWRITE)
                os.remove(debug_log_path)
                print(f"✅ 已删除已存在的debug.log文件: {debug_log_path}")
            except PermissionError as e:
                print(f"⚠️ 权限不足，无法删除debug.log文件: {str(e)}")
                # 尝试重命名文件作为备份
                try:
                    import time
                    backup_name = f"debug_backup_{int(time.time())}.log"
                    backup_path = os.path.join(output_path, backup_name)
                    os.rename(debug_log_path, backup_path)
                    print(f"📁 已将debug.log重命名为备份文件: {backup_name}")
                except Exception as rename_e:
                    print(f"❌ 无法删除或重命名debug.log文件: {str(rename_e)}")
            except Exception as e:
                print(f"⚠️ 删除debug.log文件失败: {str(e)}")
                # 尝试强制删除（Windows系统）
                try:
                    import subprocess
                    if platform.system() == "Windows":
                        subprocess.run(["del", "/f", debug_log_path], shell=True, check=True)
                        print(f"🔧 使用系统命令强制删除debug.log文件成功")
                except Exception as force_e:
                    print(f"❌ 强制删除也失败: {str(force_e)}")

        conf_filename = conf_file.filename or "config.conf"
        conf_path = os.path.join(output_path, conf_filename)
        with open(conf_path, 'wb') as f:
            f.write(conf_content)
        
        # 2. 处理配置文件转义
        try:
            processed_conf_path = process_config_file(conf_path, output_path)
            print(f"配置文件已处理并保存到: {processed_conf_path}")
            
            # 检查OptimizerMainConfig.conf文件是否成功生成
            optimizer_config_dest = os.path.join(output_path, "OptimizerMainConfig.conf")
            if os.path.exists(optimizer_config_dest):
                print(f"✅ OptimizerMainConfig.conf已成功导入到任务文件夹: {optimizer_config_dest}")
                conf_path = processed_conf_path
            else:
                print(f"⚠️ OptimizerMainConfig.conf文件未生成: {optimizer_config_dest}")
            
            # 继续使用原始配置文件路径传递给backend_runner
            # conf_path 保持为原始的 config.conf 路径
        except Exception as e:
            print(f"配置文件处理失败，使用原始文件: {str(e)}")
            # 如果处理失败，继续使用原始配置文件

        # 3. 准备并执行命令，根据操作系统切换不同写法
        is_windows = platform.system() == "Windows"

        if is_windows:
            work_dir = "D:/huawei/ML_ICdesign-dev/ML_ICdesign/ML_ICdesign/OptimAlgorithm/Algorithm/" # D:/pythonprograms/BackendRunnerServer_master1/work_dir
            python_path_param = "C:/Users/shunxuanxi/.conda/envs/pytorch112_py311/python.exe"
            python_exe = find_python_executable(python_path_param or env)

            if python_exe is None:
                raise HTTPException(status_code=500, detail="Python executable not found. Please check the python path configuration.")
            script_path = os.path.join(work_dir, "backend_runner.py")

            # 规范化路径以避免中文字符编码问题
            normalized_output_path = os.path.normpath(output_path).replace('\\', '/')
            normalized_conf_path = os.path.normpath(conf_path).replace('\\', '/')

            command_list = [
                python_exe,
                "-u",
                script_path,
                "--output_path",
                normalized_output_path,
                "--conf_path",
                normalized_conf_path,
            ]
            print("command_list:",command_list)
            shell_flag = False
        else:
            # Linux / macOS 用原先 nohup 写法
            command_template = (
                "conda activate {env} && "
                "cd /OptimAlgore'sesithm/Algorithm && "
                "export PYTHONPATH=../.. && "
                "nohup env CUDA_VISIBLE_DEVICES={cuda_devices} python -u ./backend_runner.py "
                "--output_path {output_path} --conf_path {conf_path} > "
                "./run_$(date +%Y%m%d_%H%M%S).log 2>&1 &"
            )
            command_list = command_template.format(
                env=env,
                cuda_devices=cuda_devices,
                output_path=output_path,
                conf_path=conf_path,
            )
            work_dir = None
            shell_flag = True

        # 4. 在后台线程中运行所有任务
        def run_in_background():
            if is_windows:
                # --- MODIFIED SECTION START ---
                # 准备要传递给子进程的环境变量
                proc_env = os.environ.copy()
                pythonpath_value = "D:/huawei/ML_ICdesign-dev/ML_ICdesign/ML_ICdesign"
                proc_env["PYTHONPATH"] = pythonpath_value
                # 如果也需要在Windows上设置CUDA设备，可以在这里添加
                proc_env["CUDA_VISIBLE_DEVICES"] = cuda_devices
                print(f"Starting process with PYTHONPATH={pythonpath_value} and CUDA_VISIBLE_DEVICES={cuda_devices}")

                # 使用 detached 启动并传递修改后的环境变量
                process, logfile = run_command_detached(
                    command_list,
                    work_dir=work_dir,
                    logfile_name="run_.log",
                    env=proc_env  # 传递环境变量
                )
                # --- MODIFIED SECTION END ---
                processes[output_path] = process
            else:
                process = run_command(command_list, output_path, callback_url, work_dir=work_dir, shell=shell_flag)
                processes[output_path] = process
                logfile = None
            start_file_monitoring(output_path, callback_url)
            monitor_process(process, output_path, callback_url, logfile=logfile)

        thread = threading.Thread(target=run_in_background, daemon=True)
        thread.start()

        return {"message": "Algorithm execution started successfully.", "output_path": output_path}
    
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="Invalid JSON format in run_params.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/status/{output_path}")
async def get_status(output_path: str):
    """
    获取指定任务的状态 - 增强版状态检测
    支持多种检测方式：进程状态、日志文件活动、输出文件生成等
    """
    try:
        # 检查输出目录是否存在
        if not os.path.exists(output_path):
            return {
                "status": "not_found",
                "output_path": output_path,
                "message": "Output directory not found",
                "details": {}
            }
        
        # 收集状态信息
        status_info = {
            "process_running": False,
            "process_return_code": None,
            "has_completion_markers": False,
            "log_activity": False,
            "recent_file_activity": False,
            "completion_files": [],
            "log_files": [],
            "recent_files": [],
            "last_activity_time": None
        }
        
        # 1. 检查进程状态
        if output_path in processes:
            process = processes[output_path]
            if process.poll() is None:
                status_info["process_running"] = True
            else:
                status_info["process_return_code"] = process.returncode
        
        # 2. 检查完成标志文件
        completion_files = ["final_status.log", "completed.txt", "finished.log", "done.flag"]
        for comp_file in completion_files:
            comp_path = os.path.join(output_path, comp_file)
            if os.path.exists(comp_path):
                status_info["completion_files"].append(comp_file)
                status_info["has_completion_markers"] = True
        
        # 3. 检查日志文件活动（最近5分钟内有修改）
        current_time = datetime.now()
        log_patterns = ["*.log", "*.txt", "run_*.log"]
        recent_threshold = timedelta(minutes=5)
        
        for pattern in log_patterns:
            log_files = glob.glob(os.path.join(output_path, pattern))
            for log_file in log_files:
                if os.path.exists(log_file):
                    mod_time = datetime.fromtimestamp(os.path.getmtime(log_file))
                    status_info["log_files"].append({
                        "file": os.path.basename(log_file),
                        "last_modified": mod_time.isoformat(),
                        "size": os.path.getsize(log_file)
                    })
                    
                    if current_time - mod_time < recent_threshold:
                        status_info["log_activity"] = True
                        if not status_info["last_activity_time"] or mod_time > datetime.fromisoformat(status_info["last_activity_time"]):
                            status_info["last_activity_time"] = mod_time.isoformat()
        
        # 4. 检查最近文件活动（图片、输出文件等）
        file_patterns = ["*.png", "*.jpg", "*.jpeg", "*.json", "*.csv", "*.conf"]
        for pattern in file_patterns:
            # 递归查找所有匹配的文件
            files = glob.glob(os.path.join(output_path, "**", pattern), recursive=True)
            for file_path in files:
                if os.path.exists(file_path):
                    mod_time = datetime.fromtimestamp(os.path.getmtime(file_path))
                    if current_time - mod_time < recent_threshold:
                        status_info["recent_file_activity"] = True
                        rel_path = os.path.relpath(file_path, output_path)
                        status_info["recent_files"].append({
                            "file": rel_path,
                            "last_modified": mod_time.isoformat(),
                            "size": os.path.getsize(file_path)
                        })
                        
                        if not status_info["last_activity_time"] or mod_time > datetime.fromisoformat(status_info["last_activity_time"]):
                            status_info["last_activity_time"] = mod_time.isoformat()
        
        # 5. 根据收集的信息判断任务状态
        if status_info["process_running"]:
            # 进程还在运行
            final_status = "running"
            message = "Task is currently running"
        elif status_info["has_completion_markers"]:
            # 有完成标志文件
            final_status = "completed"
            message = f"Task completed (found completion markers: {', '.join(status_info['completion_files'])})"
        elif status_info["log_activity"] or status_info["recent_file_activity"]:
            # 最近有日志或文件活动
            final_status = "running"
            message = "Task appears to be running (recent file activity detected)"
        elif output_path in processes and status_info["process_return_code"] is not None:
            # 进程已结束
            if status_info["process_return_code"] == 0:
                final_status = "completed"
                message = f"Task completed successfully (return code: {status_info['process_return_code']})"
            else:
                final_status = "failed"
                message = f"Task failed (return code: {status_info['process_return_code']})"
        else:
            # 无法确定状态，可能是未知任务或已完成但无明确标志
            # 检查目录中是否有任何输出文件
            has_output_files = bool(status_info["log_files"]) or bool(status_info["recent_files"])
            if has_output_files:
                final_status = "completed"
                message = "Task likely completed (output files found, no recent activity)"
            else:
                final_status = "unknown"
                message = "Task status unknown (no process info, no output files)"
        
        return {
            "status": final_status,
            "output_path": output_path,
            "message": message,
            "details": status_info,
            "timestamp": current_time.isoformat()
        }
        
    except Exception as e:
        return {
            "status": "error",
            "output_path": output_path,
            "message": f"Error checking task status: {str(e)}",
            "details": {},
            "timestamp": datetime.now().isoformat()
        }


@app.post("/status")
async def get_status(request: dict):
    """
    获取指定任务的状态 - 增强版状态检测
    支持多种检测方式：进程状态、日志文件活动、输出文件生成、优化进度监控等
    请求体格式: {"output_path": "绝对路径"}
    """
    try:
        # 从请求体中获取输出路径
        output_path = request.get("output_path")
        if not output_path:
            return {
                "status": "error",
                "output_path": "",
                "message": "Missing output_path in request body",
                "details": {},
                "timestamp": datetime.now().isoformat()
            }

        # 检查输出目录是否存在
        if not os.path.exists(output_path):
            return {
                "status": "not_found",
                "output_path": output_path,
                "message": "Output directory not found",
                "details": {}
            }

        # 收集状态信息
        status_info = {
            "process_running": False,
            "process_return_code": None,
            "has_completion_markers": False,
            "log_activity": False,
            "recent_file_activity": False,
            "completion_files": [],
            "log_files": [],
            "recent_files": [],
            "last_activity_time": None,
            "progress_monitor": {
                "enabled": False,
                "status": "unknown",
                "details": {},
                "progress_summary": {}
            }
        }

        # 1. 检查进程状态
        if output_path in processes:
            process = processes[output_path]
            if process.poll() is None:
                status_info["process_running"] = True
            else:
                status_info["process_return_code"] = process.returncode

        # 2. 检查完成标志文件
        completion_files = ["final_status.log", "completed.txt", "finished.log", "done.flag"]
        for comp_file in completion_files:
            comp_path = os.path.join(output_path, comp_file)
            if os.path.exists(comp_path):
                status_info["completion_files"].append(comp_file)
                status_info["has_completion_markers"] = True

        # 3. 检查日志文件活动（最近5分钟内有修改）
        current_time = datetime.now()
        log_patterns = ["*.log", "*.txt", "run_*.log"]
        recent_threshold = timedelta(minutes=5)

        for pattern in log_patterns:
            log_files = glob.glob(os.path.join(output_path, pattern))
            for log_file in log_files:
                if os.path.exists(log_file):
                    mod_time = datetime.fromtimestamp(os.path.getmtime(log_file))
                    status_info["log_files"].append({
                        "file": os.path.basename(log_file),
                        "last_modified": mod_time.isoformat(),
                        "size": os.path.getsize(log_file)
                    })

                    if current_time - mod_time < recent_threshold:
                        status_info["log_activity"] = True
                        if not status_info["last_activity_time"] or mod_time > datetime.fromisoformat(status_info["last_activity_time"]):
                            status_info["last_activity_time"] = mod_time.isoformat()

        # 4. 检查最近文件活动（图片、输出文件等）
        file_patterns = ["*.png", "*.jpg", "*.jpeg", "*.json", "*.csv", "*.conf"]
        for pattern in file_patterns:
            # 递归查找所有匹配的文件
            files = glob.glob(os.path.join(output_path, "**", pattern), recursive=True)
            for file_path in files:
                if os.path.exists(file_path):
                    mod_time = datetime.fromtimestamp(os.path.getmtime(file_path))
                    if current_time - mod_time < recent_threshold:
                        status_info["recent_file_activity"] = True
                        rel_path = os.path.relpath(file_path, output_path)
                        status_info["recent_files"].append({
                            "file": rel_path,
                            "last_modified": mod_time.isoformat(),
                            "size": os.path.getsize(file_path)
                        })

                        if not status_info["last_activity_time"] or mod_time > datetime.fromisoformat(status_info["last_activity_time"]):
                            status_info["last_activity_time"] = mod_time.isoformat()

        # 5. 进度监控检查（检查debug.log中的优化进度）
        debug_log_path = os.path.join(output_path, "debug.log")
        if os.path.exists(debug_log_path):
            try:
                # 创建进度监控器进行一次性状态检查
                progress_monitor = ProgressMonitor(
                    log_file_path=debug_log_path,
                    timeout_minutes=10  # 10分钟无进度更新视为完成
                )

                # 获取当前进度状态
                progress_status, progress_details = progress_monitor.get_current_status()
                progress_summary = progress_monitor.get_progress_summary()

                status_info["progress_monitor"] = {
                    "enabled": True,
                    "status": progress_status,
                    "details": progress_details,
                    "progress_summary": progress_summary
                }

                # 如果进度监控检测到明确状态，更新最后活动时间
                if progress_summary.get("latest_progress"):
                    latest_progress_time = progress_summary["latest_progress"]["timestamp"]
                    if not status_info["last_activity_time"] or latest_progress_time > status_info["last_activity_time"]:
                        status_info["last_activity_time"] = latest_progress_time

            except Exception as e:
                status_info["progress_monitor"] = {
                    "enabled": False,
                    "status": "error",
                    "details": {"error": f"进度监控失败: {str(e)}"},
                    "progress_summary": {}
                }

        # 6. 根据收集的信息判断任务状态（优先级：进度监控 > 进程状态 > 完成标志 > 文件活动）
        progress_monitor_status = status_info["progress_monitor"].get("status", "unknown")

        if status_info["process_running"]:
            # 进程还在运行
            final_status = "running"
            message = "Task is currently running"
        elif progress_monitor_status == "completed":
            # 进度监控检测到完成
            final_status = "completed"
            progress_reason = status_info["progress_monitor"]["details"].get("reason", "进度监控检测到完成")
            message = f"Task completed (progress monitor: {progress_reason})"
        elif progress_monitor_status == "failed":
            # 进度监控检测到失败
            final_status = "failed"
            progress_reason = status_info["progress_monitor"]["details"].get("reason", "进度监控检测到失败")
            message = f"Task failed (progress monitor: {progress_reason})"
        elif progress_monitor_status == "running":
            # 进度监控检测到运行中
            final_status = "running"
            progress_reason = status_info["progress_monitor"]["details"].get("reason", "进度监控检测到运行中")
            message = f"Task is running (progress monitor: {progress_reason})"
        elif status_info["has_completion_markers"]:
            # 有完成标志文件
            final_status = "completed"
            message = f"Task completed (found completion markers: {', '.join(status_info['completion_files'])})"
        elif status_info["log_activity"] or status_info["recent_file_activity"]:
            # 最近有日志或文件活动
            final_status = "running"
            message = "Task appears to be running (recent file activity detected)"
        elif output_path in processes and status_info["process_return_code"] is not None:
            # 进程已结束
            if status_info["process_return_code"] == 0:
                final_status = "completed"
                message = f"Task completed successfully (return code: {status_info['process_return_code']})"
            else:
                final_status = "failed"
                message = f"Task failed (return code: {status_info['process_return_code']})"
        else:
            # 无法确定状态，可能是未知任务或已完成但无明确标志
            # 检查目录中是否有任何输出文件
            has_output_files = bool(status_info["log_files"]) or bool(status_info["recent_files"])
            if has_output_files:
                final_status = "completed"
                message = "Task likely completed (output files found, no recent activity)"
            else:
                final_status = "unknown"
                message = "Task status unknown (no process info, no output files)"

        return {
            "status": final_status,
            "output_path": output_path,
            "message": message,
            "details": status_info,
            "timestamp": current_time.isoformat()
        }

    except Exception as e:
        return {
            "status": "error",
            "output_path": output_path,
            "message": f"Error checking task status: {str(e)}",
            "details": {},
            "timestamp": datetime.now().isoformat()
        }

@app.get("/api/results/{output_path}/images")
async def get_images_list(output_path: str) -> List[str]:
    """
    获取指定输出路径下的所有图片文件列表（包括子文件夹）
    """
    try:
        # 检查输出路径是否存在
        if not os.path.exists(output_path):
            raise HTTPException(status_code=404, detail=f"Output path '{output_path}' not found")
        
        # 获取所有图片文件（递归查找子文件夹）
        image_extensions = ['*.png', '*.jpg', '*.jpeg', '*.gif', '*.bmp']
        image_files = []
        
        for extension in image_extensions:
            # 使用 ** 进行递归查找
            pattern = os.path.join(output_path, '**', extension)
            image_files.extend(glob.glob(pattern, recursive=True))
        
        # 返回相对于output_path的路径
        image_paths = []
        for img in image_files:
            # 获取相对于output_path的路径
            rel_path = os.path.relpath(img, output_path)
            # 统一使用正斜杠作为路径分隔符
            rel_path = rel_path.replace(os.sep, '/')
            image_paths.append(rel_path)
        
        image_paths.sort()  # 按路径排序
        
        return image_paths
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving images: {str(e)}")


@app.get("/api/results/{output_path}/images/{image_path:path}")
async def download_image(output_path: str, image_path: str):
    """
    下载指定的图片文件（支持子文件夹路径）
    """
    try:
        # 将URL路径中的正斜杠转换为系统路径分隔符
        normalized_image_path = image_path.replace('/', os.sep)

        # 构建完整的文件路径
        file_path = os.path.join(output_path, normalized_image_path)

        # 安全检查：确保文件路径在output_path内
        real_output_path = os.path.realpath(output_path)
        real_file_path = os.path.realpath(file_path)
        if not real_file_path.startswith(real_output_path):
            raise HTTPException(status_code=400, detail="Invalid file path")

        # 检查文件是否存在
        if not os.path.exists(file_path):
            raise HTTPException(status_code=404, detail=f"Image '{image_path}' not found in '{output_path}'")

        # 获取文件名用于检查扩展名
        filename = os.path.basename(file_path)

        # 检查是否为图片文件
        if not filename.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp')):
            raise HTTPException(status_code=400, detail="File is not a valid image format")

        # 根据文件扩展名设置正确的MIME类型
        file_ext = filename.lower().split('.')[-1]
        mime_types = {
            'png': 'image/png',
            'jpg': 'image/jpeg',
            'jpeg': 'image/jpeg',
            'gif': 'image/gif',
            'bmp': 'image/bmp'
        }
        media_type = mime_types.get(file_ext, 'application/octet-stream')

        # 返回文件
        return FileResponse(
            path=file_path,
            filename=filename,
            media_type=media_type
        )

    except HTTPException:
        raise
    except HTTPException:
        raise
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error downloading image: {str(e)}")


@app.post("/api/results/images")
async def get_images_list_post(request: ImageListRequest) -> List[str]:
    """
    获取指定输出路径下的所有图片文件列表（支持绝对路径）
    """
    try:
        output_path = request.output_path

        if not os.path.exists(output_path):
            raise HTTPException(status_code=404, detail=f"Output path '{output_path}' not found")

        image_extensions = ['*.png', '*.jpg', '*.jpeg', '*.gif', '*.bmp']
        image_files = []

        for extension in image_extensions:
            pattern = os.path.join(output_path, '**', extension)
            image_files.extend(glob.glob(pattern, recursive=True))

        image_paths = []
        for img in image_files:
            rel_path = os.path.relpath(img, output_path)
            rel_path = rel_path.replace(os.sep, '/')
            image_paths.append(rel_path)

        image_paths.sort()
        return image_paths

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving images: {str(e)}")


@app.post("/api/results/images/download")
async def download_image_post(request: ImageDownloadRequest):
    """
    下载指定的图片文件（支持绝对路径）
    """
    try:
        output_path = request.output_path
        image_path = request.image_path

        normalized_image_path = image_path.replace('/', os.sep)
        file_path = os.path.join(output_path, normalized_image_path)

        # 安全检查
        if not os.path.abspath(file_path).startswith(os.path.abspath(output_path)):
            raise HTTPException(status_code=403, detail="Access denied: Invalid file path")

        if not os.path.exists(file_path):
            raise HTTPException(status_code=404, detail=f"Image file '{image_path}' not found")

        valid_extensions = ['.png', '.jpg', '.jpeg', '.gif', '.bmp']
        if not any(file_path.lower().endswith(ext) for ext in valid_extensions):
            raise HTTPException(status_code=400, detail="Invalid image file format")

        return FileResponse(
            path=file_path,
            media_type='application/octet-stream',
            filename=os.path.basename(file_path)
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error downloading image: {str(e)}")

@app.get("/local-images")
async def get_all_local_images() -> dict:
    """
    获取当前工作目录下所有输出文件夹中的图片文件
    """
    try:
        current_dir = os.getcwd()
        all_images = {}
        
        # 遍历当前目录下的所有子目录
        for item in os.listdir(current_dir):
            item_path = os.path.join(current_dir, item)
            if os.path.isdir(item_path):
                # 获取该目录下的图片文件
                image_extensions = ['*.png', '*.jpg', '*.jpeg', '*.gif', '*.bmp']
                image_files = []
                
                for extension in image_extensions:
                    pattern = os.path.join(item_path, extension)
                    image_files.extend(glob.glob(pattern))
                
                if image_files:
                    # 只保存文件名
                    image_names = [os.path.basename(img) for img in image_files]
                    image_names.sort()
                    all_images[item] = image_names
        
        return {
            "total_directories": len(all_images),
            "directories": all_images
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving local images: {str(e)}")


@app.get("/api/results/{output_path}/files")
async def get_all_files(output_path: str) -> dict:
    """
    获取指定输出路径下的所有文件信息
    """
    try:
        # 检查输出路径是否存在
        if not os.path.exists(output_path):
            raise HTTPException(status_code=404, detail=f"Output path '{output_path}' not found")
        
        files_info = []
        total_size = 0
        
        for file_name in os.listdir(output_path):
            file_path = os.path.join(output_path, file_name)
            if os.path.isfile(file_path):
                file_size = os.path.getsize(file_path)
                file_ext = os.path.splitext(file_name)[1].lower()
                
                files_info.append({
                    "name": file_name,
                    "size": file_size,
                    "extension": file_ext,
                    "is_image": file_ext in ['.png', '.jpg', '.jpeg', '.gif', '.bmp']
                })
                total_size += file_size
        
        # 按文件名排序
        files_info.sort(key=lambda x: x['name'])
        
        return {
            "output_path": output_path,
            "total_files": len(files_info),
            "total_size": total_size,
            "files": files_info
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving files: {str(e)}")


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=7000)
