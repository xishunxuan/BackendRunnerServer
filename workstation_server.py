import os
import json
import subprocess
import threading
import platform
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from pydantic import BaseModel
import uvicorn
import requests
import shutil
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

app = FastAPI()

# 全局变量来跟踪进程和监控状态
processes = {}
monitors = {}

class RunConfig(BaseModel):
    output_path: str
    conf_content: str
    server_ip: str
    server_port: int

import subprocess, os

def run_command_detached(cmd_list, work_dir=None, logfile_name="run_.log"):
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
        close_fds=True
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
    # if output_path in processes:
    #     del processes[output_path]
    
    # del processes[output_path]


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
        callback_url = params.get("callback_url") # 回调URL

        if not all([output_path, callback_url]):
             raise HTTPException(status_code=400, detail="output_path and callback_url are required.")
        
        # 1. 保存 .conf 文件
        conf_content = await conf_file.read()
        
        # 创建输出目录，如果它不存在
        os.makedirs(output_path, exist_ok=True)
        
        conf_filename = conf_file.filename or "config.conf"
        conf_path = os.path.join(output_path, conf_filename)
        with open(conf_path, 'wb') as f:
            f.write(conf_content)

        # 2. 准备并执行命令，根据操作系统切换不同写法
        is_windows = platform.system() == "Windows"

        if is_windows:
            work_dir = "D:/pythonprograms/BackendRunnerServer"
            # 优先接受 run_params 里可能传入的 python_path
            python_path_param = "C:/Users/shunxuanxi/.conda/envs/pytorch112_py311/python.exe"
            python_exe = find_python_executable(python_path_param or env)
            command_list = [
                python_exe,
                "-u",
                "backend_runner.py",
                "--output_path",
                output_path,
                "--conf_path",
                conf_path,
            ]
            shell_flag = False
        else:
            # Linux / macOS 用原先 nohup 写法
            command_template = (
                "conda activate {env} && "
                "cd /OptimAlgorithm/Algorithm && "
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

        # 3. 在后台线程中运行所有任务
        def run_in_background():
            if is_windows:
                # 使用 detached 启动并把输出写到 work_dir/run_.log
                process, logfile = run_command_detached(command_list, work_dir=work_dir, logfile_name="run_.log")
            else:
                process = run_command(command_list, output_path, callback_url, work_dir=work_dir, shell=shell_flag)
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
    获取指定任务的状态
    """
    if output_path not in processes:
        return {"status": "not_found", "message": "No process found for the given output path."}
    
    process = processes[output_path]
    if process.poll() is None:
        status = "running"
    else:
        status = "finished"
        
    return {
        "status": status,
        "output_path": output_path,
        "return_code": process.returncode
    }


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=7000)
