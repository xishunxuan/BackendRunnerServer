# BackendRunnerServer API 参数需求文档

## 概述

本文档详细说明了 BackendRunnerServer 项目中各个 API 接口的参数需求和格式要求，用于指导前端开发和回调功能的配置。

## API 接口详情

### 1. 启动算法任务 API

**接口地址**: `POST /run-algorithm/`  
**Content-Type**: `multipart/form-data`

#### 请求参数

##### 文件上传部分

| 参数名 | 类型 | 必需 | 说明 |
|--------|------|------|------|
| conf_file | UploadFile | 是 | 配置文件，格式为 .conf |

**配置文件示例内容**:
```ini
[Settings]
param1 = value1
param2 = 123
```

##### 表单数据部分

| 参数名 | 类型 | 必需 | 说明 |
|--------|------|------|------|
| run_params | string (JSON) | 是 | JSON格式的运行参数 |

**run_params JSON 结构**:

| 字段名 | 类型 | 必需 | 默认值 | 说明 |
|--------|------|------|--------|------|
| output_path | string | 是 | - | 输出文件路径 |
| callback_url | string | 是 | - | 回调URL地址 |
| env | string | 否 | - | Python环境名称 |
| cuda_devices | string | 否 | "0" | GPU设备号 |

#### 请求示例

**JavaScript 示例**:
```javascript
const formData = new FormData();

// 文件部分
formData.append('conf_file', configFile);

// 参数部分
const runParams = {
    "output_path": "D:/output/path",
    "callback_url": "https://your-callback-url.com",
    "cuda_devices": "0",
    "env": "pytorch112_py311"
};
formData.append('run_params', JSON.stringify(runParams));

// 发送请求
fetch('/run-algorithm/', {
    method: 'POST',
    body: formData
});
```

**Python 示例**:
```python
import requests
import json

files = {
    'conf_file': ('config.conf', open('config.conf', 'rb'), 'text/plain')
}

data = {
    'run_params': json.dumps({
        "output_path": "D:/output/path",
        "callback_url": "https://your-callback-url.com",
        "cuda_devices": "0",
        "env": "pytorch112_py311"
    })
}

response = requests.post('http://localhost:7000/run-algorithm/', files=files, data=data)
```

#### 响应格式

**成功响应** (200):
```json
{
    "message": "Algorithm execution started successfully.",
    "output_path": "实际的输出路径"
}
```

**错误响应**:
- **400 Bad Request**: 
  ```json
  {"detail": "Invalid JSON format in run_params."}
  ```
  ```json
  {"detail": "output_path and callback_url are required."}
  ```
- **500 Internal Server Error**:
  ```json
  {"detail": "具体错误信息"}
  ```

### 2. 状态查询 API

**接口地址**: `GET /status/{output_path}`

#### 请求参数

| 参数名 | 类型 | 位置 | 必需 | 说明 |
|--------|------|------|------|------|
| output_path | string | URL路径 | 是 | 输出路径（需要URL编码） |

#### 请求示例

```javascript
// JavaScript
const outputPath = encodeURIComponent('D:/output/path');
fetch(`/status/${outputPath}`);
```

```python
# Python
import urllib.parse

output_path = urllib.parse.quote('D:/output/path')
response = requests.get(f'http://localhost:7000/status/{output_path}')
```

#### 响应格式

```json
{
    "status": "running|finished|not_found",
    "output_path": "输出路径",
    "return_code": "进程返回码(如果已完成)"
}
```

**状态说明**:
- `running`: 进程正在运行
- `finished`: 进程已完成
- `not_found`: 未找到对应的进程

## 回调功能

### 文件变化回调

当输出目录中的文件发生变化时，服务器会向指定的 `callback_url` 发送 POST 请求。

**触发条件**: 监控的文件类型 (`.jpg`, `.png`, `.log`) 发生创建或修改事件

**请求格式**: `multipart/form-data`

| 参数名 | 类型 | 说明 |
|--------|------|------|
| file | File | 变化的文件内容 |
| output_path | string | 输出路径 |

### 进程完成回调

当算法执行完成时，服务器会向 `callback_url` 发送 POST 请求。

**请求格式**: `application/json`

```json
{
    "status": "completed",
    "output_path": "输出路径",
    "return_code": "进程返回码",
    "log_excerpt": "日志摘要(最后20000字符)"
}
```

## 配置建议

### 回调功能优化建议

1. **回调URL验证**: 建议在接收请求时验证 callback_url 的有效性
2. **重试机制**: 当回调失败时，实现指数退避重试逻辑
3. **超时设置**: 当前回调请求超时设置为10秒，可根据需要调整
4. **日志记录**: 增加详细的回调日志记录，便于调试
5. **认证机制**: 为回调添加认证头或签名验证，提高安全性
6. **错误处理**: 完善回调失败的错误处理和通知机制

### 安全注意事项

1. **路径验证**: 验证 output_path 的合法性，防止路径遍历攻击
2. **文件大小限制**: 对上传的配置文件设置合理的大小限制
3. **URL白名单**: 对 callback_url 实施白名单机制
4. **输入验证**: 对所有输入参数进行严格的格式和内容验证

## 依赖库

项目依赖的主要库（来自 requirements.txt）:

```
fastapi
uvicorn[standard]
python-multipart
requests
watchdog
Pillow
```

## 服务器配置

- **默认端口**: 7000
- **主机地址**: 0.0.0.0 (监听所有接口)
- **启动命令**: `python workstation_server.py`

## 测试

项目包含测试客户端 `test_client.py`，可用于测试 API 功能。使用前需要:

1. 更新 `CALLBACK_URL` 为有效的回调地址
2. 确保 `OUTPUT_PATH` 目录存在
3. 配置正确的 Python 环境

---

*文档生成时间: 2024年*  
*项目: BackendRunnerServer*