import logging
import requests
import re
import json
import time
from typing import Optional, Union, Generator, Callable, Dict

logger = logging.getLogger(__name__)

# DeepSeek API 端点，兼容 OpenAI 格式
DEEPSEEK_API_URL = "https://api.deepseek.com/v1/chat/completions"

def extract_diff_from_response(response_text: str) -> Optional[str]:
    """从 API 响应中提取 diff 格式的补丁内容。"""
    logger.debug(f"尝试从响应中提取 diff，响应长度: {len(response_text)}")
    # 尝试查找 Markdown 格式的 diff 代码块
    diff_pattern = re.compile(r"```(?:diff|patch)?\n([\s\S]*?)\n```", re.MULTILINE)
    match = diff_pattern.search(response_text)
    if match:
        logger.debug("在 Markdown 代码块中找到 diff")
        return match.group(1).strip()
    
    # 尝试提取直接的 diff 格式内容（以 --- 或 +++ 或 diff --git 开头的行）
    lines = response_text.split('\n')
    diff_lines = []
    in_diff = False
    
    for line in lines:
        if line.startswith(('--- ', '+++ ', 'diff --git ', '@@ ')):
            if not in_diff:
                logger.debug(f"发现 diff 行标记: {line[:30]}...")
            in_diff = True
            diff_lines.append(line)
        elif in_diff and (line.startswith(' ') or line.startswith('+') or line.startswith('-') or line.startswith('@')):
            diff_lines.append(line)
        elif in_diff and line.strip() == '':
            diff_lines.append(line)  # 保留空行
        elif in_diff:
            # 如果已经在 diff 块中，但当前行不符合 diff 格式，则结束提取
            logger.debug(f"diff 块结束于: {line[:30]}...")
            in_diff = False
    
    if diff_lines:
        logger.debug(f"从原始文本中提取了 {len(diff_lines)} 行 diff 内容")
        return '\n'.join(diff_lines)
    
    # 如果以上方法都失败，则假设整个响应可能是一个 diff（较少见）
    if '--- ' in response_text and '+++ ' in response_text:
        logger.debug("整个响应似乎是一个完整的 diff")
        return response_text
    
    logger.warning("无法从响应中提取 diff 格式内容")
    return None

def _process_deepseek_response(data) -> str:
    """从 DeepSeek API 响应中提取内容"""
    if data.get("choices") and len(data["choices"]) > 0 and data["choices"][0].get("message"):
        patch_content = data["choices"][0]["message"].get("content", "").strip()
        return patch_content
    return ""

def _process_deepseek_stream_chunk(chunk) -> str:
    """从 DeepSeek API 流式响应块中提取内容"""
    if "choices" in chunk and len(chunk["choices"]) > 0:
        delta = chunk["choices"][0].get("delta", {})
        if "content" in delta:
            return delta["content"]
    return ""

def generate_patch_with_deepseek(
    api_key: str, 
    file_content: str, 
    issue_description: str, 
    file_path: str, 
    previous_error_log: str | None = None,
    stream: bool = False,
    stream_callback: Callable[[str], None] = None
) -> Union[str, Generator[str, None, str], None]:
    """
    使用 DeepSeek API 生成代码补丁。
    返回 unified diff 格式的补丁字符串，或在流式模式下返回生成器。
    
    :param api_key: DeepSeek API 密钥
    :param file_content: 需要修复的文件内容
    :param issue_description: GitHub Issue 描述
    :param file_path: 文件路径
    :param previous_error_log: 之前修复尝试的错误日志（可选）
    :param stream: 是否使用流式响应
    :param stream_callback: 流式响应回调函数，接收每个块的内容
    :return: unified diff 格式的补丁、生成器或 None（如果生成失败）
    """
    logger.info(f"开始为文件 {file_path} 调用 DeepSeek API 生成补丁...")
    logger.debug(f"使用 {'流式' if stream else '非流式'} API 调用模式")

    # 构建提示词
    prompt = f"""你是一个专业的C/C++代码修复助手。
请分析以下C/C++代码文件 '{file_path}' 的内容以及相关的GitHub Issue描述，并生成一个修复该问题的补丁。

GitHub Issue 描述:
---
{issue_description}
---

文件 '{file_path}' 的原始内容:
---
{file_content}
---
"""

    if previous_error_log:
        prompt += f"""
上一次尝试生成的补丁在测试时产生了以下错误，请根据此错误信息改进你的补丁:
---
{previous_error_log}
---
"""
    prompt += "\n请输出 unified diff 格式的修复补丁。确保补丁格式正确，可以直接应用到原始文件。\n"

    logger.debug(f"生成的提示词长度: {len(prompt)} 字符")

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    
    # 根据 DeepSeek API 文档调整 payload
    payload = {
        "model": "deepseek-chat",  # 使用最新的 DeepSeek-V3 模型
        "messages": [
            {"role": "system", "content": "You are a C/C++ code repair assistant specialized in creating patches in unified diff format."},
            {"role": "user", "content": prompt}
        ],
        "max_tokens": 2000,  # 适当增加 token 数量以确保完整的补丁
        "temperature": 0.1,  # 使用较低的温度以获得更确定的输出
        "stream": stream     # 根据参数决定是否使用流式输出
    }

    logger.debug(f"API 请求头: {headers}")
    logger.debug(f"API 请求负载: {json.dumps(payload)}")

    max_retries = 3
    retry_delay = 2  # 秒
    
    for attempt in range(max_retries):
        try:
            logger.info(f"发送 API 请求 (尝试 {attempt+1}/{max_retries})...")
            
            if stream:
                return _handle_stream_response(DEEPSEEK_API_URL, headers, payload, stream_callback)
            else:
                response = requests.post(
                    DEEPSEEK_API_URL, 
                    headers=headers, 
                    json=payload, 
                    timeout=120  # 设置较长的超时时间
                )
                
                # 记录完整的 API 响应
                logger.debug(f"API 响应状态码: {response.status_code}")
                logger.debug(f"API 响应头: {response.headers}")
                
                # 检查 HTTP 状态码
                if response.status_code == 429:  # 速率限制
                    retry_after = int(response.headers.get('Retry-After', retry_delay))
                    logger.warning(f"API 速率限制，等待 {retry_after} 秒后重试...")
                    time.sleep(retry_after)
                    continue
                    
                response.raise_for_status()  # 处理其他 HTTP 错误
                
                data = response.json()
                logger.debug(f"API 响应正文: {json.dumps(data, indent=2)}")
                
                # 解析 DeepSeek 的响应以获取补丁
                patch_content = _process_deepseek_response(data)
                
                if not patch_content:
                    logger.error("API 返回了空的补丁内容")
                    return None
                
                logger.debug(f"从API获取的原始内容 ({len(patch_content)} 字符): {patch_content[:200]}...")
                
                # 尝试提取 diff 格式的内容
                diff_content = extract_diff_from_response(patch_content)
                
                if diff_content:
                    logger.info(f"成功从 DeepSeek API 获取到补丁 ({len(diff_content)} 字符)")
                    return diff_content
                else:
                    logger.warning(f"API 返回的内容似乎不是有效的 diff 格式，将尝试直接使用返回内容。")
                    return patch_content

        except requests.exceptions.RequestException as e:
            logger.error(f"调用 DeepSeek API 失败 (尝试 {attempt+1}/{max_retries}): {e}")
            if attempt < max_retries - 1:
                sleep_time = retry_delay * (attempt + 1)  # 指数退避
                logger.info(f"将在 {sleep_time} 秒后重试")
                time.sleep(sleep_time)
            else:
                logger.error("所有重试失败")
                return None
        except Exception as e:
            logger.exception(f"处理 DeepSeek API 响应时发生未知错误: {e}")
            return None
    
    logger.error("所有 API 调用尝试均失败")
    return None  # 所有尝试都失败

def _handle_stream_response(url, headers, payload, callback=None) -> Generator[str, None, str]:
    """
    处理流式 API 响应
    
    :param url: API 端点
    :param headers: 请求头
    :param payload: 请求负载
    :param callback: 可选的回调函数，接收每个块的文本
    :return: 生成器，产出每个响应块的内容
    """
    logger.info("开始流式 API 调用")
    
    # 确保是流式调用
    payload["stream"] = True
    
    try:
        with requests.post(url, headers=headers, json=payload, stream=True, timeout=300) as response:
            response.raise_for_status()
            
            logger.debug(f"流式响应开始，状态码: {response.status_code}")
            
            accumulated_content = ""
            
            for line in response.iter_lines():
                if line:
                    # 跳过空行和 'data: [DONE]'
                    if line == b'data: [DONE]':
                        logger.debug("收到流式响应结束标记")
                        continue
                        
                    # 移除 'data: ' 前缀并解析 JSON
                    if line.startswith(b'data: '):
                        json_str = line[6:].decode('utf-8')
                        try:
                            chunk = json.loads(json_str)
                            logger.debug(f"收到流式响应块: {json_str[:100]}...")
                            
                            content = _process_deepseek_stream_chunk(chunk)
                            if content:
                                logger.debug(f"提取的内容块: {content}")
                                accumulated_content += content
                                
                                if callback:
                                    callback(content)
                                    
                                # 同时产出给调用者
                                yield content
                        except json.JSONDecodeError:
                            logger.warning(f"无法解析响应块 JSON: {line}")
            
            logger.info(f"流式响应完成，累积内容长度: {len(accumulated_content)}")
            
            # 尝试从累积的内容中提取 diff
            diff_content = extract_diff_from_response(accumulated_content)
            if diff_content:
                logger.info("成功从流式响应中提取 diff")
                return diff_content
            else:
                logger.warning("无法从流式响应中提取有效的 diff，返回原始内容")
                return accumulated_content
                
    except Exception as e:
        logger.exception(f"流式响应处理失败: {e}")
        return None

# 在 ai_handler.py 中添加对应的函数：

def generate_patch_with_context(
    api_key: str, 
    file_content: str, 
    issue_description: str, 
    file_path: str, 
    context_files: dict = None,
    previous_error_log: str = None,
    stream: bool = False,
    stream_callback = None
) -> str | None:
    """
    使用 DeepSeek API 生成代码补丁，支持多文件上下文。
    
    :param api_key: DeepSeek API 密钥
    :param file_content: 需要修复的主文件内容
    :param issue_description: GitHub Issue 描述
    :param file_path: 主文件路径
    :param context_files: 相关联的上下文文件 {文件路径: 文件内容}
    :param previous_error_log: 之前修复尝试的错误日志（可选）
    :param stream: 是否使用流式响应
    :param stream_callback: 流式响应回调函数
    :return: unified diff 格式的补丁或 None（如果生成失败）
    """
    logger.info(f"开始为文件 {file_path} 调用 DeepSeek API 生成补丁...")
    
    # 处理上下文文件
    context_str = ""
    if context_files and len(context_files) > 1:  # 有除了主文件外的上下文
        context_str = "相关的上下文文件:\n"
        for rel_path, content in context_files.items():
            if rel_path != file_path:  # 跳过主文件
                # 只包含文件的前 1000 个字符作为上下文
                preview = content[:1000] + ("..." if len(content) > 1000 else "")
                context_str += f"\n文件 '{rel_path}':\n```\n{preview}\n```\n"

    # 构建提示词
    prompt = f"""你是一个专业的C/C++代码修复助手。
请分析以下C/C++代码文件 '{file_path}' 的内容以及相关的GitHub Issue描述，并生成一个修复该问题的补丁。

GitHub Issue 描述:
---
{issue_description}
---

文件 '{file_path}' 的原始内容:
---
{file_content}
---
"""

    if context_str:
        prompt += f"\n{context_str}\n"

    if previous_error_log:
        prompt += f"""
上一次尝试生成的补丁在测试时产生了以下错误，请根据此错误信息改进你的补丁:
---
{previous_error_log}
---
"""
    prompt += "\n请输出 unified diff 格式的修复补丁。确保补丁格式正确，可以直接应用到原始文件。\n"

    # 使用现有的生成逻辑，但传递修改后的提示词
    if stream:
        return _handle_stream_response(DEEPSEEK_API_URL, 
                                      {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
                                      {
                                          "model": "deepseek-chat",
                                          "messages": [
                                              {"role": "system", "content": "You are a C/C++ code repair assistant specialized in creating patches in unified diff format."},
                                              {"role": "user", "content": prompt}
                                          ],
                                          "max_tokens": 2000,
                                          "temperature": 0.1,
                                          "stream": True
                                      },
                                      stream_callback)
    else:
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": "deepseek-chat",
            "messages": [
                {"role": "system", "content": "You are a C/C++ code repair assistant specialized in creating patches in unified diff format."},
                {"role": "user", "content": prompt}
            ],
            "max_tokens": 2000,
            "temperature": 0.1,
            "stream": False
        }
        
        # 调用现有的非流式API
        try:
            response = requests.post(
                DEEPSEEK_API_URL, 
                headers=headers, 
                json=payload, 
                timeout=120
            )
            
            response.raise_for_status()
            data = response.json()
            
            if data.get("choices") and len(data["choices"]) > 0 and data["choices"][0].get("message"):
                patch_content = data["choices"][0]["message"].get("content", "").strip()
                
                if not patch_content:
                    logger.error("API 返回了空的补丁内容")
                    return None
                
                diff_content = extract_diff_from_response(patch_content)
                
                if diff_content:
                    logger.info("成功从 DeepSeek API 获取到补丁。")
                    return diff_content
                else:
                    logger.warning(f"API 返回的内容似乎不是有效的 diff 格式，将尝试直接使用返回内容。")
                    return patch_content
            else:
                logger.error(f"DeepSeek API 响应格式不符合预期: {data}")
                return None
        except Exception as e:
            logger.exception(f"调用 DeepSeek API 失败: {e}")
            return None