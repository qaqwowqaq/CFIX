import logging
import requests
import re
import json
import time
from typing import List
from typing import Optional, Union, Generator, Callable, Dict, Tuple

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

def _process_deepseek_response_content(data) -> str:
    """从 DeepSeek API 响应数据中提取 message content"""
    if data.get("choices") and len(data["choices"]) > 0 and data["choices"][0].get("message"):
        return data["choices"][0]["message"].get("content", "").strip()
    return ""

# ... ( _process_deepseek_stream_chunk and _handle_stream_response can remain similar, 
# but _handle_stream_response should also return the full accumulated_content and prompt)
# For simplicity in this step, we'll focus on the non-streaming part first.
# The streaming part would need a more significant refactor to return all necessary info.

def generate_patch_with_context(
    api_key: str,
    file_content: str,
    issue_description: str,
    file_path: str,
    context_files: dict = None,
    previous_error_log: str = None,
    test_cases: Optional[List[Dict[str, str]]] = None, # 新增: test_cases = [{"name": "test1.cpp", "content": "...", "description": "..."}, ...]
    test_patch_diff_content: Optional[str] = None,
    stream: bool = False, # Streaming not fully adapted for detailed return yet
    stream_callback=None
) -> Dict[str, any]:
    """
    使用 DeepSeek API 生成代码补丁，支持多文件上下文。
    
    :return: 字典，包含 "prompt_sent", "raw_response", "extracted_patch", "error_message", "status_code"
    """
    logger.info(f"开始为文件 {file_path} 调用 DeepSeek API 生成补丁...")
    
    # 初始化返回结果字典
    result = {
        "prompt_sent": None,
        "raw_response": None,
        "extracted_patch": None,
        "error_message": None,
        "status_code": None,
        "api_call_successful": False
    }

        # 构建提示词
    prompt_parts = [
        "你是一个专业的C/C++代码修复助手。",
        f"请分析以下C/C++代码文件 '{file_path}' 的内容以及相关的GitHub Issue描述、测试用例，并生成一个修复该问题的补丁。\n"
    ]

    prompt_parts.append("GitHub Issue 描述:\n---")
    prompt_parts.append(issue_description)
    prompt_parts.append("---\n")

    if test_cases: # 新增测试用例部分
        prompt_parts.append("相关的测试用例:\n---")
        for i, tc in enumerate(test_cases):
            tc_name = tc.get("name", f"Test Case {i+1}")
            tc_desc = tc.get("description", "")
            tc_content = tc.get("content", "")
            if tc_content:
                prompt_parts.append(f"// {tc_name} ({tc_desc if tc_desc else 'N/A'})")
                prompt_parts.append(f"---BEGIN_TEST_CASE_{i+1}---")
                prompt_parts.append(tc_content)
                prompt_parts.append(f"---END_TEST_CASE_{i+1}---\n")
        prompt_parts.append("---\n")


    prompt_parts.append(f"以下是需要修复的原始文件 '{file_path}' 的完整内容：")
    prompt_parts.append("---BEGIN_ORIGINAL_FILE_CONTENT---")
    prompt_parts.append(file_content) # 原始文件内容
    prompt_parts.append("---END_ORIGINAL_FILE_CONTENT---\n")

    if context_files and len(context_files) > 1:
        context_str = "相关的上下文文件:\n"
        for rel_path, content in context_files.items():
            if rel_path != file_path: # 避免重复包含主文件
                preview = content[:1000] + ("..." if len(content) > 1000 else "")
                context_str += f"\n文件 '{rel_path}':\n```cpp\n{preview}\n```\n" # 使用cpp标记上下文代码块
        prompt_parts.append(context_str)

    if previous_error_log:
        prompt_parts.append(
            "上一次尝试生成的补丁在测试时产生了以下错误，请根据此错误信息改进你的补丁:\n---"
        )
        prompt_parts.append(previous_error_log)
        prompt_parts.append("---\n")

    prompt_parts.append(
        "请输出 unified diff 格式的修复补丁。\n"
        "重要提示：补丁中的所有行号（例如 `@@ -L1,C1 +L2,C2 @@` 中的 `L1` 和 `L2`）"
        "必须严格基于 `---BEGIN_ORIGINAL_FILE_CONTENT---` 和 `---END_ORIGINAL_FILE_CONTENT---` "
        "标记之间的原始文件内容进行计算。不要使用相对于整个输入文本的行号。\n"
        "确保补丁格式正确，可以直接应用到原始文件。"
    )
    if test_cases:
        prompt_parts.append("你的修复应该使提供的所有测试用例能够通过。")
    
    prompt = "\n".join(prompt_parts)
    result["prompt_sent"] = prompt

    if context_files and len(context_files) > 1:
        context_str = "相关的上下文文件:\n"
        for rel_path, content in context_files.items():
            if rel_path != file_path:
                preview = content[:1000] + ("..." if len(content) > 1000 else "")
                context_str += f"\n文件 '{rel_path}':\n```\n{preview}\n```\n"
        prompt += f"\n{context_str}\n"

    if previous_error_log:
        prompt += f"""
上一次尝试生成的补丁在测试时产生了以下错误，请根据此错误信息改进你的补丁:
---
{previous_error_log}
---
"""
    prompt += "\n请输出 unified diff 格式的修复补丁。确保补丁格式正确，可以直接应用到原始文件。\n"
    
    result["prompt_sent"] = prompt # Store the full prompt

    if stream:
        # Streaming part needs more work to return detailed info like non-streaming
        logger.warning("流式传输模式下，详细的 prompt/response 记录尚未完全实现。")
        # Simplified call for now, or adapt _handle_stream_response
        # For now, let's make it behave like non-streaming for return structure
        # extracted_content_stream = _handle_stream_response(...) 
        # result["raw_response"] = "Streamed response - full content needs aggregation"
        # result["extracted_patch"] = extract_diff_from_response(result["raw_response"]) if result["raw_response"] else None
        # result["api_call_successful"] = bool(result["extracted_patch"])
        # return result 
        # Temporarily disable stream for full logging or implement full return for stream
        logger.error("Stream mode is not fully supported with detailed trajectory logging yet. Please use non-stream mode.")
        result["error_message"] = "Stream mode not fully supported for detailed logging in this version."
        return result


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
        "max_tokens": 3000, # Increased max_tokens
        "temperature": 0.7,
        "stream": False # Forcing non-stream for now for detailed logging
    }
    
    logger.debug(f"API 请求负载: {json.dumps(payload, ensure_ascii=False)}")

    max_retries = 3 # As defined in previous version, can be from config
    retry_delay = 5 # Increased retry delay

    for attempt in range(max_retries):
        try:
            logger.info(f"发送 API 请求 (尝试 {attempt+1}/{max_retries})...")
            response = requests.post(
                DEEPSEEK_API_URL,
                headers=headers,
                json=payload,
                timeout=180  # Increased timeout
            )
            result["status_code"] = response.status_code
            
            logger.debug(f"API 响应状态码: {response.status_code}")
            # Try to get raw text first, then json, to handle non-JSON error responses better
            try:
                result["raw_response"] = response.text
                logger.debug(f"API 原始响应 (前500字符): {response.text[:500]}")
            except Exception as raw_e:
                logger.warning(f"无法获取原始文本响应: {raw_e}")
                result["raw_response"] = f"Error getting raw response: {raw_e}"


            if response.status_code == 429:
                retry_after = int(response.headers.get('Retry-After', retry_delay))
                logger.warning(f"API 速率限制，等待 {retry_after} 秒后重试...")
                result["error_message"] = f"Rate limited. Retry after {retry_after}s. Attempt {attempt+1}."
                time.sleep(retry_after)
                if attempt < max_retries -1 : continue
                else: break # Break if last attempt

            response.raise_for_status() # Raise HTTPError for bad responses (4xx or 5xx)
            
            data = response.json() # If raise_for_status didn't raise, we should have JSON
            # Log the JSON data as well if needed, but raw_response already has it
            # logger.debug(f"API 响应 JSON: {json.dumps(data, indent=2, ensure_ascii=False)}")

            raw_patch_content = _process_deepseek_response_content(data)
            
            if not raw_patch_content:
                logger.error("API 返回了空的补丁内容 (choices[0].message.content is empty)")
                result["error_message"] = "API returned empty patch content."
                # No need to retry if API call was successful but content is empty, unless it's a transient issue
                # For now, we break and return this result.
                break 
            
            result["api_call_successful"] = True # Mark as successful API call with content
            # The raw_patch_content is part of result["raw_response"] if it was JSON.
            # If we want to store just the 'message.content' separately:
            # result["ai_message_content"] = raw_patch_content 
            
            extracted_diff = extract_diff_from_response(raw_patch_content)
            
            if extracted_diff:
                logger.info(f"成功从 DeepSeek API 获取并提取补丁 ({len(extracted_diff)} 字符)")
                result["extracted_patch"] = extracted_diff
            else:
                logger.warning(f"API 返回的内容似乎不是有效的 diff 格式。将使用原始返回内容作为补丁。")
                result["extracted_patch"] = raw_patch_content # Fallback to raw content
            
            return result # Success, return

        except requests.exceptions.HTTPError as http_err:
            logger.error(f"HTTP 错误 (尝试 {attempt+1}/{max_retries}): {http_err}")
            result["error_message"] = f"HTTPError: {str(http_err)}. Response: {result['raw_response'][:500]}"
            if response.status_code == 500 or response.status_code == 503: # Server errors, retry
                 if attempt < max_retries - 1:
                    sleep_time = retry_delay * (2 ** attempt) # Exponential backoff
                    logger.info(f"将在 {sleep_time} 秒后重试")
                    time.sleep(sleep_time)
                    continue
            break # For other HTTP errors (like 400, 401, 403), don't retry unless specific
        except requests.exceptions.RequestException as req_err:
            logger.error(f"请求错误 (尝试 {attempt+1}/{max_retries}): {req_err}")
            result["error_message"] = f"RequestException: {str(req_err)}"
            if attempt < max_retries - 1:
                sleep_time = retry_delay * (2 ** attempt) # Exponential backoff
                logger.info(f"将在 {sleep_time} 秒后重试")
                time.sleep(sleep_time)
            else:
                logger.error("所有重试失败")
                break 
        except json.JSONDecodeError as json_err:
            logger.error(f"无法解析 API 响应 JSON (尝试 {attempt+1}/{max_retries}): {json_err}. 响应内容: {result['raw_response'][:500]}")
            result["error_message"] = f"JSONDecodeError: {str(json_err)}. Response: {result['raw_response'][:500]}"
            # If server returns non-JSON for an error, it might not be worth retrying without change
            break
        except Exception as e:
            logger.exception(f"处理 DeepSeek API 响应时发生未知错误 (尝试 {attempt+1}/{max_retries}): {e}")
            result["error_message"] = f"Unknown error: {str(e)}"
            break # Unknown error, break retry loop

    if not result["api_call_successful"] and not result["error_message"]:
        result["error_message"] = "All API call attempts failed or no patch content."
        
    return result

def generate_patch_with_enhanced_prompt(
    api_key: str,
    enhanced_prompt: str,
    stream: bool = False
) -> Dict[str, any]:
    """
    使用增强prompt生成补丁，基于现有的generate_patch_with_context实现
    
    :param api_key: DeepSeek API密钥
    :param enhanced_prompt: 完整的增强prompt
    :param stream: 是否使用流式传输
    :return: 字典，包含 "prompt_sent", "raw_response", "extracted_patch", "error_message", "status_code", "api_call_successful"
    """
    logger.info("使用增强prompt调用AI生成补丁...")
    
    # 初始化返回结果字典，保持与现有函数一致的结构
    result = {
        "prompt_sent": enhanced_prompt,
        "raw_response": None,
        "extracted_patch": None,
        "error_message": None,
        "status_code": None,
        "api_call_successful": False
    }

    if stream:
        logger.warning("流式传输模式下，详细的 prompt/response 记录尚未完全实现。")
        result["error_message"] = "Stream mode not fully supported for detailed logging in this version."
        return result

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    
    payload = {
        "model": "deepseek-coder",  # 使用 deepseek-coder 模型，更适合代码修复
        "messages": [
            {"role": "system", "content": "You are a professional C/C++ code repair expert specialized in creating patches in unified diff format."},
            {"role": "user", "content": enhanced_prompt}
        ],
        "max_tokens": 4000,  # 增加token限制以支持更复杂的修复
        "temperature": 0.7,  # 降低随机性，提高一致性
        "stream": False
    }
    
    logger.debug(f"API 请求负载: {json.dumps(payload, ensure_ascii=False)}")

    max_retries = 3
    retry_delay = 5

    for attempt in range(max_retries):
        try:
            logger.info(f"发送 API 请求 (尝试 {attempt+1}/{max_retries})...")
            response = requests.post(
                DEEPSEEK_API_URL,
                headers=headers,
                json=payload,
                timeout=120  # 2分钟超时
            )
            result["status_code"] = response.status_code
            
            logger.debug(f"API 响应状态码: {response.status_code}")
            
            try:
                result["raw_response"] = response.text
                logger.debug(f"API 原始响应 (前500字符): {response.text[:500]}")
            except Exception as raw_e:
                logger.warning(f"无法获取原始文本响应: {raw_e}")
                result["raw_response"] = f"Error getting raw response: {raw_e}"

            if response.status_code == 429:
                retry_after = int(response.headers.get('Retry-After', retry_delay))
                logger.warning(f"API 速率限制，等待 {retry_after} 秒后重试...")
                result["error_message"] = f"Rate limited. Retry after {retry_after}s. Attempt {attempt+1}."
                time.sleep(retry_after)
                if attempt < max_retries - 1:
                    continue
                else:
                    break

            response.raise_for_status()
            
            data = response.json()
            raw_patch_content = _process_deepseek_response_content(data)
            
            if not raw_patch_content:
                logger.error("API 返回了空的补丁内容")
                result["error_message"] = "API returned empty patch content."
                break 
            
            result["api_call_successful"] = True
            
            # 使用现有的extract_diff_from_response函数提取补丁
            extracted_diff = extract_diff_from_response(raw_patch_content)
            
            if extracted_diff:
                logger.info(f"成功从 DeepSeek API 获取并提取补丁 ({len(extracted_diff)} 字符)")
                result["extracted_patch"] = extracted_diff
            else:
                logger.warning(f"API 返回的内容似乎不是有效的 diff 格式。将使用原始返回内容作为补丁。")
                result["extracted_patch"] = raw_patch_content
            
            return result

        except requests.exceptions.HTTPError as http_err:
            logger.error(f"HTTP 错误 (尝试 {attempt+1}/{max_retries}): {http_err}")
            result["error_message"] = f"HTTPError: {str(http_err)}. Response: {result['raw_response'][:500] if result['raw_response'] else 'No response'}"
            if response.status_code in [500, 503]:  # 服务器错误，可以重试
                if attempt < max_retries - 1:
                    sleep_time = retry_delay * (2 ** attempt)
                    logger.info(f"将在 {sleep_time} 秒后重试")
                    time.sleep(sleep_time)
                    continue
            break
            
        except requests.exceptions.RequestException as req_err:
            logger.error(f"请求错误 (尝试 {attempt+1}/{max_retries}): {req_err}")
            result["error_message"] = f"RequestException: {str(req_err)}"
            if attempt < max_retries - 1:
                sleep_time = retry_delay * (2 ** attempt)
                logger.info(f"将在 {sleep_time} 秒后重试")
                time.sleep(sleep_time)
            else:
                logger.error("所有重试失败")
                break
                
        except json.JSONDecodeError as json_err:
            logger.error(f"无法解析 API 响应 JSON (尝试 {attempt+1}/{max_retries}): {json_err}")
            result["error_message"] = f"JSONDecodeError: {str(json_err)}. Response: {result['raw_response'][:500] if result['raw_response'] else 'No response'}"
            break
            
        except Exception as e:
            logger.exception(f"处理 DeepSeek API 响应时发生未知错误 (尝试 {attempt+1}/{max_retries}): {e}")
            result["error_message"] = f"Unknown error: {str(e)}"
            break

    if not result["api_call_successful"] and not result["error_message"]:
        result["error_message"] = "All API call attempts failed or no patch content."
        
    return result


def simple_api_call(api_key: str, prompt: str) -> dict:
    """
    简单的API调用，用于文件定位等任务
    基于现有实现，但简化为单轮对话
    """
    logger.info("执行简单API调用...")
    
    result = {
        "api_call_successful": False,
        "response": None,
        "error_message": None,
        "status_code": None
    }
    
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    
    payload = {
        "model": "deepseek-chat",
        "messages": [
            {"role": "user", "content": prompt}
        ],
        "max_tokens": 2000,
        "temperature": 0.7,
        "stream": False
    }
    
    try:
        response = requests.post(
            DEEPSEEK_API_URL,
            headers=headers,
            json=payload,
            timeout=60
        )
        
        result["status_code"] = response.status_code
        
        if response.status_code == 200:
            data = response.json()
            content = _process_deepseek_response_content(data)
            
            if content:
                result["api_call_successful"] = True
                result["response"] = content
                logger.info(f"简单API调用成功，响应长度: {len(content)}")
            else:
                result["error_message"] = "API returned empty content"
                logger.error("API返回空内容")
        else:
            result["error_message"] = f"API请求失败: {response.status_code} - {response.text}"
            logger.error(f"API请求失败: {response.status_code}")
            
    except requests.exceptions.RequestException as e:
        result["error_message"] = f"Request exception: {str(e)}"
        logger.error(f"API请求异常: {e}")
    except json.JSONDecodeError as e:
        result["error_message"] = f"JSON decode error: {str(e)}"
        logger.error(f"JSON解析错误: {e}")
    except Exception as e:
        result["error_message"] = f"Unexpected error: {str(e)}"
        logger.error(f"未预期错误: {e}")
        
    return result


def extract_patch_from_response(response_text: str) -> Optional[str]:
    """
    从AI响应中提取补丁内容
    这是对现有extract_diff_from_response函数的别名，保持接口一致性
    """
    return extract_diff_from_response(response_text)