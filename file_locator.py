import logging
import os
import re
from typing import List, Tuple, Dict, Optional
from collections import defaultdict

logger = logging.getLogger(__name__)

def find_file_to_fix(issue_details: dict, project_path: str) -> str | None:
    """
    使用多阶段策略定位需要修复的文件
    """
    # 1. 从所有可能的文本源中收集信息
    text_sources = [
        issue_details.get("title", ""),
        issue_details.get("body", "")
    ]
    
    # 添加评论内容
    comments = issue_details.get("comments", [])
    text_sources.extend(comments)
    
    # 合并所有文本用于分析
    all_text = "\n".join([t for t in text_sources if t])
    
    # 2. 提取代码示例和潜在问题描述
    code_blocks = extract_code_blocks(all_text)
    potential_files = extract_file_paths(all_text)
    
    logger.info(f"在 Issue 中找到 {len(code_blocks)} 个代码块和 {len(potential_files)} 个潜在文件路径")
    
    # 3. 多阶段文件定位策略
    
    # 阶段1: 直接路径匹配
    for file_path in potential_files:
        normalized_path = file_path.lstrip('/\\')
        full_path = os.path.join(project_path, normalized_path)
        if os.path.isfile(full_path):
            logger.info(f"阶段1-直接路径匹配成功: {full_path}")
            return normalized_path
    
    # 阶段2: 文件名匹配 (忽略路径)
    for file_path in potential_files:
        base_name = os.path.basename(file_path)
        for root, _, files in os.walk(project_path):
            if base_name in files:
                found_path = os.path.join(root, base_name)
                rel_path = os.path.relpath(found_path, project_path)
                logger.info(f"阶段2-文件名匹配成功: {found_path}")
                return rel_path
    
    # 阶段3: 扩展名变体匹配
    for file_path in potential_files:
        base_name = os.path.basename(file_path)
        name, ext = os.path.splitext(base_name)
        
        variants = []
        if ext.lower() == '.h':
            variants.append(f"{name}.hpp")
        elif ext.lower() == '.hpp':
            variants.append(f"{name}.h")
        # 可以添加更多变体规则
        
        for variant in variants:
            for root, _, files in os.walk(project_path):
                if variant in files:
                    found_path = os.path.join(root, variant)
                    rel_path = os.path.relpath(found_path, project_path)
                    logger.info(f"阶段3-扩展名变体匹配成功: {found_path}")
                    return rel_path
    
    # 阶段4: 根据代码内容或关键函数名匹配
    if code_blocks:
        keywords = extract_code_keywords(code_blocks)
        candidates = find_files_by_keywords(project_path, keywords)
        
        if candidates:
            # 返回最匹配的文件
            best_match = candidates[0][0]
            logger.info(f"阶段4-代码内容匹配成功: {best_match} (分数: {candidates[0][1]})")
            return os.path.relpath(best_match, project_path)
    
    # 阶段5: 查找主要头文件或核心文件
    core_files = find_core_files(project_path)
    if core_files:
        best_match = core_files[0]
        logger.info(f"阶段5-核心文件匹配: {best_match}")
        return os.path.relpath(best_match, project_path)
    
    logger.warning(f"未能在项目 {project_path} 中定位到需要修复的文件")
    return None

def extract_code_blocks(text: str) -> List[str]:
    """从 Markdown 文本中提取代码块"""
    # 匹配 Markdown 代码块: ```[language] ... ```
    code_pattern = re.compile(r'```(?:\w+)?\s*\n(.*?)\n```', re.DOTALL)
    code_blocks = code_pattern.findall(text)
    
    # 匹配缩进代码块 (4个空格或制表符开头的连续行)
    indented_pattern = re.compile(r'(?:^|\n)((?:[ \t]{4}.*\n)+)', re.MULTILINE)
    indented_blocks = indented_pattern.findall(text)
    code_blocks.extend([block.replace('\n    ', '\n') for block in indented_blocks])
    
    return code_blocks

def extract_file_paths(text: str) -> List[str]:
    """从文本中提取可能的文件路径"""
    # 匹配常见的 C/C++ 文件路径
    file_patterns = [
        # 匹配 .h, .hpp, .c, .cpp, .cc 等文件
        r'(?:^|\s|"|\()([a-zA-Z0-9_/\\.-]+\.(?:h|hpp|c|cpp|cc|cxx|hxx))(?:$|\s|"|\))',
        # 匹配头文件引用格式
        r'#include\s+[<"]([^>"]+)[>"]',
        # 匹配可能的类名(可能是文件名)
        r'class\s+([A-Z][a-zA-Z0-9_]+)'
    ]
    
    potential_files = []
    for pattern in file_patterns:
        matches = re.findall(pattern, text)
        potential_files.extend(matches)
    
    # 移除重复项并返回
    return list(set(potential_files))

def extract_code_keywords(code_blocks: List[str]) -> List[str]:
    """从代码块中提取关键标识符"""
    keywords = set()
    
    for block in code_blocks:
        # 提取函数名、类名、变量名等
        # 函数调用 name(...)
        func_calls = re.findall(r'(\w+)\s*\(', block)
        keywords.update(func_calls)
        
        # 类名 (通常首字母大写)
        class_names = re.findall(r'class\s+([A-Z][a-zA-Z0-9_]*)', block)
        keywords.update(class_names)
        
        # 方法调用 obj.method()
        methods = re.findall(r'\.(\w+)\s*\(', block)
        keywords.update(methods)
        
        # 其他特殊标识符，针对具体问题
        special_terms = ["parse_args", "ArgumentParser", "add_argument", "get"]
        for term in special_terms:
            if term in block:
                keywords.add(term)
    
    # 过滤掉太短或通用的关键词
    filtered = []
    common_words = {'if', 'else', 'for', 'while', 'int', 'char', 'bool', 'void', 'return'}
    for kw in keywords:
        if len(kw) > 2 and kw not in common_words:
            filtered.append(kw)
    
    return filtered

def find_files_by_keywords(project_path: str, keywords: List[str]) -> List[Tuple[str, int]]:
    """根据关键词在项目中查找可能相关的文件"""
    if not keywords:
        return []
        
    candidates = []
    
    # 仅搜索可能的C/C++源文件
    for root, _, files in os.walk(project_path):
        for file in files:
            if file.lower().endswith(('.h', '.hpp', '.c', '.cpp', '.cc', '.cxx')):
                try:
                    file_path = os.path.join(root, file)
                    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                        content = f.read()
                    
                    # 计算文件与关键词的相关性分数
                    score = 0
                    for kw in keywords:
                        count = content.count(kw)
                        if count > 0:
                            # 关键词出现越多，分数越高
                            score += count * 10
                            
                            # 如果关键词在文件名中，额外加分
                            if kw.lower() in file.lower():
                                score += 50
                    
                    # 针对特定的issue类型，额外增加相关性
                    if "parse_args" in keywords and "parse_args" in content:
                        score += 100
                    if "ArgumentParser" in keywords and "ArgumentParser" in content:
                        score += 100
                    
                    if score > 0:
                        candidates.append((file_path, score))
                except Exception as e:
                    logger.error(f"读取文件 {file} 时出错: {e}")
    
    # 按相关性分数排序
    candidates.sort(key=lambda x: x[1], reverse=True)
    return candidates

def find_core_files(project_path: str) -> List[str]:
    """查找项目中可能的核心文件"""
    # 寻找可能的主要头文件
    core_patterns = [
        # 项目名称相关文件
        lambda proj: f"{os.path.basename(proj)}.h",
        lambda proj: f"{os.path.basename(proj)}.hpp",
        # 核心类文件
        "main.h", "main.hpp", "core.h", "core.hpp",
        "argparse.h", "argparse.hpp",  # 针对argparse项目
    ]
    
    project_name = os.path.basename(os.path.normpath(project_path))
    
    candidates = []
    
    for pattern in core_patterns:
        if callable(pattern):
            pattern = pattern(project_name)
            
        for root, _, files in os.walk(project_path):
            if pattern in files:
                candidates.append(os.path.join(root, pattern))
    
    return candidates

def collect_context_files(project_path: str, main_file_path: str) -> Dict[str, str]:
    """收集与主文件相关的上下文文件"""
    context_files = {}
    
    # 读取主文件内容
    try:
        with open(os.path.join(project_path, main_file_path), 'r', encoding='utf-8', errors='ignore') as f:
            main_content = f.read()
        context_files[main_file_path] = main_content
    except Exception as e:
        logger.error(f"读取主文件失败: {e}")
        return context_files
    
    # 提取包含的头文件
    include_pattern = re.compile(r'#include\s+[<"]([^>"]+)[>"]')
    includes = include_pattern.findall(main_content)
    
    # 在项目中查找这些头文件
    for inc in includes:
        for root, _, files in os.walk(project_path):
            if inc in files:
                inc_path = os.path.relpath(os.path.join(root, inc), project_path)
                try:
                    with open(os.path.join(project_path, inc_path), 'r', encoding='utf-8', errors='ignore') as f:
                        inc_content = f.read()
                    context_files[inc_path] = inc_content
                except Exception as e:
                    logger.error(f"读取包含文件 {inc_path} 失败: {e}")
                break
    
    # 限制上下文文件数量，避免太大
    max_context_files = 5
    if len(context_files) > max_context_files:
        # 保留主文件和几个最相关的文件
        sorted_files = sorted(list(context_files.keys()))
        files_to_keep = [main_file_path] + sorted_files[:max_context_files-1]
        context_files = {k: context_files[k] for k in files_to_keep}
    
    return context_files