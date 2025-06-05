import logging
import os
import re
from typing import List, Tuple, Dict, Optional, Set
from collections import defaultdict
import json
from datetime import datetime, timezone
import ai_handler
logger = logging.getLogger(__name__)

# Precompile common regex patterns
R_CODE_BLOCK_MARKDOWN = re.compile(r'```(?:\w+)?\s*\n(.*?)\n```', re.DOTALL)
R_CODE_BLOCK_INDENTED = re.compile(r'(?:^|\n)((?:[ \t]{4}[^\n]*\n)+)', re.MULTILINE) # Ensure it captures lines starting with 4 spaces
R_INCLUDE_DIRECTIVE = re.compile(r'#include\s+[<"]([^>"]+)[>"]')
R_FILE_PATH_LIKE = re.compile(
    r'\b([a-zA-Z0-9_./\\-]+\.(?:h|hpp|c|cpp|cc|cxx|hxx|inl|tcc))\b', 
    re.IGNORECASE
)
R_CLASS_DECL = re.compile(r'\bclass\s+([a-zA-Z_][a-zA-Z0-9_]*)\b')
R_NAMESPACE_ACCESS = re.compile(r'\b([a-zA-Z_][a-zA-Z0-9_]*)::([a-zA-Z_][a-zA-Z0-9_]+)\b')

# 在 file_locator.py 开头添加导入


def find_file_to_fix_with_ai(issue_details: dict, project_path: str, test_patch_diff: str = "", 
                            ai_api_key: str = None, enable_ai: bool = True) -> tuple:
    """
    使用AI增强的文件定位策略
    返回: (best_file_path, ai_analysis, localization_details)
    """
    
    # 如果启用AI且有API密钥，先尝试AI定位
    ai_analysis = None
    if enable_ai and ai_api_key:
        try:
            ai_analysis = ai_locate_target_file(issue_details, project_path, test_patch_diff, ai_api_key)
            logger.info(f"AI文件定位结果: {ai_analysis}")
        except Exception as e:
            logger.warning(f"AI文件定位失败，回退到传统方法: {e}")
    
    # 执行传统的多阶段定位
    traditional_result = find_file_to_fix(issue_details, project_path)
    
    # 融合AI和传统方法的结果
    final_result = merge_ai_and_traditional_results(ai_analysis, traditional_result, project_path)
    
    return final_result, ai_analysis, []  # localization_details暂时返回空列表

def ai_locate_target_file(issue_details: dict, project_path: str, test_patch_diff: str, api_key: str) -> dict:
    """使用AI分析并定位目标文件"""
    
    # 扫描项目结构
    logger.info("扫描项目结构...")
    project_structure = scan_project_files_for_ai(project_path)
    logger.info(f"项目结构扫描完成，找到 {len(project_structure.split('\\n'))} 个文件")
    
    # 构建AI分析prompt
    prompt = build_file_localization_prompt(issue_details, test_patch_diff, project_structure)
    logger.info(f"构建AI定位prompt，长度: {len(prompt)}")
    
    
    logger.info("开始调用AI API...")
    ai_result = ai_handler.simple_api_call(api_key, prompt)
    
    if not ai_result.get('api_call_successful'):
        error_msg = ai_result.get('error_message', 'Unknown error')
        logger.error(f"AI API调用失败: {error_msg}")
        raise Exception(f"AI API调用失败: {error_msg}")
    
    logger.info("AI API调用成功，开始解析响应...")
    
    # 解析AI响应 - 改进的解析逻辑
    try:
        raw_response = ai_result['response']
        logger.debug(f"AI原始响应: {raw_response[:500]}...")
        
        # 清理响应内容，移除代码块标记
        cleaned_response = raw_response.strip()
        
        # 移除可能的代码块标记
        if cleaned_response.startswith('```json'):
            cleaned_response = cleaned_response[7:]  # 移除 ```json
        elif cleaned_response.startswith('```'):
            cleaned_response = cleaned_response[3:]   # 移除 ```
            
        if cleaned_response.endswith('```'):
            cleaned_response = cleaned_response[:-3]  # 移除结尾的 ```
            
        cleaned_response = cleaned_response.strip()
        
        # 尝试提取JSON部分
        json_start = cleaned_response.find('{')
        json_end = cleaned_response.rfind('}') + 1
        
        if json_start >= 0 and json_end > json_start:
            json_str = cleaned_response[json_start:json_end]
            logger.debug(f"提取的JSON: {json_str[:200]}...")
            ai_analysis = json.loads(json_str)
        else:
            # 如果没有找到JSON，尝试直接解析整个响应
            logger.warning("未找到明确的JSON边界，尝试直接解析")
            ai_analysis = json.loads(cleaned_response)
            
        # 验证AI返回的结构
        required_keys = ['target_file', 'confidence', 'reasoning', 'fix_strategy']
        missing_keys = [key for key in required_keys if key not in ai_analysis]
        if missing_keys:
            logger.error(f"AI返回缺少必需字段: {missing_keys}")
            raise ValueError(f"AI返回格式不完整，缺少: {missing_keys}")
            
        logger.info(f"AI分析成功解析: 目标文件={ai_analysis.get('target_file')}, 置信度={ai_analysis.get('confidence')}")
        return ai_analysis
        
    except (json.JSONDecodeError, ValueError) as e:
        logger.error(f"AI响应解析失败: {e}")
        logger.error(f"原始响应内容: {ai_result['response'][:1000]}...")
        logger.error(f"清理后内容: {cleaned_response[:1000] if 'cleaned_response' in locals() else 'N/A'}...")
        raise

def scan_project_files_for_ai(project_path: str, max_files: int = 30) -> str:
    """扫描项目文件结构，为AI提供上下文"""
    
    file_info = []
    file_count = 0
    
    # 优先级目录（按重要性排序）
    priority_dirs = ['src', 'include', 'lib', 'source', '', 'core']
    
    for priority_dir in priority_dirs:
        if file_count >= max_files:
            break
            
        search_dir = os.path.join(project_path, priority_dir) if priority_dir else project_path
        if not os.path.exists(search_dir):
            continue
            
        # 只遍历当前目录和一级子目录
        max_depth = 1 if priority_dir else 2
        
        for root, dirs, files in os.walk(search_dir):
            # 限制搜索深度
            level = root.replace(search_dir, '').count(os.sep)
            if level >= max_depth:
                dirs[:] = []  # 不继续深入
                continue
                
            # 跳过无关目录
            dirs[:] = [d for d in dirs if d not in ['.git', 'build', '__pycache__', 'node_modules', 'cmake-build']]
            
            for file in files:
                if file_count >= max_files:
                    break
                    
                # 只关注源代码文件
                if not file.lower().endswith(('.c', '.cpp', '.cc', '.cxx', '.h', '.hpp', '.hxx')):
                    continue
                    
                file_path = os.path.join(root, file)
                rel_path = os.path.relpath(file_path, project_path)
                
                try:
                    # 获取文件基本信息
                    size = os.path.getsize(file_path)
                    
                    # 读取文件开头几行来了解内容
                    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                        lines = []
                        for i, line in enumerate(f):
                            if i >= 10:  # 只读前10行
                                break
                            line = line.strip()
                            if line and not line.startswith('//') and not line.startswith('/*'):
                                lines.append(line)
                        preview = '\n'.join(lines[:5])  # 取前5个有效行
                    
                    # 分析文件类型和重要性
                    file_type = analyze_file_importance(rel_path, preview)
                    
                    file_info.append(f"{rel_path} ({size}B) - {file_type}: {preview[:100]}...")
                    file_count += 1
                    
                except Exception as e:
                    logger.debug(f"读取文件信息失败 {rel_path}: {e}")
                    file_info.append(f"{rel_path} - 无法读取")
                    file_count += 1
    
    return '\n'.join(file_info)

def analyze_file_importance(file_path: str, content_preview: str) -> str:
    """分析文件的类型和重要性"""
    path_lower = file_path.lower()
    
    # 测试文件
    if 'test' in path_lower:
        return "TEST"
    
    # 头文件
    if file_path.endswith(('.h', '.hpp', '.hxx')):
        if 'main' in path_lower or 'core' in path_lower:
            return "HEADER_CORE"
        return "HEADER"
    
    # 源文件
    if file_path.endswith(('.c', '.cpp', '.cc', '.cxx')):
        if 'main' in path_lower:
            return "SOURCE_MAIN"
        if any(keyword in content_preview.lower() for keyword in ['main(', 'int main', 'void main']):
            return "SOURCE_MAIN"
        return "SOURCE"
    
    return "OTHER"

def build_file_localization_prompt(issue_details: dict, test_patch_diff: str, project_structure: str) -> str:
    """构建文件定位的AI prompt"""
    
    problem_statement = issue_details.get('problem_statement', issue_details.get('body', ''))
    hints_text = issue_details.get('hints_text', '')
    title = issue_details.get('title', '')
    
    prompt = f"""你是一个专业的C/C++代码分析专家。请分析以下信息并精确定位需要修复的源代码文件。

## 问题标题
{title}

## 问题描述
{problem_statement}

## 提示信息
{hints_text}

## 测试变更差异（显示期望的修复行为）
```diff
{test_patch_diff}
项目文件结构
{project_structure}

分析要求
请仔细分析以上信息，特别注意：

问题描述的关键词：从问题描述中提取功能相关的关键词
测试变更线索：分析测试变更暴露的问题域和涉及的函数/API
文件名匹配：根据问题域找到最可能的源文件
排除测试文件：优先选择实现文件而非测试文件
代码逻辑：考虑哪个文件最可能包含需要修复的逻辑
输出格式
请以严格的JSON格式回答：
{{
    "target_file": "需要修复的文件相对路径",
    "confidence": 8,
    "reasoning": "详细说明选择这个文件的理由，包括从问题描述、测试变更和文件结构中得出的线索",
    "fix_strategy": "基于问题描述和测试变更，简述修复策略",
    "test_insights": "从测试变更中获得的关键信息",
    "alternative_files": ["可能的备选文件1", "可能的备选文件2"]
}}
注意：

confidence为1-10的整数，10表示非常确定

target_file必须是项目结构中实际存在的文件

优先选择源文件(.c/.cpp)而非头文件(.h/.hpp)

不要选择测试文件，除非问题明确指向测试代码 """
    return prompt
def merge_ai_and_traditional_results(ai_analysis: dict, traditional_result: str, project_path: str) -> str: 
    """融合AI和传统方法的结果"""
    if not ai_analysis:
        logger.info("无AI分析结果，使用传统方法结果")
        return traditional_result

    ai_target = ai_analysis.get('target_file', '').replace('\\', os.sep).replace('/', os.sep)
    ai_confidence = ai_analysis.get('confidence', 0)

    # 验证AI选择的文件是否存在
    if ai_target:
        ai_target_abs = os.path.join(project_path, ai_target)
        if os.path.exists(ai_target_abs):
            # 如果AI的置信度很高，优先使用AI结果
            if ai_confidence >= 7:
                logger.info(f"AI高置信度({ai_confidence})选择: {ai_target}")
                return ai_target
            # 如果传统方法没有结果，使用AI结果
            elif not traditional_result:
                logger.info(f"传统方法无结果，使用AI选择: {ai_target}")
                return ai_target
            # 如果两个结果相同，使用该结果
            elif traditional_result and os.path.normpath(ai_target) == os.path.normpath(traditional_result):
                logger.info(f"AI和传统方法一致选择: {ai_target}")
                return ai_target
            else:
                # 结果不同时，根据置信度决定
                if ai_confidence >= 5:
                    logger.info(f"AI中等置信度({ai_confidence})，优先AI选择: {ai_target} (传统:{traditional_result})")
                    return ai_target
                else:
                    logger.info(f"AI低置信度({ai_confidence})，使用传统方法: {traditional_result} (AI:{ai_target})")
                    return traditional_result
        else:
            logger.warning(f"AI选择的文件不存在: {ai_target}，使用传统方法结果")

    return traditional_result

def find_file_to_fix(issue_details: dict, project_path: str) -> Optional[str]:
    """
    Uses a multi-stage strategy to locate the file to be fixed, with weighted scoring.
    Returns the relative path to the best candidate file or None.
    """
    all_text_sources = [
        issue_details.get("title", ""),
        issue_details.get("body", ""),
    ] + issue_details.get("comments", [])
    full_issue_text = "\n".join(filter(None, all_text_sources))

    code_blocks = extract_code_blocks(full_issue_text)
    # Explicitly mentioned file paths or #includes in the issue text
    mentioned_file_paths = extract_mentioned_file_paths(full_issue_text)
    # File name clues derived from class names or namespace usage in code blocks
    derived_file_clues = extract_derived_file_clues(code_blocks)

    all_potential_file_references = list(set(mentioned_file_paths + derived_file_clues))
    
    logger.info(f"Found {len(code_blocks)} code blocks in issue.")
    logger.info(f"Potential file references from issue: {all_potential_file_references}")

    # Store candidates with their scores: Dict[rel_path, float]
    candidate_scores: Dict[str, float] = defaultdict(float)
    # Log for debugging and trajectory
    localization_details: List[Dict[str, any]] = []

    # --- Stage 1: Direct Path Match (Highest Confidence) ---
    for path_ref in mentioned_file_paths: # Only use explicitly mentioned paths here
        normalized_ref = path_ref.lstrip('/\\')
        abs_path = os.path.join(project_path, normalized_ref)
        if os.path.isfile(abs_path):
            rel_path = os.path.relpath(abs_path, project_path)
            candidate_scores[rel_path] += 100.0
            localization_details.append({
                "stage": "Direct Path Match", "reference": path_ref, 
                "found": rel_path, "score_increase": 100.0
            })
            logger.info(f"Direct Path Match: '{rel_path}' from '{path_ref}' (Score +100.0)")

    # --- Stage 2: File Name Match (from all references) ---
    for file_ref in all_potential_file_references:
        base_name = os.path.basename(file_ref)
        if not base_name or '.' not in base_name: continue # Ensure it looks like a filename

        for root, _, files in os.walk(project_path):
            if ".git" in root or "build" in root.lower(): continue
            if base_name in files:
                abs_path = os.path.join(root, base_name)
                rel_path = os.path.relpath(abs_path, project_path)
                score_increase = 75.0
                if is_test_file(rel_path): score_increase *= 0.5
                candidate_scores[rel_path] += score_increase
                localization_details.append({
                    "stage": "File Name Match", "reference": base_name, 
                    "found": rel_path, "score_increase": score_increase
                })
                logger.info(f"File Name Match: '{rel_path}' from '{base_name}' (Score +{score_increase:.1f})")

    # --- Stage 3: Extension Variant Match (.h <-> .hpp) ---
    for file_ref in all_potential_file_references:
        base_name = os.path.basename(file_ref)
        name, ext = os.path.splitext(base_name)
        if not name or not ext: continue
        
        variants = []
        if ext.lower() == ".h": variants.append(f"{name}.hpp")
        elif ext.lower() == ".hpp": variants.append(f"{name}.h")

        for variant_name in variants:
            for root, _, files in os.walk(project_path):
                if ".git" in root or "build" in root.lower(): continue
                if variant_name in files:
                    abs_path = os.path.join(root, variant_name)
                    rel_path = os.path.relpath(abs_path, project_path)
                    score_increase = 60.0
                    if is_test_file(rel_path): score_increase *= 0.5
                    candidate_scores[rel_path] += score_increase
                    localization_details.append({
                        "stage": "Extension Variant Match", "original": base_name, "variant": variant_name,
                        "found": rel_path, "score_increase": score_increase
                    })
                    logger.info(f"Extension Variant: '{rel_path}' for '{base_name}' (Score +{score_increase:.1f})")

    # --- Stage 4: Keyword-based Content Match ---
    if code_blocks or all_potential_file_references: # Use file refs as keywords too
        keywords = extract_code_keywords(code_blocks, all_potential_file_references)
        if keywords:
            logger.info(f"Keywords for content search: {keywords[:10]}...")
            keyword_matches = find_files_by_keywords(project_path, keywords)
            for path, score, reason in keyword_matches:
                candidate_scores[path] += score # Score from find_files_by_keywords is already weighted
                localization_details.append({
                    "stage": "Keyword Content Match", "keywords_used_count": len(keywords),
                    "found": path, "score_increase": score, "reason": reason
                })
                logger.info(f"Keyword Match: '{path}' (Score +{score:.1f}, Reason: {reason})")
    
    # --- Stage 5: Core File Heuristics ---
    # This stage primarily boosts scores of existing candidates if they are core, or introduces them with a moderate score.
    core_files_found = find_project_core_files(project_path)
    for core_file_rel_path, reason in core_files_found:
        score_increase = 40.0 # Base score for being a core file
        if core_file_rel_path in candidate_scores: # Boost if already a candidate
            score_increase += 20.0 
        if is_test_file(core_file_rel_path): score_increase *= 0.3 # Heavy penalty if a "core" file is in test
        
        candidate_scores[core_file_rel_path] += score_increase
        localization_details.append({
            "stage": "Core File Heuristic", "reason": reason,
            "found": core_file_rel_path, "score_increase": score_increase
        })
        logger.info(f"Core File Heuristic: '{core_file_rel_path}' ({reason}) (Score +{score_increase:.1f})")

    if not candidate_scores:
        logger.warning("No candidate files found after all stages.")
        # As a last resort, if find_project_core_files found something, pick the first one.
        if core_files_found:
            logger.warning(f"Falling back to first core file found: {core_files_found[0][0]}")
            return core_files_found[0][0]
        return None

    # --- Final Selection ---
    # Sort by score, then prefer non-test files, then shorter paths (more likely to be root/include)
    sorted_candidates = sorted(
        candidate_scores.items(),
        key=lambda item: (
            item[1],  # Score
            is_test_file(item[0]),  # False (non-test) comes before True (test)
            len(item[0].split(os.sep)), # Path depth
            len(item[0]) # Path length
        ),
        reverse=True # Highest score first
    )
    
    logger.info(f"Final candidate scores (path, score): {[(p, s) for p, s in sorted_candidates]}")
    # The sorting key already handles preference, so the first element is the best
    best_candidate_path, best_score = sorted_candidates[0]
    
    # Log all details for trajectory (can be large, consider summarizing)
    # For now, let's assume main.py will handle what to put in trajectory.
    # This function could return (best_candidate_path, localization_details)
    
    logger.info(f"Best candidate selected: '{best_candidate_path}' with score {best_score:.1f}")
    return best_candidate_path

def is_test_file(file_path: str) -> bool:
    """Checks if a file path seems to be a test file."""
    path_parts = file_path.lower().split(os.sep)
    test_indicators = {"test", "tests", "unittest", "example", "examples", "sample", "samples", "demo"}
    filename = os.path.basename(file_path).lower()
    
    if any(indicator in path_parts for indicator in test_indicators):
        return True
    if filename.startswith("test_") or filename.endswith("_test.cpp") or filename.endswith("_test.h"):
        return True
    return False

def extract_code_blocks(text: str) -> List[str]:
    """Extracts code blocks from Markdown text (fenced and indented)."""
    blocks = R_CODE_BLOCK_MARKDOWN.findall(text)
    # For indented blocks, need to clean up leading spaces per line
    indented_matches = R_CODE_BLOCK_INDENTED.finditer(text)
    for match in indented_matches:
        indented_block_raw = match.group(1)
        lines = indented_block_raw.splitlines()
        cleaned_lines = []
        for line in lines:
            if line.startswith('    '):
                cleaned_lines.append(line[4:])
            elif line.startswith('\t'): # Also handle tab-indented
                cleaned_lines.append(line[1:])
            else:
                cleaned_lines.append(line) # Should not happen if regex is correct
        if cleaned_lines:
            blocks.append("\n".join(cleaned_lines))
    return blocks

def extract_mentioned_file_paths(text: str) -> List[str]:
    """Extracts file paths explicitly mentioned or #included in text."""
    paths: Set[str] = set()
    for match in R_INCLUDE_DIRECTIVE.finditer(text):
        paths.add(match.group(1))
    for match in R_FILE_PATH_LIKE.finditer(text):
        # Further validation to reduce noise from R_FILE_PATH_LIKE
        path_candidate = match.group(1)
        # Heuristic: if it contains a slash or starts with typical relative path indicators
        if '/' in path_candidate or '\\' in path_candidate or \
           path_candidate.startswith('.') or path_candidate.startswith('src/') or \
           path_candidate.startswith('include/'):
            paths.add(path_candidate)
        # Or if it's a simple filename.ext that's likely not part of a sentence.
        elif '.' in os.path.basename(path_candidate) and \
             (match.start() == 0 or text[match.start()-1].isspace()) and \
             (match.end() == len(text) or text[match.end()].isspace() or text[match.end()] in ')"\''):
            paths.add(path_candidate)
            
    return list(paths)

def extract_derived_file_clues(code_blocks: List[str]) -> List[str]:
    """Derives potential file name clues from class names and namespace usage in code blocks."""
    clues: Set[str] = set()
    for block in code_blocks:
        for match in R_CLASS_DECL.finditer(block): # e.g. class MyClass
            class_name = match.group(1)
            clues.add(f"{class_name}.h")
            clues.add(f"{class_name}.hpp")
            # Simple CamelCase to snake_case for another variant
            snake_name = re.sub(r'(?<!^)(?=[A-Z])', '_', class_name).lower()
            if snake_name != class_name.lower():
                clues.add(f"{snake_name}.h")
                clues.add(f"{snake_name}.hpp")
        
        for match in R_NAMESPACE_ACCESS.finditer(block): # e.g. my_namespace::MyClass
            namespace = match.group(1)
            # entity = match.group(2)
            if namespace.lower() not in ["std", "boost", "cv", "qt", "detail", "internal"]: # Filter common/generic
                clues.add(f"{namespace}.h") # Namespace itself could be a file
                clues.add(f"{namespace}.hpp")
    return list(clues)

def extract_code_keywords(code_blocks: List[str], file_references: List[str]) -> List[str]:
    """Extracts keywords from code blocks and file references."""
    keywords: Set[str] = set()
    common_cpp_tokens = { # Expanded set
        'if', 'else', 'for', 'while', 'int', 'char', 'bool', 'void', 'return', 'class', 
        'struct', 'template', 'typename', 'const', 'static', 'public', 'private', 
        'protected', 'namespace', 'using', 'std', 'include', 'define', 'nullptr',
        'auto', 'decltype', 'virtual', 'override', 'final', 'explicit', 'inline',
        'unsigned', 'signed', 'short', 'long', 'float', 'double', 'true', 'false',
        'try', 'catch', 'throw', 'new', 'delete', 'this', 'operator', 'friend',
        'static_cast', 'dynamic_cast', 'reinterpret_cast', 'const_cast', 'enum',
        'size_t', 'int8_t', 'uint8_t', 'int16_t', 'uint16_t', 'int32_t', 'uint32_t',
        'int64_t', 'uint64_t', 'string', 'vector', 'map', 'set', 'list', 'iostream'
    }

    # Add parts of file references as keywords
    for ref in file_references:
        name_part = os.path.splitext(os.path.basename(ref))[0]
        # Split by common delimiters and add parts if they are somewhat unique
        sub_parts = re.split(r'[_/-]', name_part)
        for part in sub_parts:
            if len(part) > 3 and part.lower() not in common_cpp_tokens:
                keywords.add(part)

    for block in code_blocks:
        # Identifiers (function names, variable names, class names used)
        # A more robust way is needed for true AST parsing, but regex can approximate
        identifiers = re.findall(r'\b([a-zA-Z_][a-zA-Z0-9_]{3,})\b', block) # Min length 4
        for ident in identifiers:
            if ident.lower() not in common_cpp_tokens and not ident.isupper(): # Avoid ALL_CAPS macros for now
                keywords.add(ident)
        
        # Specific API calls mentioned in issues, e.g., "parse_args", "ArgumentParser"
        # These could be passed in or hardcoded if very common for the target domain
        domain_specific_terms = ["parse_args", "ArgumentParser", "add_argument", "get", "scan"]
        for term in domain_specific_terms:
            if term in block:
                keywords.add(term)
                
    return list(k for k in keywords if k) # Ensure no empty strings

def find_files_by_keywords(project_path: str, keywords: List[str]) -> List[Tuple[str, float, str]]:
    """Finds files by keywords, returning (rel_path, score_contribution, reason)."""
    if not keywords: return []
    
    keyword_matches: List[Tuple[str, float, str]] = []
    # Score weights
    W_CONTENT_MATCH = 10.0
    W_FILENAME_MATCH = 30.0 # Higher weight if keyword is part of filename
    W_IMPORTANT_KEYWORD = 2.0 # Multiplier for important keywords in content

    # Keywords that are particularly indicative if found in content
    important_keywords_set = {"argumentparser", "parse_args", "add_argument"} # lowercase

    for root, _, files in os.walk(project_path):
        if ".git" in root or "build" in root.lower() or "cmake-build" in root.lower():
            continue

        for file_name in files:
            if not file_name.lower().endswith(('.h', '.hpp', '.c', '.cpp', '.cc', '.cxx', '.inl', '.tcc')):
                continue
            
            abs_path = os.path.join(root, file_name)
            rel_path = os.path.relpath(abs_path, project_path)
            current_file_score = 0.0
            match_details = []

            try:
                with open(abs_path, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read()
                
                content_lower = content.lower()
                filename_lower_no_ext = os.path.splitext(file_name.lower())[0]

                for kw in keywords:
                    kw_lower = kw.lower()
                    if not kw_lower: continue

                    # Keyword in content
                    count = content_lower.count(kw_lower)
                    if count > 0:
                        score_add = count * W_CONTENT_MATCH
                        if kw_lower in important_keywords_set:
                            score_add *= W_IMPORTANT_KEYWORD
                        current_file_score += score_add
                        match_details.append(f"'{kw}' in content ({count}x)")
                    
                    # Keyword in filename (without extension)
                    if kw_lower in filename_lower_no_ext:
                        current_file_score += W_FILENAME_MATCH
                        match_details.append(f"'{kw}' in filename")
                
                if current_file_score > 0:
                    if is_test_file(rel_path):
                        current_file_score *= 0.3 # Penalize test files more heavily in keyword search
                        match_details.append("Test file penalty")
                    
                    if current_file_score > 1.0: # Threshold to add
                        keyword_matches.append((rel_path, current_file_score, "; ".join(match_details)))
            except Exception as e:
                logger.debug(f"Error reading/processing {abs_path} for keywords: {e}")
                
    keyword_matches.sort(key=lambda x: x[1], reverse=True)
    return keyword_matches[:15] # Return top N candidates from keyword search

def find_project_core_files(project_path: str) -> List[Tuple[str, str]]:
    """
    Heuristically finds core project files.
    Returns a list of (relative_path, reason_string).
    """
    core_files: Dict[str, str] = {} # rel_path -> reason
    project_name = os.path.basename(os.path.normpath(project_path)).lower()
    
    # Common core filenames and patterns
    core_filename_patterns = [
        project_name + r"\.(h|hpp|cpp|c)", # Project name itself
        r"main\.(h|hpp|cpp|c)",
        r"core\.(h|hpp|cpp|c)",
        r"lib\.(h|hpp|cpp|c)",
        r"app\.(h|hpp|cpp|c)",
        r"application\.(h|hpp|cpp|c)",
        r"plugin\.(h|hpp|cpp|c)",
        # For argparse example
        r"argparse\.(h|hpp)",
        r"argument_parser\.(h|hpp)",
    ]

    # Prefer files in root, include/, src/
    preferred_dirs = ["", "include", "src", "lib", "source", project_name] # Relative to project_path

    for rel_dir_prefix in preferred_dirs:
        abs_dir_prefix = os.path.join(project_path, rel_dir_prefix)
        if not os.path.isdir(abs_dir_prefix):
            continue

        for root, _, files in os.walk(abs_dir_prefix):
            # Limit depth within preferred_dirs to avoid going too deep into submodules unless it's the root
            if rel_dir_prefix and root.count(os.sep) - abs_dir_prefix.count(os.sep) > 1:
                continue
            if ".git" in root or "test" in root.lower() or "example" in root.lower() or "build" in root.lower():
                continue

            for f_name in files:
                for pattern_str in core_filename_patterns:
                    if re.fullmatch(pattern_str, f_name.lower()):
                        abs_f_path = os.path.join(root, f_name)
                        rel_f_path = os.path.relpath(abs_f_path, project_path)
                        if rel_f_path not in core_files: # Add if not already found by a more specific pattern
                            core_files[rel_f_path] = f"Matches core pattern '{pattern_str}' in '{rel_dir_prefix or 'root'}'"
                            break # File matched one pattern, move to next file
    
    return list(core_files.items())


def collect_context_files(project_path: str, main_file_rel_path: str, max_files: int = 5) -> Dict[str, str]:
    """Collects content of the main file and its directly included local headers."""
    context: Dict[str, str] = {}
    main_file_abs_path = os.path.join(project_path, main_file_rel_path)

    if not os.path.isfile(main_file_abs_path):
        logger.error(f"Main context file not found: {main_file_abs_path}")
        return context

    try:
        with open(main_file_abs_path, 'r', encoding='utf-8', errors='ignore') as f:
            main_content = f.read()
        context[main_file_rel_path] = main_content
    except Exception as e:
        logger.error(f"Failed to read main context file {main_file_rel_path}: {e}")
        return context # If main file fails, abort

    if len(context) >= max_files:
        return context

    # Find #included files
    included_headers = R_INCLUDE_DIRECTIVE.findall(main_content)
    main_file_dir_abs = os.path.dirname(main_file_abs_path)

    for header_ref in included_headers:
        if len(context) >= max_files:
            break

        # Try to resolve relative to the main file's directory first
        potential_path_abs = os.path.normpath(os.path.join(main_file_dir_abs, header_ref))
        
        # If not found, try relative to project root (common for includes like "include/common.h")
        if not os.path.isfile(potential_path_abs):
            potential_path_abs_from_proj = os.path.normpath(os.path.join(project_path, header_ref))
            if os.path.isfile(potential_path_abs_from_proj):
                 potential_path_abs = potential_path_abs_from_proj
            else: # If still not found, could be a system header or in other include paths not easily known
                logger.debug(f"Could not resolve local include: {header_ref}")
                continue
        
        header_rel_path = os.path.relpath(potential_path_abs, project_path)
        if header_rel_path not in context: # Avoid duplicates and self-include if somehow listed
            try:
                with open(potential_path_abs, 'r', encoding='utf-8', errors='ignore') as f:
                    context[header_rel_path] = f.read()
                logger.info(f"Added context file (included): {header_rel_path}")
            except Exception as e:
                logger.warning(f"Failed to read included context file {header_rel_path}: {e}")
    
    logger.info(f"Collected {len(context)} files for context. Keys: {list(context.keys())}")
    return context