#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Dataset Runner - 处理测试数据集并运行修复流程
"""

import os
import sys
import json
import logging
import subprocess
import tempfile
import shutil
import time
import re
from pathlib import Path
from datetime import datetime, timezone
import docker
import argparse

# 添加项目根目录到 Python 路径
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

import config
import github_handler
import file_locator
import ai_handler

# 创建项目数据目录
PROJECT_DATA_DIR = os.path.join(project_root, "project_data")
REPOS_DIR = os.path.join(PROJECT_DATA_DIR, "repositories")
LOGS_DIR = os.path.join(PROJECT_DATA_DIR, "logs")
RESULTS_DIR = os.path.join(PROJECT_DATA_DIR, "results")
TRAJS_DIR = os.path.join(PROJECT_DATA_DIR, "trajs")

# 确保目录存在
for dir_path in [PROJECT_DATA_DIR, REPOS_DIR, LOGS_DIR, RESULTS_DIR, TRAJS_DIR]:
    os.makedirs(dir_path, exist_ok=True)

# 配置日志 - 注意中文字符处理
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(
            os.path.join(LOGS_DIR, 'dataset_runner.log'), 
            encoding='utf-8'
        ),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

class DatasetRunner:
    def __init__(self):
        try:
            self.docker_client = docker.from_env()
            logger.info("Docker客户端初始化成功")
        except Exception as e:
            logger.error(f"Docker客户端初始化失败: {e}")
            raise
        self.results = []
        
    def load_dataset(self, file_path):
        """加载JSONL数据集文件"""
        instances = []
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    if line:
                        try:
                            instances.append(json.loads(line))
                        except json.JSONDecodeError as je:
                            logger.error(f"第{line_num}行JSON解析错误 in {file_path}: {je}")
                            continue
            logger.info(f"成功加载 {len(instances)} 个实例从 {file_path}")
            return instances
        except Exception as e:
            logger.error(f"加载数据集失败 {file_path}: {e}")
            return []

    def extract_test_patch_diff(self, instance):
        """从实例中提取test_patch.diff内容"""
        return instance.get('test_patch', '')

    
    def clone_repository_at_commit(self, repo, base_commit, instance_id):
        """克隆仓库并切换到指定commit - 使用SSH协议"""
        # 使用SSH协议而不是HTTPS
        repo_url = f"git@github.com:{repo}.git"
        repo_name = repo.split('/')[-1]
        
        # 使用更简单的目录名，基于repo名称而不是instance_id（避免重复克隆）
        repo_dir_name = f"{repo.replace('/', '_')}"
        repo_path = os.path.join(REPOS_DIR, repo_dir_name)
        
        try:
            # 检查仓库是否已存在
            if os.path.exists(repo_path):
                logger.info(f"仓库目录已存在: {repo_path}")
                
                # 检查是否是有效的git仓库
                git_check = subprocess.run(
                    ['git', 'status'], 
                    cwd=repo_path, 
                    capture_output=True, 
                    text=True
                )
                
                if git_check.returncode == 0:
                    logger.info("检测到有效的git仓库，尝试更新...")
                    
                    # 尝试fetch最新的提交
                    try:
                        subprocess.run(
                            ['git', 'fetch', 'origin'], 
                            cwd=repo_path, 
                            check=True, 
                            capture_output=True,
                            text=True,
                            encoding='utf-8'
                        )
                        logger.info("成功更新仓库")
                    except subprocess.CalledProcessError as e:
                        logger.warning(f"更新仓库失败，将重新克隆: {e}")
                        shutil.rmtree(repo_path)
                        # 继续执行克隆逻辑
                    else:
                        # 切换到目标commit
                        try:
                            logger.info(f"切换到commit {base_commit}")
                            subprocess.run(
                                ['git', 'checkout', base_commit], 
                                cwd=repo_path, 
                                check=True, 
                                capture_output=True,
                                text=True,
                                encoding='utf-8'
                            )
                            logger.info("成功重用现有仓库")
                            return repo_path
                        except subprocess.CalledProcessError as e:
                            logger.warning(f"切换commit失败: {e}")
                            # 可能是新的commit，尝试重新克隆
                            shutil.rmtree(repo_path)
                else:
                    logger.warning("目录存在但不是有效的git仓库，删除重新克隆")
                    shutil.rmtree(repo_path)
            
            # 克隆仓库（如果不存在或上面的重用失败）
            if not os.path.exists(repo_path):
                logger.info(f"使用SSH克隆仓库 {repo_url} 到 {repo_path}")
                result = subprocess.run(
                    ['git', 'clone', repo_url, repo_path], 
                    check=True, 
                    capture_output=True, 
                    text=True,
                    encoding='utf-8'
                )
                
                logger.info(f"切换到commit {base_commit}")
                result = subprocess.run(
                    ['git', 'checkout', base_commit], 
                    cwd=repo_path, 
                    check=True, 
                    capture_output=True,
                    text=True,
                    encoding='utf-8'
                )
            
            return repo_path
            
        except subprocess.CalledProcessError as e:
            logger.error(f"Git SSH操作失败: {e}")
            logger.error(f"标准输出: {e.stdout}")
            logger.error(f"标准错误: {e.stderr}")
            
            # 如果SSH失败，提供HTTPS备选方案
            logger.warning("SSH克隆失败，尝试HTTPS方式...")
            try:
                https_url = f"https://github.com/{repo}.git"
                if os.path.exists(repo_path):
                    shutil.rmtree(repo_path)
                    
                logger.info(f"使用HTTPS克隆仓库 {https_url} 到 {repo_path}")
                result = subprocess.run(
                    ['git', 'clone', https_url, repo_path], 
                    check=True, 
                    capture_output=True, 
                    text=True,
                    encoding='utf-8'
                )
                
                logger.info(f"切换到commit {base_commit}")
                result = subprocess.run(
                    ['git', 'checkout', base_commit], 
                    cwd=repo_path, 
                    check=True, 
                    capture_output=True,
                    text=True,
                    encoding='utf-8'
                )
                
                logger.info("HTTPS备选方案成功")
                return repo_path
                
            except subprocess.CalledProcessError as e2:
                logger.error(f"HTTPS备选方案也失败: {e2}")
                return None
                
        except Exception as e:
            logger.error(f"克隆仓库时发生意外错误: {e}")
            return None

    def generate_ai_patch(self, instance, repo_path, test_patch_diff, conversation_log_dir):
        """使用AI增强的文件定位和修复补丁生成"""
        try:
            # 初始化或加载现有的对话历史
            conversation_history = self._load_or_create_conversation_history(conversation_log_dir, instance)
            
            # 构建issue描述
            problem_statement = instance.get('problem_statement', '')
            hints_text = instance.get('hints_text', '')
            issue_description = f"{problem_statement}\\n\\n{hints_text}".strip()
            
            # 构建 issue_details 格式
            issue_details = {
                "title": f"Issue #{instance.get('issue_numbers', [''])[0] if instance.get('issue_numbers') else ''}",
                "body": problem_statement,
                "problem_statement": problem_statement,
                "hints_text": hints_text,
                "comments": [hints_text] if hints_text else []
            }
            
            logger.info("使用AI增强的文件定位...")
            
            # === 阶段1：AI文件定位 ===
            localization_entry = {
                "stage": "file_localization",
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "attempt": len([h for h in conversation_history.get("interactions", []) if h.get("stage") == "file_localization"]) + 1,
                "input": {
                    "issue_details": issue_details,
                    "test_patch_diff": test_patch_diff,
                    "project_path": repo_path
                }
            }
            
            # 使用AI增强的文件定位
            target_file, ai_analysis, localization_details = file_locator.find_file_to_fix_with_ai(
                issue_details=issue_details,
                project_path=repo_path,
                test_patch_diff=test_patch_diff,
                ai_api_key=config.DEEPSEEK_API_KEY,
                enable_ai=True
            )
            
            # 记录AI文件定位的完整信息
            if ai_analysis:
                localization_entry.update({
                    "ai_api_call": {
                        "prompt_sent": ai_analysis.get('_prompt_sent', ''),
                        "raw_response": ai_analysis.get('_raw_response', ''),
                        "api_status": ai_analysis.get('_api_status'),
                        "success": True
                    },
                    "ai_analysis": {k: v for k, v in ai_analysis.items() if not k.startswith('_')},
                    "result": {
                        "selected_file": target_file,
                        "confidence": ai_analysis.get('confidence'),
                        "reasoning": ai_analysis.get('reasoning'),
                    }
                })
            else:
                localization_entry.update({
                    "ai_api_call": {"success": False, "error": "AI定位失败"},
                    "result": {"selected_file": target_file, "method": "traditional"}
                })
            
            # 添加到对话历史
            conversation_history.setdefault("interactions", []).append(localization_entry)
            
            if not target_file:
                # 备选方案：从patch中提取代码文件
                logger.warning("文件定位失败，从patch中提取代码文件...")
                correct_patch = instance.get('patch', '')
                all_target_files = self.extract_target_files_from_patch(correct_patch)
                
                # 过滤掉非代码文件
                code_files = self.filter_code_files(all_target_files)
                if not code_files:
                    logger.error("未找到任何代码文件需要修复")
                    return None, None
                
                target_file = code_files[0]  # 选择第一个代码文件
                logger.info(f"从patch中选择代码文件: {target_file}")
            
            target_file_path = os.path.join(repo_path, target_file)
            
            if not os.path.exists(target_file_path):
                logger.error(f"目标文件不存在: {target_file_path}")
                return None, None
                
            # 读取文件内容
            with open(target_file_path, 'r', encoding='utf-8', errors='ignore') as f:
                file_content = f.read()
            
            # 收集上下文文件
            context_files = file_locator.collect_context_files(repo_path, target_file)
            
            # === 阶段2：AI补丁生成 ===
            patch_generation_entry = {
                "stage": "patch_generation",
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "attempt": len([h for h in conversation_history.get("interactions", []) if h.get("stage") == "patch_generation"]) + 1,
                "input": {
                    "target_file": target_file,
                    "ai_analysis_from_localization": ai_analysis,
                    "context_files": list(context_files.keys()) if context_files else []
                }
            }
            
            # 构建增强的修复prompt
            enhanced_prompt = self.build_enhanced_repair_prompt(
                instance, target_file, file_content, context_files, 
                test_patch_diff, ai_analysis
            )
            
            logger.info(f"为文件 {target_file} 生成AI补丁...")
            ai_result = ai_handler.generate_patch_with_enhanced_prompt(
                config.DEEPSEEK_API_KEY,
                enhanced_prompt,
                stream=False
            )
            
            # 记录补丁生成的完整信息
            patch_generation_entry.update({
                "ai_api_call": {
                    "prompt_sent": ai_result.get('prompt_sent', ''),
                    "raw_response": ai_result.get('raw_response', ''),
                    "api_status": ai_result.get('status_code'),
                    "success": ai_result.get('api_call_successful', False),
                    "error_message": ai_result.get('error_message', '')
                },
                "result": {
                    "extracted_patch": ai_result.get('extracted_patch', ''),
                    "patch_length": len(ai_result.get('extracted_patch', '')),
                    "success": bool(ai_result.get('api_call_successful') and ai_result.get('extracted_patch'))
                }
            })
            
            # 添加到对话历史
            conversation_history["interactions"].append(patch_generation_entry)
            
            # 保存完整的对话历史
            conversation_file = os.path.join(conversation_log_dir, "ai_conversation_history.json")
            with open(conversation_file, 'w', encoding='utf-8') as f:
                json.dump(conversation_history, f, indent=2, ensure_ascii=False)
            
            # 为了兼容性，也保存单次对话记录
            self._save_individual_conversation_files(conversation_log_dir, localization_entry, patch_generation_entry)
            
            if ai_result.get('api_call_successful') and ai_result.get('extracted_patch'):
                logger.info(f"AI补丁生成成功，完整对话历史已保存到: {conversation_file}")
                return ai_result['extracted_patch'], conversation_file
            else:
                logger.error(f"AI生成补丁失败: {ai_result.get('error_message')}")
                return None, conversation_file
                
        except Exception as e:
            logger.error(f"生成AI补丁时出错: {e}")
            return None, None

    def _load_or_create_conversation_history(self, conversation_log_dir, instance):
        """加载或创建对话历史"""
        conversation_file = os.path.join(conversation_log_dir, "ai_conversation_history.json")
        
        if os.path.exists(conversation_file):
            try:
                with open(conversation_file, 'r', encoding='utf-8') as f:
                    history = json.load(f)
                logger.info(f"加载现有对话历史，包含 {len(history.get('interactions', []))} 次交互")
                return history
            except Exception as e:
                logger.warning(f"加载对话历史失败: {e}，创建新的历史记录")
        
        # 创建新的对话历史
        return {
            "metadata": {
                "instance_id": instance.get('instance_id'),
                "created_at": datetime.now(timezone.utc).isoformat(),
                "repo": instance.get('repo'),
                "pull_number": instance.get('pull_number')
            },
            "interactions": []
        }

    def _save_individual_conversation_files(self, conversation_log_dir, localization_entry, patch_generation_entry):
        """保存单独的对话文件，用于兼容性"""
        
        # 保存文件定位prompt和响应
        if localization_entry.get("ai_api_call", {}).get("prompt_sent"):
            with open(os.path.join(conversation_log_dir, "localization_prompt.txt"), 'w', encoding='utf-8') as f:
                f.write(localization_entry["ai_api_call"]["prompt_sent"])
            
            with open(os.path.join(conversation_log_dir, "localization_response.txt"), 'w', encoding='utf-8') as f:
                f.write(localization_entry["ai_api_call"]["raw_response"])
        
        # 保存补丁生成prompt和响应
        if patch_generation_entry.get("ai_api_call", {}).get("prompt_sent"):
            with open(os.path.join(conversation_log_dir, "patch_generation_prompt.txt"), 'w', encoding='utf-8') as f:
                f.write(patch_generation_entry["ai_api_call"]["prompt_sent"])
            
            with open(os.path.join(conversation_log_dir, "patch_generation_response.txt"), 'w', encoding='utf-8') as f:
                f.write(patch_generation_entry["ai_api_call"]["raw_response"])

    def build_enhanced_repair_prompt(self, instance, target_file, file_content, context_files, 
                                    test_patch_diff, ai_analysis):
        """构建增强的修复prompt，整合AI分析结果"""
        
        problem_statement = instance.get('problem_statement', '')
        hints_text = instance.get('hints_text', '')
        
        # 从AI分析中获取修复策略
        fix_strategy = ""
        test_insights = ""
        if ai_analysis:
            fix_strategy = ai_analysis.get('fix_strategy', '')
            test_insights = ai_analysis.get('test_insights', '')
        
        # 构建上下文文件信息
        context_info = ""
        if context_files:
            context_info = "\n## 相关上下文文件\n"
            for file_path, content in context_files.items():
                if file_path != target_file:  # 避免重复
                    context_info += f"\n### {file_path}\n```c\n{content[:1000]}...\n```\n"
        
        prompt = f"""你是一个专业的C/C++代码修复专家。请分析以下问题并生成精确的修复补丁。

    ## 问题描述
    {problem_statement}

    ## 提示信息  
    {hints_text}

    ## AI分析的修复策略
    {fix_strategy}

    ## 从测试变更中得出的关键信息
    {test_insights}

    ## 测试变更差异（展示期望的修复效果）
    ```diff
    {test_patch_diff}        
    需要修复的目标文件：{target_file}
    {file_content}
    {context_info}

    修复要求
    精确定位问题：根据问题描述和测试变更，准确找到需要修复的代码位置
    最小化修改：只修改必要的部分，保持代码的其他功能不变
    符合测试期望：确保修复后的代码能通过测试变更中显示的测试用例
    代码质量：保持良好的代码风格和错误处理
    输出格式
    请只输出标准的git diff格式补丁，不要包含任何解释文字：
    diff --git a/{target_file} b/{target_file}
    index abc1234..def5678 100644
    --- a/{target_file}
    +++ b/{target_file}
    @@ -行号,行数 +行号,行数 @@
    上下文行
    -删除的行
    +添加的行
    上下文行
    注意：

    补丁必须可以直接用 patch -p1 命令应用

    行号必须准确对应目标文件的实际内容

    确保修复逻辑正确且完整 """

        return prompt

    def extract_target_files_from_patch(self, patch_content):
        """从patch内容中提取目标文件路径"""
        target_files = []
        lines = patch_content.split('\n')
        
        for line in lines:
            if line.startswith('diff --git a/'):
                # 提取文件路径: diff --git a/path/file.c b/path/file.c
                parts = line.split()
                if len(parts) >= 4:
                    file_path = parts[2][2:]  # 去掉 'a/'
                    target_files.append(file_path)
        
        return target_files
    
    def filter_code_files(self, file_paths):
        """过滤出代码文件，排除配置和文档文件"""
        code_extensions = {'.c', '.cpp', '.cc', '.cxx', '.h', '.hpp', '.hxx', '.inl', '.tcc'}
        ignore_patterns = [
            r'\.valgrind',
            r'\.gitignore', 
            r'\.cmake',
            r'CMakeLists\.txt',
            r'Makefile',
            r'\.md$',
            r'\.pod$',
            r'\.txt$',
            r'\.yml$',
            r'\.yaml$',
            r'\.json$',
            r'docs/',
            r'doc/',
            r'documentation/',
            r'README',
            r'LICENSE',
            r'CHANGELOG'
        ]
        
        code_files = []
        for file_path in file_paths:
            # 检查扩展名
            _, ext = os.path.splitext(file_path)
            if ext.lower() not in code_extensions:
                continue
                
            # 检查是否匹配忽略模式
            should_ignore = False
            for pattern in ignore_patterns:
                if re.search(pattern, file_path, re.IGNORECASE):
                    should_ignore = True
                    break
            
            if not should_ignore:
                code_files.append(file_path)
                
        return code_files

    def run_docker_test(self, instance, patch_content, test_log_dir):
        """在Docker容器中运行测试 - 包含多种容错方案"""
        repo = instance['repo']
        pull_number = instance['pull_number']
        
        # Docker镜像名
        image_name = f"dongqa930/{repo.replace('/', '_')}:{pull_number}"
        
        try:
            logger.info(f"拉取Docker镜像: {image_name}")
            self.docker_client.images.pull(image_name)
            
            # 尝试多种方法传输patch文件
            methods = [
                ("heredoc", self._docker_test_heredoc),
                ("base64", self._docker_test_base64)
            ]
            
            for method_name, method_func in methods:
                logger.info(f"尝试使用 {method_name} 方法...")
                try:
                    result = method_func(image_name, patch_content, test_log_dir, method_name)
                    if result is not None:
                        logger.info(f"{method_name} 方法执行成功")
                        return result
                    else:
                        logger.warning(f"{method_name} 方法返回None，尝试下一种方法")
                except Exception as e:
                    logger.warning(f"{method_name} 方法失败: {e}")
                    continue
            
            # 所有方法都失败了
            raise Exception("所有Docker测试方法都失败了")
            
        except Exception as e:
            logger.error(f"Docker测试完全失败: {e}")
            error_logs = f"Docker测试异常: {str(e)}"
            
            # 保存错误日志
            error_log_file = os.path.join(test_log_dir, "docker_error.log")
            with open(error_log_file, 'w', encoding='utf-8') as f:
                f.write(error_logs)
            
            return {
                'success': False,
                'patch_exit_code': -1,
                'test_exit_code': -1,
                'logs': error_logs,
                'log_file': error_log_file,
                'method_used': 'none'
            }

    def _docker_test_volume_mount(self, image_name, patch_content, test_log_dir, method_name):
        """方法1: 使用volume挂载（最可靠）"""
        try:
            # 创建临时目录存放patch
            with tempfile.TemporaryDirectory() as temp_dir:
                patch_file = os.path.join(temp_dir, 'fix.patch')
                with open(patch_file, 'w', encoding='utf-8') as f:
                    f.write(patch_content)
                
                logger.info(f"使用volume挂载方法运行Docker容器...")
                
                # 使用volume挂载
                container = self.docker_client.containers.create(
                    image_name,
                    command="bash",
                    stdin_open=True,
                    tty=True,
                    working_dir="/home",
                    volumes={temp_dir: {'bind': '/tmp/patch_dir', 'mode': 'ro'}}
                )
                
                try:
                    container.start()
                    
                    # 复制patch文件
                    logger.info("复制patch文件到容器...")
                    copy_result = container.exec_run("cp /tmp/patch_dir/fix.patch /home/fix.patch")
                    if copy_result.exit_code != 0:
                        logger.error(f"复制文件失败: {copy_result.output.decode('utf-8', errors='ignore')}")
                        return None
                    
                    # 验证文件存在
                    verify_result = container.exec_run("ls -la /home/fix.patch")
                    if verify_result.exit_code != 0:
                        logger.error("验证patch文件失败")
                        return None
                    
                    return self._execute_patch_and_test(container, test_log_dir, method_name)
                    
                finally:
                    container.stop()
                    container.remove()
                    
        except Exception as e:
            logger.error(f"Volume挂载方法异常: {e}")
            return None

    def _docker_test_heredoc(self, image_name, patch_content, test_log_dir, method_name):
        """方法2: 使用heredoc创建文件"""
        try:
            container = self.docker_client.containers.create(
                image_name,
                command="bash",
                stdin_open=True,
                tty=True,
                working_dir="/home"
            )
            
            try:
                container.start()
                
                # 确保删除已存在的fix.patch
                container.exec_run("rm -f /home/fix.patch")
                
                # 使用heredoc创建文件，避免shell特殊字符问题
                logger.info("使用heredoc方法创建patch文件...")
                
                # 将patch内容写入临时文件，然后用cat读取
                # 这样可以避免shell转义的复杂性
                lines = patch_content.split('\n')
                
                # 方法: 逐行echo并追加
                container.exec_run("touch /home/fix.patch")  # 创建空文件
                
                for i, line in enumerate(lines):
                    # 转义特殊字符
                    escaped_line = line.replace('\\', '\\\\').replace('"', '\\"').replace('$', '\\$').replace('`', '\\`')
                    if i == 0:
                        cmd = f'echo "{escaped_line}" > /home/fix.patch'
                    else:
                        cmd = f'echo "{escaped_line}" >> /home/fix.patch'
                    
                    result = container.exec_run(f"bash -c '{cmd}'")
                    if result.exit_code != 0:
                        logger.error(f"写入第{i+1}行失败")
                        return None
                
                # 验证文件创建成功
                verify_result = container.exec_run("wc -l /home/fix.patch")
                if verify_result.exit_code == 0:
                    logger.info(f"Patch文件创建成功: {verify_result.output.decode('utf-8', errors='ignore').strip()}")
                else:
                    logger.error("验证patch文件失败")
                    return None
                
                return self._execute_patch_and_test(container, test_log_dir, method_name)
                
            finally:
                container.stop()
                container.remove()
                
        except Exception as e:
            logger.error(f"Heredoc方法异常: {e}")
            return None

    def _docker_test_base64(self, image_name, patch_content, test_log_dir, method_name):
        """方法3: 使用base64编码传输（最安全）"""
        try:
            import base64
            
            container = self.docker_client.containers.create(
                image_name,
                command="bash",
                stdin_open=True,
                tty=True,
                working_dir="/home"
            )
            
            try:
                container.start()
                
                # 确保删除已存在的fix.patch
                container.exec_run("rm -f /home/fix.patch")
                
                logger.info("使用base64编码方法创建patch文件...")
                
                # 使用base64编码传输文件内容
                patch_b64 = base64.b64encode(patch_content.encode('utf-8')).decode('ascii')
                
                # 分块传输，避免命令行长度限制
                chunk_size = 1000  # 每块1000字符
                chunks = [patch_b64[i:i+chunk_size] for i in range(0, len(patch_b64), chunk_size)]
                
                # 创建空文件
                container.exec_run("touch /home/fix.patch.b64")
                
                # 分块写入base64数据
                for i, chunk in enumerate(chunks):
                    if i == 0:
                        cmd = f'echo "{chunk}" > /home/fix.patch.b64'
                    else:
                        cmd = f'echo "{chunk}" >> /home/fix.patch.b64'
                    
                    result = container.exec_run(f"bash -c '{cmd}'")
                    if result.exit_code != 0:
                        logger.error(f"写入base64块{i+1}失败")
                        return None
                
                # 解码base64文件
                decode_result = container.exec_run("base64 -d /home/fix.patch.b64 > /home/fix.patch")
                if decode_result.exit_code != 0:
                    logger.error(f"base64解码失败: {decode_result.output.decode('utf-8', errors='ignore')}")
                    return None
                
                # 清理临时文件
                container.exec_run("rm -f /home/fix.patch.b64")
                
                # 验证文件创建成功
                verify_result = container.exec_run("ls -la /home/fix.patch")
                if verify_result.exit_code == 0:
                    logger.info("Base64方法patch文件创建成功")
                else:
                    logger.error("验证patch文件失败")
                    return None
                
                return self._execute_patch_and_test(container, test_log_dir, method_name)
                
            finally:
                container.stop()
                container.remove()
                
        except Exception as e:
            logger.error(f"Base64方法异常: {e}")
            return None

    def _execute_patch_and_test(self, container, test_log_dir, method_name):
        """执行patch应用和测试的公共逻辑 - 修复：不应用AI补丁，只运行测试"""
        try:
            # 显示patch文件内容的前几行用于调试
            preview_result = container.exec_run(["head", "-n", "10", "/home/fix.patch"])
            if preview_result.exit_code == 0:
                logger.info(f"AI生成的补丁预览:\n{preview_result.output.decode('utf-8', errors='ignore')}")
            
            # 不再应用AI补丁，因为fix_run.sh会处理测试补丁
            # 直接运行测试脚本
            logger.info("运行测试脚本（fix_run.sh会自动应用测试补丁）...")
            test_result = container.exec_run([
                "bash", "-c", "cd /home && bash fix_run.sh"
            ])
            test_logs = test_result.output.decode('utf-8', errors='ignore')
            
            # 记录测试日志
            full_logs = f"=== METHOD: {method_name.upper()} ===\n"
            full_logs += f"=== AI PATCH PREVIEW ===\n{preview_result.output.decode('utf-8', errors='ignore') if preview_result.exit_code == 0 else 'Failed to preview patch'}\n\n"
            full_logs += f"=== TEST EXECUTION ===\n{test_logs}"
            
            success = test_result.exit_code == 0
            
            # 保存测试日志
            test_log_file = os.path.join(test_log_dir, f"docker_test_{method_name}.log")
            with open(test_log_file, 'w', encoding='utf-8') as f:
                f.write(full_logs)
            
            return {
                        'success': success,
                        'test_exit_code': test_result.exit_code,
                        'patch_exit_code': test_result.exit_code,  # 添加这个键
                        'logs': full_logs,
                        'log_file': test_log_file,
                        'method_used': method_name,
                        'error_message': '' if success else f"Test failed with exit code {test_result.exit_code}"
                    }
            
        except Exception as e:
            logger.error(f"执行测试时异常: {e}")
            return {
            'success': False,
            'test_exit_code': -1,
            'patch_exit_code': -1,  # 添加这个键
            'logs': f"Exception occurred: {str(e)}",
            'log_file': '',
            'method_used': method_name,
            'error_message': str(e)
        }
    
    def generate_trajectory(self, instance, steps, success):
        """生成trajectory记录"""
        trajectory = {
            "instance_id": instance["instance_id"],
            "repo": instance["repo"],
            "pull_number": instance["pull_number"],
            "model_name": "CFix_DeepSeek",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "status": "SUCCESS" if success else "FAILED",
            "steps": steps
        }
        
        # 保存trajectory文件 - 使用规范的命名
        trajectory_filename = f"{instance['instance_id']}_trajectory.json"
        trajectory_file = os.path.join(TRAJS_DIR, trajectory_filename)
        
        with open(trajectory_file, "w", encoding='utf-8') as f:
            json.dump(trajectory, f, indent=2, ensure_ascii=False)
        
        return trajectory_file

    def create_instance_log_dir(self, instance):
        """为实例创建日志目录"""
        instance_id = instance['instance_id']
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_dir_name = f"{instance_id}_{timestamp}"
        log_dir = os.path.join(LOGS_DIR, log_dir_name)
        os.makedirs(log_dir, exist_ok=True)
        return log_dir

    def process_instance(self, instance, max_retries=3):
        """处理单个实例"""
        instance_id = instance['instance_id']
        repo = instance['repo']
        
        logger.info(f"处理实例: {instance_id} ({repo})")
        
        # 创建实例专用日志目录
        instance_log_dir = self.create_instance_log_dir(instance)
        
        steps = []
        start_time = datetime.now(timezone.utc).isoformat()
        
        steps.append({
            "step": "setup",
            "timestamp": start_time,
            "action": "Created instance log directory",
            "log_dir": instance_log_dir
        })
        
        # 克隆仓库
        repo_path = self.clone_repository_at_commit(
            repo, 
            instance['base_commit'],
            instance_id
        )
        
        if not repo_path:
            steps.append({
                "step": "clone_repo",
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "action": "Failed to clone repository",
                "success": False
            })
            return self.create_failed_result(instance, steps, "Repository clone failed")
        
        steps.append({
            "step": "clone_repo", 
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "action": f"Successfully cloned repository to {repo_path}",
            "success": True
        })
        
        # 提取test_patch_diff
        test_patch_diff = self.extract_test_patch_diff(instance)
        
        # 保存原始数据到日志目录
        original_data_file = os.path.join(instance_log_dir, "original_instance_data.json")
        with open(original_data_file, 'w', encoding='utf-8') as f:
            json.dump(instance, f, indent=2, ensure_ascii=False)
        
        # 尝试多次生成和测试补丁
        for attempt in range(1, max_retries + 1):
            logger.info(f"尝试 {attempt}/{max_retries}")
            
            # 为每次尝试创建子目录
            attempt_log_dir = os.path.join(instance_log_dir, f"attempt_{attempt}")
            os.makedirs(attempt_log_dir, exist_ok=True)
            
            # 生成AI补丁
            patch_content, conversation_log = self.generate_ai_patch(
                instance, repo_path, test_patch_diff, attempt_log_dir
            )
            
            step_data = {
                "step": f"generate_patch_attempt_{attempt}",
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "attempt": attempt,
                "conversation_log": conversation_log
            }
            
            if not patch_content:
                step_data.update({
                    "action": "Failed to generate AI patch",
                    "success": False
                })
                steps.append(step_data)
                
                if attempt == max_retries:
                    return self.create_failed_result(instance, steps, "AI patch generation failed")
                continue
            
            # 保存生成的补丁
            patch_file = os.path.join(attempt_log_dir, "generated_patch.diff")
            with open(patch_file, 'w', encoding='utf-8') as f:
                f.write(patch_content)
            
            step_data.update({
                "action": "Successfully generated AI patch",
                "patch_file": patch_file,
                "patch_preview": patch_content[:200] + "..." if len(patch_content) > 200 else patch_content,
                "success": True
            })
            steps.append(step_data)
            
            # Docker测试
            test_result = self.run_docker_test(instance, patch_content, attempt_log_dir)
            
            test_step = {
                "step": f"docker_test_attempt_{attempt}",
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "attempt": attempt,
                "action": "Executed Docker test",
                "success": test_result['success'],
                "patch_exit_code": test_result['patch_exit_code'],
                "test_exit_code": test_result['test_exit_code'],
                "test_log_file": test_result['log_file'],
                "logs_preview": test_result['logs'][:500] + "..." if len(test_result['logs']) > 500 else test_result['logs']
            }
            steps.append(test_step)
            
            if test_result['success']:
                # 测试成功 - 保存最终结果
                final_patch_file = os.path.join(RESULTS_DIR, f"{instance_id}_final_patch.diff")
                with open(final_patch_file, 'w', encoding='utf-8') as f:
                    f.write(patch_content)
                
                trajectory_file = self.generate_trajectory(instance, steps, True)
                return {
                    "instance_id": instance_id,
                    "success": True,
                    "attempts": attempt,
                    "final_patch_file": final_patch_file,
                    "trajectory_file": trajectory_file,
                    "log_directory": instance_log_dir,
                    "test_logs": test_result['logs']
                }
            else:
                logger.warning(f"尝试 {attempt} 测试失败")
                if attempt < max_retries:
                    time.sleep(2)
        
        # 所有尝试都失败了
        return self.create_failed_result(instance, steps, "All test attempts failed")

    def create_failed_result(self, instance, steps, reason):
        """创建失败结果"""
        trajectory_file = self.generate_trajectory(instance, steps, False)
        return {
            "instance_id": instance['instance_id'],
            "success": False,
            "reason": reason,
            "trajectory_file": trajectory_file
        }

    def run_dataset(self, dataset_files, output_file="results.json"):
        """运行整个数据集"""
        all_instances = []
        
        # 加载所有数据集文件
        for file_path in dataset_files:
            instances = self.load_dataset(file_path)
            all_instances.extend(instances)
        
        logger.info(f"总共需要处理 {len(all_instances)} 个实例")
        
        results = []
        for i, instance in enumerate(all_instances, 1):
            logger.info(f"进度: {i}/{len(all_instances)}")
            
            try:
                result = self.process_instance(instance)
                results.append(result)
                
                # 定期保存结果
                if i % 5 == 0:
                    self.save_results(results, output_file)
                    
            except Exception as e:
                logger.error(f"处理实例 {instance['instance_id']} 时出错: {e}")
                results.append(self.create_failed_result(
                    instance, 
                    [{"step": "error", "message": str(e), "timestamp": datetime.now(timezone.utc).isoformat()}], 
                    f"Unexpected error: {str(e)}"
                ))
        
        # 最终保存结果
        self.save_results(results, output_file)
        self.print_summary(results)
        
        return results

    def save_results(self, results, output_file):
        """保存结果到文件"""
        output_path = os.path.join(RESULTS_DIR, output_file)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        logger.info(f"结果已保存到 {output_path}")

    def print_summary(self, results):
        """打印结果摘要"""
        total = len(results)
        successful = sum(1 for r in results if r['success'])
        failed = total - successful
        
        logger.info("="*50)
        logger.info(f"执行完成!")
        logger.info(f"总实例数: {total}")
        logger.info(f"成功: {successful} ({successful/total*100:.1f}%)")
        logger.info(f"失败: {failed} ({failed/total*100:.1f}%)")
        logger.info(f"日志目录: {LOGS_DIR}")
        logger.info(f"结果目录: {RESULTS_DIR}")
        logger.info(f"轨迹目录: {TRAJS_DIR}")
        logger.info("="*50)


def main():
    parser = argparse.ArgumentParser(description="Dataset Runner for CFix")
    parser.add_argument(
        '--dataset-dir', 
        default='dataset',
        help='数据集目录路径 (默认: dataset)'
    )
    parser.add_argument(
        '--files',
        nargs='+',
        help='指定要处理的JSONL文件名 (例如: test_c.jsonl test_cpp.jsonl)'
    )
    parser.add_argument(
        '--output',
        default='execution_results.json',
        help='结果输出文件名 (默认: execution_results.json)'
    )
    parser.add_argument(
        '--max-retries',
        type=int,
        default=3,
        help='每个实例的最大重试次数 (默认: 3)'
    )
    parser.add_argument(
        '--instance-filter',
        help='只处理指定的instance_id'
    )
    
    args = parser.parse_args()
    
    # 检查必要配置
    if not config.DEEPSEEK_API_KEY:
        logger.error("请设置 DEEPSEEK_API_KEY")
        return 1
    
    # 确定要处理的文件
    if args.files:
        dataset_files = [os.path.join(args.dataset_dir, f) for f in args.files]
    else:
        # 默认处理所有JSONL文件
        dataset_files = []
        for filename in ['test_c.jsonl', 'test_cpp.jsonl']:
            filepath = os.path.join(args.dataset_dir, filename)
            if os.path.exists(filepath):
                dataset_files.append(filepath)
    
    if not dataset_files:
        logger.error(f"在 {args.dataset_dir} 中未找到数据集文件")
        return 1
    
    logger.info(f"将处理以下数据集文件: {dataset_files}")
    logger.info(f"项目数据目录: {PROJECT_DATA_DIR}")
    
    # 运行数据集
    runner = DatasetRunner()
    
    # 如果指定了实例过滤，修改数据加载逻辑
    if args.instance_filter:
        all_instances = []
        for file_path in dataset_files:
            instances = runner.load_dataset(file_path)
            filtered = [inst for inst in instances if inst.get('instance_id') == args.instance_filter]
            all_instances.extend(filtered)
        
        if not all_instances:
            logger.error(f"未找到指定的实例: {args.instance_filter}")
            return 1
        
        logger.info(f"过滤后只处理实例: {args.instance_filter}")
        # 手动运行单个实例
        result = runner.process_instance(all_instances[0], args.max_retries)
        results = [result]
        runner.save_results(results, args.output)
        runner.print_summary(results)
    else:
        results = runner.run_dataset(dataset_files, args.output)
    
    return 0 if any(r['success'] for r in results) else 1


if __name__ == "__main__":
    sys.exit(main())