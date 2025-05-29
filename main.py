import logging
import os
import shutil
import argparse
import time
import json

import config
import github_handler
import file_locator
import ai_handler
import patch_handler
import docker_runner

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(config.LOG_FILE, encoding='utf-8'),  # 添加 encoding='utf-8'
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def save_issue_info(issue_details, owner, repo_name, issue_number):
    """保存原始 issue 信息到专门的文件中。"""
    issues_dir = os.path.join(config.TEMP_WORK_DIR, "issues")
    os.makedirs(issues_dir, exist_ok=True)
    
    issue_filename = f"{owner}_{repo_name}_{issue_number}.json"
    issue_file_path = os.path.join(issues_dir, issue_filename)
    
    # 创建要保存的 issue 信息对象
    issue_info = {
        "title": issue_details.get("title", ""),
        "body": issue_details.get("body", ""),
        "url": issue_details.get("url", ""),
        "created_at": issue_details.get("created_at", "").isoformat() if issue_details.get("created_at") else "",
        "comments": issue_details.get("comments", []),
        "default_branch": issue_details.get("default_branch", "")
    }
    
    # 同时保存原始文本描述，便于AI处理
    text_description = f"Title: {issue_info['title']}\n\nBody:\n{issue_info['body']}"
    if issue_info["comments"]:
        text_description += "\n\nComments:\n" + "\n---\n".join(issue_info["comments"])
    
    # 保存 JSON 格式
    with open(issue_file_path, "w", encoding="utf-8") as f:
        json.dump(issue_info, f, ensure_ascii=False, indent=2)
        
    # 保存文本格式
    text_file_path = os.path.join(issues_dir, f"{owner}_{repo_name}_{issue_number}.txt")
    with open(text_file_path, "w", encoding="utf-8") as f:
        f.write(text_description)
        
    logger.info(f"Issue 信息已保存到 {issue_file_path} 和 {text_file_path}")
    return text_description

def get_or_clone_repository(owner, repo_name, issue_number, issue_created_at=None, default_branch=None):
    """获取或克隆仓库，并检出到指定版本。优先检查已有仓库。"""
    repos_base_dir = os.path.join(config.TEMP_WORK_DIR, "repositories")
    os.makedirs(repos_base_dir, exist_ok=True)
    
    # 仓库目录名使用 owner_repo，不包含 issue_number 和时间戳
    # 这样多个 issue 可以复用同一个仓库
    repo_dir_name = f"{owner}_{repo_name}"
    repo_path = os.path.join(repos_base_dir, repo_dir_name)
    
    repo_url = github_handler.generate_repo_url(owner, repo_name, use_ssh=True)
    
    logger.info(f"检查是否已存在仓库: {repo_path}")
    
    if os.path.exists(os.path.join(repo_path, ".git")):
        logger.info(f"已发现现有仓库: {repo_path}")
        # 仓库已存在，更新远程并切换到特定提交
        try:
            from git import Repo
            repo = Repo(repo_path)
            # 获取最新的远程变更
            logger.info(f"更新远程仓库...")
            repo.git.fetch("--all")
            
            # 如果提供了 issue_created_at 和 default_branch，尝试检出到特定提交
            if issue_created_at and default_branch:
                logger.info(f"尝试检出到 {default_branch} 分支在 {issue_created_at.isoformat()} 之前的提交")
                until_date_str = issue_created_at.strftime('%Y-%m-%dT%H:%M:%SZ')
                
                # 确保存在该分支
                if f"origin/{default_branch}" in [ref.name for ref in repo.references]:
                    # 找到特定日期之前的最新提交
                    commits = list(repo.iter_commits(f"origin/{default_branch}", max_count=1, until=until_date_str))
                    if commits:
                        commit_to_checkout = commits[0]
                        logger.info(f"找到提交: {commit_to_checkout.hexsha} (日期: {commit_to_checkout.committed_datetime})")
                        repo.git.checkout(commit_to_checkout.hexsha)
                        logger.info(f"成功检出到提交: {commit_to_checkout.hexsha}")
                    else:
                        logger.warning(f"在 {default_branch} 分支上未找到 {issue_created_at.isoformat()} 之前的提交。")
                else:
                    logger.warning(f"仓库中不存在分支: origin/{default_branch}")
            return repo_path
        except Exception as e:
            logger.error(f"使用现有仓库时出错: {e}")
            logger.info(f"将尝试删除并重新克隆仓库")
            try:
                shutil.rmtree(repo_path)
            except Exception as rm_e:
                logger.warning(f"删除仓库目录失败: {rm_e}")
    
    # 克隆新仓库
    logger.info(f"克隆仓库 {repo_url} 到 {repo_path}")
    return github_handler.clone_repository(repo_url, repo_path, issue_created_at, default_branch)

def cleanup_directory(dir_path, keep_dir=False):
    """清理目录内容，可选择保留目录本身。"""
    if not os.path.exists(dir_path):
        logger.info(f"目录不存在，无需清理: {dir_path}")
        return
        
    logger.info(f"开始清理目录: {dir_path}")
    try:
        if keep_dir:
            # 只删除目录内容，保留目录本身
            for item in os.listdir(dir_path):
                item_path = os.path.join(dir_path, item)
                if os.path.isfile(item_path):
                    os.unlink(item_path)
                elif os.path.isdir(item_path):
                    shutil.rmtree(item_path)
            logger.info(f"目录内容已清理，保留目录: {dir_path}")
        else:
            # 删除整个目录
            import stat
            
            def on_rm_error(func, path, exc_info):
                # 处理权限错误
                if not os.access(path, os.W_OK):
                    # 尝试修改文件权限
                    os.chmod(path, stat.S_IWUSR)
                    # 再次尝试删除
                    func(path)
                    
            shutil.rmtree(dir_path, onerror=on_rm_error)
            logger.info(f"目录已完全清理: {dir_path}")
    except Exception as e:
        logger.error(f"清理目录 {dir_path} 失败: {e}")


def find_test_dockerfile(issue_url: str, project_name: str, base_dockerfile_path: str) -> str | None:
    """
    根据 issue 或项目信息定位测试用的 Dockerfile。
    这是一个占位符，你需要根据你的测试集结构来实现。
    例如，如果 Dockerfile 以项目名或 issue ID 命名。
    """
    # 示例逻辑：假设 Dockerfile 与项目同名，存放在 DOCKERFILES_BASE_PATH 下
    # e.g., if project_name is "myproject", look for "myproject.Dockerfile" or "Dockerfile.myproject"
    
    # 尝试直接用项目名
    potential_dockerfile_name1 = f"{project_name}.Dockerfile"
    potential_dockerfile_name2 = f"Dockerfile.{project_name}"
    potential_dockerfile_name3 = "Dockerfile" # 通用名

    for df_name in [potential_dockerfile_name1, potential_dockerfile_name2, potential_dockerfile_name3]:
        df_path = os.path.join(base_dockerfile_path, df_name)
        if os.path.isfile(df_path):
            logger.info(f"找到测试 Dockerfile: {df_path}")
            return df_path
        
        # 如果项目名包含子目录，例如 "owner_repo", 尝试只用 repo 部分
        if '_' in project_name:
            repo_only_name = project_name.split('_',1)[1]
            potential_dockerfile_name_repo1 = f"{repo_only_name}.Dockerfile"
            potential_dockerfile_name_repo2 = f"Dockerfile.{repo_only_name}"
            for df_repo_name in [potential_dockerfile_name_repo1, potential_dockerfile_name_repo2]:
                 df_repo_path = os.path.join(base_dockerfile_path, df_repo_name)
                 if os.path.isfile(df_repo_path):
                    logger.info(f"找到测试 Dockerfile: {df_repo_path}")
                    return df_repo_path


    logger.warning(f"未能找到项目 '{project_name}' 对应的测试 Dockerfile 于 '{base_dockerfile_path}'")
    # 也可以尝试从 issue URL 中提取 issue number 来命名 Dockerfile
    parsed_url = github_handler.parse_issue_url(issue_url)
    if parsed_url:
        _, repo, issue_num = parsed_url
        potential_dockerfile_issue_name = f"{repo}-issue-{issue_num}.Dockerfile"
        df_issue_path = os.path.join(base_dockerfile_path, potential_dockerfile_issue_name)
        if os.path.isfile(df_issue_path):
            logger.info(f"找到基于 Issue 的测试 Dockerfile: {df_issue_path}")
            return df_issue_path

    return None


def process_issue(issue_url: str):
    logger.info(f"开始处理 Issue: {issue_url}")

    if not config.GITHUB_TOKEN:
        logger.error("GITHUB_TOKEN 未配置。请在 .env 文件中设置。")
        return
    if not config.DEEPSEEK_API_KEY:
        logger.error("DEEPSEEK_API_KEY 未配置。请在 .env 文件中设置。")
        return

    parsed_url = github_handler.parse_issue_url(issue_url)
    if not parsed_url:
        return
    owner, repo_name, issue_number = parsed_url
    
    # 为本次运行创建唯一的工作目录
    run_id = int(time.time())
    run_dir_name = f"{owner}_{repo_name}_{issue_number}_{run_id}"
    run_work_dir = os.path.join(config.TEMP_WORK_DIR, "runs", run_dir_name)
    os.makedirs(run_work_dir, exist_ok=True)
    
    try:
        # 获取 issue 详情
        issue_details = github_handler.get_issue_details(owner, repo_name, issue_number, config.GITHUB_TOKEN)
        if not issue_details:
            logger.error("获取 Issue 详细信息失败。")
            return
        
        # 保存 issue 信息
        issue_full_description = save_issue_info(issue_details, owner, repo_name, issue_number)
        
        # 获取 issue 创建时间和默认分支
        issue_created_at = issue_details.get("created_at")
        default_branch = issue_details.get("default_branch")
        
        # 获取或克隆仓库
        cloned_project_path = get_or_clone_repository(owner, repo_name, issue_number, issue_created_at, default_branch)
        if not cloned_project_path:
            logger.error(f"获取仓库代码失败。")
            return
        
        logger.info(f"Issue 标题: {issue_details['title']}")


        # 定位需修复的文件
        target_file_relative_path = file_locator.find_file_to_fix(issue_details, cloned_project_path)
        if not target_file_relative_path:
            logger.error("未能定位到需要修复的文件。")
            # 尝试查找可能的核心文件，作为备选
            possible_core_files = file_locator.find_core_files(cloned_project_path)
            if possible_core_files:
                logger.info(f"未能精确定位文件，但找到了 {len(possible_core_files)} 个可能的核心文件:")
                for idx, file in enumerate(possible_core_files[:5]):
                    logger.info(f"  {idx+1}. {os.path.relpath(file, cloned_project_path)}")
                # 使用第一个核心文件作为目标文件
                target_file_relative_path = os.path.relpath(possible_core_files[0], cloned_project_path)
                logger.info(f"将使用 {target_file_relative_path} 作为目标文件。")
            else:
                return

        logger.info(f"定位到待修复文件 (相对于项目根目录): {target_file_relative_path}")

        # 收集相关联的上下文文件
        context_files = file_locator.collect_context_files(cloned_project_path, target_file_relative_path)
        logger.info(f"收集了 {len(context_files)} 个相关联的文件作为上下文")
        for ctx_file in list(context_files.keys())[:5]:  # 只显示前5个
            logger.info(f"  - {ctx_file}")

        # 读取原始文件内容
        full_original_file_path = os.path.join(cloned_project_path, target_file_relative_path)
        if not os.path.exists(full_original_file_path):
            logger.error(f"待修复文件不存在于克隆的项目中: {full_original_file_path}")
            return

        with open(full_original_file_path, "r", encoding="utf-8", errors="ignore") as f:
            original_file_content = f.read()

        # AI 生成补丁并测试
        last_error_log = None
        successful_patch = None

        for attempt in range(config.MAX_RETRIES):
            logger.info(f"尝试生成补丁 (第 {attempt + 1}/{config.MAX_RETRIES} 次)...")
            
            # 保存每次尝试的补丁，不论成功与否
            attempt_patch_dir = os.path.join(run_work_dir, f"attempt_{attempt+1}")
            os.makedirs(attempt_patch_dir, exist_ok=True)
            
            # 修改为使用上下文文件的 API 调用
            generated_patch_str = ai_handler.generate_patch_with_context(
                config.DEEPSEEK_API_KEY,
                original_file_content,
                issue_full_description,
                target_file_relative_path, 
                context_files,
                last_error_log 
            )

            if not generated_patch_str:
                logger.warning("AI 未能生成补丁。")
                last_error_log = "AI failed to generate a patch in the previous attempt."
                if attempt < config.MAX_RETRIES -1:
                    time.sleep(5)
                continue
            
            # 保存生成的补丁
            attempt_patch_path = os.path.join(attempt_patch_dir, "patch.diff")
            with open(attempt_patch_path, "w", encoding="utf-8") as pf:
                pf.write(generated_patch_str)
                
            logger.info(f"AI 生成的补丁已保存到 {attempt_patch_path}")
            logger.info(f"补丁内容预览:\n{generated_patch_str[:500]}...\n")

            # 准备测试环境
            temp_test_env_path = docker_runner.prepare_test_environment(
                cloned_project_path, 
                target_file_relative_path, 
                generated_patch_str,
                os.path.join(run_work_dir, f"test_env_{attempt+1}")  # 使用运行目录下的子目录
            )

            if not temp_test_env_path:
                logger.error("准备测试环境失败。")
                last_error_log = "Failed to prepare the test environment with the generated patch."
                continue

            # 定位测试用的 Dockerfile
            test_dockerfile_path = find_test_dockerfile(issue_url, f"{owner}_{repo_name}", config.DOCKERFILES_BASE_PATH)
            if not test_dockerfile_path:
                test_dockerfile_path = find_test_dockerfile(issue_url, repo_name, config.DOCKERFILES_BASE_PATH)

            if not test_dockerfile_path:
                logger.error(f"未能找到适用于 {owner}/{repo_name} 的测试 Dockerfile。请检查 config.DOCKERFILES_BASE_PATH ({config.DOCKERFILES_BASE_PATH})。")
                last_error_log = "Could not find the test Dockerfile for the project."
                cleanup_directory(temp_test_env_path, keep_dir=False)
                break

            logger.info(f"将使用 Dockerfile: {test_dockerfile_path} 进行测试，上下文: {temp_test_env_path}")
            test_success, test_logs = docker_runner.run_test_with_dockerfile(test_dockerfile_path, temp_test_env_path)
            
            # 保存测试日志
            test_log_path = os.path.join(attempt_patch_dir, "test_log.txt")
            with open(test_log_path, "w", encoding="utf-8") as log_file:
                log_file.write(test_logs)
                
            logger.info(f"Docker 测试日志已保存到 {test_log_path}")
            logger.info(f"测试结果: {'成功' if test_success else '失败'}")
            
            # 现在可以安全地清理测试环境，因为日志已保存
            cleanup_directory(temp_test_env_path, keep_dir=False)

            if test_success:
                logger.info(f"补丁成功通过测试! (尝试 {attempt + 1})")
                successful_patch = generated_patch_str
                
                # 保存成功补丁到专门的目录
                success_dir = os.path.join(config.TEMP_WORK_DIR, "successful_patches")
                os.makedirs(success_dir, exist_ok=True)
                patch_filename = f"{owner}_{repo_name}_{issue_number}_{run_id}.diff"
                patch_save_path = os.path.join(success_dir, patch_filename)
                with open(patch_save_path, "w", encoding="utf-8") as pf:
                    pf.write(successful_patch)
                logger.info(f"成功补丁已保存到: {patch_save_path}")
                break
            else:
                logger.warning(f"补丁测试失败 (尝试 {attempt + 1})。")
                last_error_log = test_logs
                if attempt == config.MAX_RETRIES - 1:
                    logger.error("已达到最大重试次数，修复失败。")

        if successful_patch:
            logger.info("流程成功完成。")
            # 记录成功信息到运行日志中
            with open(os.path.join(run_work_dir, "result.txt"), "w", encoding="utf-8") as f:
                f.write("SUCCESS\n")
        else:
            logger.error("所有尝试均失败，未能生成有效补丁。")
            # 记录失败信息到运行日志中
            with open(os.path.join(run_work_dir, "result.txt"), "w", encoding="utf-8") as f:
                f.write("FAILED\n")

    except Exception as e:
        logger.exception(f"处理 Issue {issue_url} 时发生未捕获的异常: {e}")
        # 记录异常信息到运行日志中
        with open(os.path.join(run_work_dir, "result.txt"), "w", encoding="utf-8") as f:
            f.write(f"ERROR\n{str(e)}")
    finally:
        # 不再清理克隆的仓库，让其可以复用
        logger.info(f"Issue {issue_url} 处理完毕。运行结果保存在 {run_work_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="C/C++ Code Fixer using AI and Docker testing.")
    parser.add_argument("issue_url", help="The URL of the GitHub issue to fix.")
    args = parser.parse_args()

    if not os.path.exists(config.TEMP_WORK_DIR):
        os.makedirs(config.TEMP_WORK_DIR)
    
    # 确保 DOCKERFILES_BASE_PATH 存在，如果不存在则给出警告
    if not os.path.isdir(config.DOCKERFILES_BASE_PATH):
        logger.warning(f"指定的 Dockerfile 基础路径 DOCKERFILES_BASE_PATH ('{config.DOCKERFILES_BASE_PATH}') 不存在或不是一个目录。请确保路径正确并包含测试用的 Dockerfile。")
        # 可以选择在这里退出，或者让 find_test_dockerfile 处理找不到文件的情况
        # exit(1)

    process_issue(args.issue_url)