import logging
import os
import shutil
import argparse
import time
import json
from datetime import datetime, timezone # 新增

import config
import github_handler
import file_locator
import ai_handler
import patch_handler
# import docker_runner # 暂时注释掉 Docker 相关

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(config.LOG_FILE, encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# 创建 Logs 目录 (在脚本顶层，与 main.py 同级)
LOGS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Logs")
if not os.path.exists(LOGS_DIR):
    os.makedirs(LOGS_DIR)
    logger.info(f"创建 Trajectory 日志目录: {LOGS_DIR}")

def save_issue_info(issue_details, owner, repo_name, issue_number):
    """保存原始 issue 信息到专门的文件中。"""
    issues_dir = os.path.join(config.TEMP_WORK_DIR, "issues")
    os.makedirs(issues_dir, exist_ok=True)
    
    issue_filename = f"{owner}_{repo_name}_{issue_number}.json"
    issue_file_path = os.path.join(issues_dir, issue_filename)
    
    issue_info = {
        "title": issue_details.get("title", ""),
        "body": issue_details.get("body", ""),
        "url": issue_details.get("url", ""),
        "created_at": issue_details.get("created_at", "").isoformat() if issue_details.get("created_at") else "",
        "comments": issue_details.get("comments", []),
        "default_branch": issue_details.get("default_branch", "")
    }
    
    text_description = f"Title: {issue_info['title']}\n\nBody:\n{issue_info['body']}"
    if issue_info["comments"]:
        text_description += "\n\nComments:\n" + "\n---\n".join(issue_info["comments"])
    
    with open(issue_file_path, "w", encoding="utf-8") as f:
        json.dump(issue_info, f, ensure_ascii=False, indent=2)
        
    text_file_path = os.path.join(issues_dir, f"{owner}_{repo_name}_{issue_number}.txt")
    with open(text_file_path, "w", encoding="utf-8") as f:
        f.write(text_description)
        
    logger.info(f"Issue 信息已保存到 {issue_file_path} 和 {text_file_path}")
    return text_description, issue_file_path, text_file_path

def get_or_clone_repository(owner, repo_name, issue_number, issue_created_at=None, default_branch=None):
    """获取或克隆仓库，并检出到指定版本。优先检查已有仓库。"""
    repos_base_dir = os.path.join(config.TEMP_WORK_DIR, "repositories")
    os.makedirs(repos_base_dir, exist_ok=True)
    
    repo_dir_name = f"{owner}_{repo_name}"
    repo_path = os.path.join(repos_base_dir, repo_dir_name)
    
    repo_url = github_handler.generate_repo_url(owner, repo_name, use_ssh=True) # 考虑配置 use_ssh
    
    logger.info(f"检查是否已存在仓库: {repo_path}")
    checked_out_commit_sha = None # 用于 trajectory
    
    if os.path.exists(os.path.join(repo_path, ".git")):
        logger.info(f"已发现现有仓库: {repo_path}")
        try:
            from git import Repo, InvalidGitRepositoryError, NoSuchPathError
            try:
                repo = Repo(repo_path)
            except (InvalidGitRepositoryError, NoSuchPathError): # 仓库目录存在但不是有效git仓库
                logger.warning(f"目录 {repo_path} 不是有效的Git仓库，将删除并重新克隆。")
                shutil.rmtree(repo_path)
                cloned_path, checked_out_commit_sha = github_handler.clone_repository(repo_url, repo_path, issue_created_at, default_branch)
                return cloned_path, checked_out_commit_sha

            logger.info(f"更新远程仓库...")
            repo.git.fetch("--all", "--prune") # 清理不存在的远程分支
            
            if issue_created_at and default_branch:
                logger.info(f"尝试检出到 {default_branch} 分支在 {issue_created_at.isoformat()} 之前的提交")
                until_date_str = issue_created_at.strftime('%Y-%m-%dT%H:%M:%SZ')
                
                remote_branch_name = f"origin/{default_branch}"
                if remote_branch_name in [ref.name for ref in repo.remote().refs]:
                    commits = list(repo.iter_commits(remote_branch_name, max_count=1, until=until_date_str))
                    if commits:
                        commit_to_checkout = commits[0]
                        checked_out_commit_sha = commit_to_checkout.hexsha
                        logger.info(f"找到提交: {checked_out_commit_sha} (日期: {commit_to_checkout.committed_datetime})")
                        repo.git.checkout(checked_out_commit_sha, b=f"issue_{issue_number}_{int(time.time())}") # 创建新分支检出，避免 detached HEAD
                        logger.info(f"成功检出到新分支上的提交: {checked_out_commit_sha}")
                    else:
                        logger.warning(f"在 {remote_branch_name} 分支上未找到 {issue_created_at.isoformat()} 之前的提交。将使用分支最新提交。")
                        repo.git.checkout(remote_branch_name, b=f"issue_{issue_number}_{int(time.time())}")
                        checked_out_commit_sha = repo.head.commit.hexsha
                else:
                    logger.warning(f"仓库中不存在远程分支: {remote_branch_name}。将尝试使用本地默认分支。")
                    # Fallback or error
            return repo_path, checked_out_commit_sha
        except Exception as e:
            logger.error(f"使用现有仓库时出错: {e}")
            logger.info(f"将尝试删除并重新克隆仓库")
            try:
                shutil.rmtree(repo_path)
            except Exception as rm_e:
                logger.warning(f"删除仓库目录失败: {rm_e}")
    
    logger.info(f"克隆仓库 {repo_url} 到 {repo_path}")
    cloned_path, checked_out_commit_sha = github_handler.clone_repository(repo_url, repo_path, issue_created_at, default_branch)
    return cloned_path, checked_out_commit_sha

# ... (cleanup_directory and find_test_dockerfile can remain as they are for now) ...
def cleanup_directory(dir_path, keep_dir=False):
    """清理目录内容，可选择保留目录本身。"""
    if not os.path.exists(dir_path):
        logger.info(f"目录不存在，无需清理: {dir_path}")
        return
        
    logger.info(f"开始清理目录: {dir_path}")
    try:
        if keep_dir:
            for item in os.listdir(dir_path):
                item_path = os.path.join(dir_path, item)
                if os.path.isfile(item_path):
                    os.unlink(item_path)
                elif os.path.isdir(item_path):
                    shutil.rmtree(item_path)
            logger.info(f"目录内容已清理，保留目录: {dir_path}")
        else:
            import stat
            def on_rm_error(func, path, exc_info):
                if not os.access(path, os.W_OK):
                    os.chmod(path, stat.S_IWUSR)
                    func(path)
            shutil.rmtree(dir_path, onerror=on_rm_error)
            logger.info(f"目录已完全清理: {dir_path}")
    except Exception as e:
        logger.error(f"清理目录 {dir_path} 失败: {e}")

def find_test_dockerfile(issue_url: str, project_name: str, base_dockerfile_path: str) -> str | None:
    # ... (existing implementation) ...
    return None


def process_issue(issue_url: str):
    logger.info(f"开始处理 Issue: {issue_url}")
    
    run_id = int(time.time()) # Unique ID for this run
    trajectory_data = {
        "run_id": run_id,
        "issue_url": issue_url,
        "timestamp_start": datetime.now(timezone.utc).isoformat(),
        "status_overall": "PENDING",
        "stages": {
            "issue_parsing": {"status": "PENDING"},
            "repository_setup": {"status": "PENDING"},
            "test_case_retrieval": {"status": "PENDING"},
            "file_localization": {"status": "PENDING", "details": []},
            "context_collection": {"status": "PENDING"},
            "ai_patch_generation": {"status": "PENDING", "attempts": []},
            "patch_testing": {"status": "SKIPPED_FOR_NOW"}, 
        },
        "results": {
            "final_patch_path": None,
            "error_message": None,
        },
        "timestamp_end": None,
    }
    
    trajectory_filename = f"trajectory_init_{run_id}.json" 
    trajectory_filepath = os.path.join(LOGS_DIR, trajectory_filename)

    try:
        # ... (initial setup, URL parsing, issue details, repo cloning) ...
        # Ensure this part is robust as in your previous version
        if not config.GITHUB_TOKEN:
            raise ValueError("GITHUB_TOKEN 未配置。请在 .env 文件中设置。")
        if not config.DEEPSEEK_API_KEY:
            raise ValueError("DEEPSEEK_API_KEY 未配置。请在 .env 文件中设置。")

        parsed_url = github_handler.parse_issue_url(issue_url)
        if not parsed_url:
            raise ValueError(f"无法解析 Issue URL: {issue_url}")
        
        owner, repo_name, issue_number = parsed_url
        trajectory_data["stages"]["issue_parsing"]["parsed_owner"] = owner
        trajectory_data["stages"]["issue_parsing"]["parsed_repo_name"] = repo_name
        trajectory_data["stages"]["issue_parsing"]["parsed_issue_number"] = issue_number
        
        trajectory_filename = f"{owner}_{repo_name}_{issue_number}_{run_id}.json"
        trajectory_filepath = os.path.join(LOGS_DIR, trajectory_filename)

        run_dir_name = f"{owner}_{repo_name}_{issue_number}_{run_id}"
        run_work_dir = os.path.join(config.TEMP_WORK_DIR, "runs", run_dir_name)
        os.makedirs(run_work_dir, exist_ok=True)
        trajectory_data["run_work_dir"] = run_work_dir

        issue_details = github_handler.get_issue_details(owner, repo_name, issue_number, config.GITHUB_TOKEN)
        if not issue_details:
            raise RuntimeError("获取 Issue 详细信息失败。")
        
        trajectory_data["stages"]["issue_parsing"]["original_title"] = issue_details.get("title")
        trajectory_data["stages"]["issue_parsing"]["original_url"] = issue_details.get("html_url")
        trajectory_data["stages"]["issue_parsing"]["status"] = "SUCCESS"
        
        issue_full_description, saved_json_path, saved_text_path = save_issue_info(issue_details, owner, repo_name, issue_number)
        trajectory_data["stages"]["issue_parsing"]["saved_json_path"] = saved_json_path
        trajectory_data["stages"]["issue_parsing"]["saved_text_path"] = saved_text_path
        
        issue_created_at = issue_details.get("created_at")
        default_branch = issue_details.get("default_branch")
        
        cloned_project_path, checked_out_commit = get_or_clone_repository(owner, repo_name, issue_number, issue_created_at, default_branch)
        if not cloned_project_path:
            raise RuntimeError(f"获取仓库代码失败: {owner}/{repo_name}")
        
        trajectory_data["stages"]["repository_setup"]["path"] = cloned_project_path
        trajectory_data["stages"]["repository_setup"]["checked_out_commit"] = checked_out_commit
        trajectory_data["stages"]["repository_setup"]["status"] = "SUCCESS"
        # --- 在这里添加获取测试用例的逻辑 ---
        test_cases_for_ai = []
        trajectory_data["stages"]["test_case_retrieval"]["status"] = "ATTEMPTING"
        try:
            logger.info(f"尝试为 Issue #{issue_number} 获取相关的测试用例...")
            retrieved_test_cases = github_handler.get_test_cases_for_issue(
                owner, repo_name, issue_number, config.GITHUB_TOKEN
            )
            if retrieved_test_cases:
                test_cases_for_ai = retrieved_test_cases
                logger.info(f"成功获取到 {len(test_cases_for_ai)} 个相关的测试用例。")
                trajectory_data["stages"]["test_case_retrieval"]["status"] = "SUCCESS"
                trajectory_data["stages"]["test_case_retrieval"]["count"] = len(test_cases_for_ai)
                # 为了避免 trajectory 文件过大，只记录测试用例的名称和描述（如果可用）
                trajectory_data["stages"]["test_case_retrieval"]["details"] = [
                    {"name": tc.get("name"), "description": tc.get("description", "N/A")} 
                    for tc in test_cases_for_ai
                ]
                # 也可以考虑将测试用例内容保存到 run_work_dir 下的文件中，然后在 trajectory 中记录路径
                for i, tc_data in enumerate(test_cases_for_ai):
                    tc_filename = f"retrieved_test_case_{i+1}_{os.path.basename(tc_data.get('name', 'unknown.txt'))}"
                    tc_save_path = os.path.join(run_work_dir, "retrieved_test_cases")
                    os.makedirs(tc_save_path, exist_ok=True)
                    full_tc_path = os.path.join(tc_save_path, tc_filename)
                    with open(full_tc_path, "w", encoding="utf-8") as tcf:
                        tcf.write(tc_data.get("content", ""))
                    # 更新 trajectory 中的 details，指向保存的路径
                    if "details" in trajectory_data["stages"]["test_case_retrieval"] and \
                       i < len(trajectory_data["stages"]["test_case_retrieval"]["details"]):
                        trajectory_data["stages"]["test_case_retrieval"]["details"][i]["saved_path"] = full_tc_path
            else:
                logger.info("未能找到与 Issue 关联的明确测试用例。")
                trajectory_data["stages"]["test_case_retrieval"]["status"] = "NOT_FOUND"
        except Exception as tc_e:
            logger.error(f"获取测试用例时出错: {tc_e}")
            trajectory_data["stages"]["test_case_retrieval"]["status"] = "ERROR"
            trajectory_data["stages"]["test_case_retrieval"]["message"] = str(tc_e)
        # --- 测试用例获取逻辑结束 ---
        logger.info(f"Issue 标题: {issue_details['title']}")

        target_file_relative_path = file_locator.find_file_to_fix(issue_details, cloned_project_path)
        loc_stage_summary = {"strategy": "primary_search"}
        if not target_file_relative_path:
            loc_stage_summary["status"] = "FAILED"
            trajectory_data["stages"]["file_localization"]["details"].append(loc_stage_summary)
            logger.error("未能定位到需要修复的文件。尝试备选策略...")
            
            possible_core_files = file_locator.find_core_files(cloned_project_path)
            fallback_loc_summary = {"strategy": "fallback_core_files"}
            if possible_core_files:
                fallback_loc_summary["status"] = "FOUND_CANDIDATES"
                fallback_loc_summary["candidates_preview"] = [os.path.relpath(f, cloned_project_path) for f in possible_core_files[:3]]
                target_file_relative_path = os.path.relpath(possible_core_files[0], cloned_project_path)
                fallback_loc_summary["selected"] = target_file_relative_path
                logger.info(f"备选策略选中: {target_file_relative_path}")
            else:
                fallback_loc_summary["status"] = "FAILED"
                trajectory_data["stages"]["file_localization"]["details"].append(fallback_loc_summary)
                raise RuntimeError("文件定位失败，包括备选策略。")
            trajectory_data["stages"]["file_localization"]["details"].append(fallback_loc_summary)
        else:
            loc_stage_summary["status"] = "SUCCESS"
            loc_stage_summary["selected"] = target_file_relative_path
            trajectory_data["stages"]["file_localization"]["details"].append(loc_stage_summary)

        trajectory_data["stages"]["file_localization"]["final_selected_file"] = target_file_relative_path
        trajectory_data["stages"]["file_localization"]["status"] = "SUCCESS"
        logger.info(f"定位到待修复文件: {target_file_relative_path}")

        context_files = file_locator.collect_context_files(cloned_project_path, target_file_relative_path)
        trajectory_data["stages"]["context_collection"]["count"] = len(context_files)
        trajectory_data["stages"]["context_collection"]["paths_preview"] = list(context_files.keys())[:5]
        trajectory_data["stages"]["context_collection"]["status"] = "SUCCESS"
        logger.info(f"收集了 {len(context_files)} 个上下文文件")

        full_original_file_path = os.path.join(cloned_project_path, target_file_relative_path)
        if not os.path.exists(full_original_file_path):
            raise FileNotFoundError(f"目标文件不存在: {full_original_file_path}")

        with open(full_original_file_path, "r", encoding="utf-8", errors="ignore") as f:
            original_file_content = f.read()

        last_error_log_for_ai = None # Renamed to avoid confusion with trajectory error log
        successful_patch_str = None
        trajectory_data["stages"]["ai_patch_generation"]["status"] = "IN_PROGRESS"

        for attempt in range(config.MAX_RETRIES):
            attempt_start_time = datetime.now(timezone.utc).isoformat()
            logger.info(f"AI补丁生成尝试 {attempt + 1}/{config.MAX_RETRIES}...")
            
            attempt_patch_dir = os.path.join(run_work_dir, f"attempt_{attempt+1}")
            os.makedirs(attempt_patch_dir, exist_ok=True)
            
            ai_result = ai_handler.generate_patch_with_context(
                config.DEEPSEEK_API_KEY,
                original_file_content,
                issue_full_description,
                target_file_relative_path, 
                context_files,
                last_error_log_for_ai,
                test_cases=test_cases_for_ai, 
                stream=False # Explicitly false for now
            )
            
            # Save prompt and raw response to files for easier inspection if they are very large
            prompt_log_path = os.path.join(attempt_patch_dir, "prompt.txt")
            with open(prompt_log_path, "w", encoding="utf-8") as pf:
                pf.write(ai_result.get("prompt_sent") or "Prompt not available.")
            
            response_log_path = os.path.join(attempt_patch_dir, "response.txt")
            with open(response_log_path, "w", encoding="utf-8") as rf:
                rf.write(ai_result.get("raw_response") or "Raw response not available.")

            attempt_log = {
                "attempt_number": attempt + 1,
                "timestamp_start": attempt_start_time,
                "had_previous_error_log": bool(last_error_log_for_ai),
                "prompt_saved_path": prompt_log_path, # Path to the saved prompt
                "raw_response_saved_path": response_log_path, # Path to the saved raw response
                "api_status_code": ai_result.get("status_code"),
                "api_error_message": ai_result.get("error_message"),
                "patch_generated": bool(ai_result.get("extracted_patch")),
                "patch_saved_path": None,
                "test_status": "SKIPPED_FOR_NOW",
                "timestamp_end": None,
            }

            if not ai_result.get("api_call_successful") or not ai_result.get("extracted_patch"):
                logger.warning(f"AI 未能生成有效补丁或API调用失败。错误: {ai_result.get('error_message')}")
                last_error_log_for_ai = f"Previous attempt failed. API Status: {ai_result.get('status_code')}. Error: {ai_result.get('error_message')}. Raw Response Snippet: {(ai_result.get('raw_response') or '')[:200]}"
                attempt_log["timestamp_end"] = datetime.now(timezone.utc).isoformat()
                trajectory_data["stages"]["ai_patch_generation"]["attempts"].append(attempt_log)
                if attempt < config.MAX_RETRIES - 1:
                    time.sleep(config.RETRY_DELAY_SECONDS)
                continue
            
            generated_patch_str = ai_result["extracted_patch"]
            attempt_patch_path = os.path.join(attempt_patch_dir, "patch.diff")
            with open(attempt_patch_path, "w", encoding="utf-8") as pf:
                pf.write(generated_patch_str)
            attempt_log["patch_saved_path"] = attempt_patch_path
            logger.info(f"AI 生成的补丁已保存到 {attempt_patch_path}")
            
            # --- DOCKER TESTING SKIPPED ---
            logger.info(f"补丁生成成功 (尝试 {attempt + 1}). Docker测试已跳过。")
            successful_patch_str = generated_patch_str
            attempt_log["test_status"] = "SUCCESS_GENERATED_NO_TEST"
            attempt_log["timestamp_end"] = datetime.now(timezone.utc).isoformat()
            trajectory_data["stages"]["ai_patch_generation"]["attempts"].append(attempt_log)
            
            success_dir = os.path.join(config.TEMP_WORK_DIR, "successful_patches")
            os.makedirs(success_dir, exist_ok=True)
            final_patch_filename = f"{owner}_{repo_name}_{issue_number}_{run_id}.diff"
            final_patch_save_path = os.path.join(success_dir, final_patch_filename)
            with open(final_patch_save_path, "w", encoding="utf-8") as pf:
                pf.write(successful_patch_str)
            trajectory_data["results"]["final_patch_path"] = final_patch_save_path
            logger.info(f"成功补丁已保存到: {final_patch_save_path}")
            break 
            # --- END SKIPPED DOCKER ---

        if successful_patch_str:
            trajectory_data["stages"]["ai_patch_generation"]["status"] = "SUCCESS"
            trajectory_data["status_overall"] = "SUCCESS_PATCH_GENERATED"
            logger.info("流程成功完成 (Docker测试已跳过)。")
            # ... (save result.txt)
        else:
            trajectory_data["stages"]["ai_patch_generation"]["status"] = "FAILED_MAX_RETRIES"
            trajectory_data["status_overall"] = "FAILED_PATCH_GENERATION"
            logger.error("所有尝试均失败，未能生成有效补丁。")
            # ... (save result.txt)

    except Exception as e:
        # ... (existing exception handling) ...
        logger.exception(f"处理 Issue {issue_url} 时发生未捕获的异常: {e}")
        trajectory_data["status_overall"] = "ERROR_UNHANDLED"
        trajectory_data["results"]["error_message"] = str(e)
        import traceback
        trajectory_data["results"]["error_traceback_snippet"] = traceback.format_exc()[:1000]
        
        if 'run_work_dir' in locals() and run_work_dir:
             with open(os.path.join(run_work_dir, "result.txt"), "w", encoding="utf-8") as f:
                f.write(f"ERROR_UNHANDLED\n{str(e)}")

    finally:
        # ... (existing finally block to save trajectory_data) ...
        trajectory_data["timestamp_end"] = datetime.now(timezone.utc).isoformat()
        if not os.path.isabs(trajectory_filepath) or os.path.dirname(trajectory_filepath) != LOGS_DIR :
            trajectory_filename = f"trajectory_error_{trajectory_data.get('run_id', int(time.time()))}.json"
            trajectory_filepath = os.path.join(LOGS_DIR, trajectory_filename)

        try:
            with open(trajectory_filepath, "w", encoding="utf-8") as f:
                json.dump(trajectory_data, f, ensure_ascii=False, indent=2)
            logger.info(f"Trajectory 日志已保存到: {trajectory_filepath}")
        except Exception as traj_save_e:
            logger.error(f"保存 Trajectory 日志失败: {traj_save_e} ({trajectory_filepath})")

        if 'run_work_dir' in locals() and run_work_dir:
            logger.info(f"Issue {issue_url} 处理完毕。运行数据保存在 {run_work_dir}")
        else:
            logger.info(f"Issue {issue_url} 处理完毕。")


if __name__ == "__main__":
    # ... (argparse and main call) ...
    parser = argparse.ArgumentParser(description="C/C++ Code Fixer using AI.")
    parser.add_argument("issue_url", help="The URL of the GitHub issue to fix.")
    args = parser.parse_args()

    if not os.path.exists(config.TEMP_WORK_DIR):
        os.makedirs(config.TEMP_WORK_DIR)
    
    process_issue(args.issue_url)