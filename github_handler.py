import logging
import git
import re
from github import Github, GithubException, PullRequest, Commit # 新增 Commit
import datetime
import os
from typing import List, Dict, Optional, Tuple, Set

logger = logging.getLogger(__name__)

def parse_issue_url(issue_url: str) -> Optional[Tuple[str, str, int]]: # 修改返回类型提示
    """解析 GitHub issue URL，提取 owner, repo, issue_number。"""
    match = re.match(r"https://github\.com/([^/]+)/([^/]+)/issues/(\d+)", issue_url)
    if match:
        owner, repo, issue_number = match.groups()
        return owner, repo, int(issue_number)
    logger.error(f"无法解析 Issue URL: {issue_url}")
    return None

def generate_repo_url(owner: str, repo: str, use_ssh: bool = True) -> str:
    """
    根据所有者和仓库名生成 GitHub 仓库 URL
    
    :param owner: 仓库所有者 (用户名或组织名)
    :param repo: 仓库名
    :param use_ssh: 是否使用 SSH 格式 (默认为 True)
    :return: 仓库 URL
    """
    if use_ssh:
        return f"git@github.com:{owner}/{repo}.git"
    else:
        return f"https://github.com/{owner}/{repo}.git"

def get_issue_details(owner: str, repo_name: str, issue_number: int, token: str) -> Optional[Dict]: # 修改返回类型提示
    """获取 issue 的标题、描述、创建时间、评论和仓库默认分支等信息。"""
    try:
        g = Github(token)
        repo_obj = g.get_repo(f"{owner}/{repo_name}")
        issue = repo_obj.get_issue(number=issue_number)
        
        # 获取与 issue 关联的 Pull Requests
        linked_prs_info = []
        # 检查 issue 的 timeline 事件，看是否有 PR 关闭了此 issue
        # 这是一个比较可靠的方式，但 API 调用可能较多
        # 另一种方式是搜索所有 PR，看其 body 或 title 是否引用了此 issue
        # 这里我们先简化，假设 PR 的 body 中会明确提到 "fixes #issue_number" 或 "closes #issue_number"
        # 或者 issue 本身有 linked_pull_request (但这不总是直接可用或准确)

        # 尝试通过搜索PR来找到关联的PR
        # query = f"repo:{owner}/{repo_name} is:pr mentions-issue:{issue_number} is:merged"
        # try:
        #     for pr_search_item in g.search_issues(query=query): # search_issues can find PRs
        #         if pr_search_item.is_pull_request(): # Ensure it's a PR
        #             pr = repo_obj.get_pull(pr_search_item.number)
        #             linked_prs_info.append({"number": pr.number, "url": pr.html_url, "merged_at": pr.merged_at, "merge_commit_sha": pr.merge_commit_sha})
        # except GithubException as search_e:
        #     logger.warning(f"搜索关联PR时出错: {search_e}")
        # 这种搜索方式可能不总是完美，实际项目中可能需要更复杂的逻辑

        details = {
            "title": issue.title,
            "body": issue.body,
            "url": issue.html_url,
            "comments": [comment.body for comment in issue.get_comments()],
            "created_at": issue.created_at,
            "default_branch": repo_obj.default_branch,
            # "linked_prs": linked_prs_info # 暂时注释，因为获取PR的逻辑需要更完善
        }
        logger.info(f"成功获取 Issue #{issue_number} 的详细信息来自 {owner}/{repo_name}")
        return details
    except GithubException as e:
        logger.error(f"获取 Issue 详细信息失败: {e}")
        return None
    except Exception as e:
        logger.error(f"获取 Issue 详细信息时发生未知错误: {e}")
        return None


def clone_repository(repo_url: str, clone_to_path: str, checkout_before_datetime: Optional[datetime.datetime] = None, default_branch: Optional[str] = None) -> Optional[Tuple[str, Optional[str]]]: # 返回路径和commit SHA
    """
    克隆指定的 GitHub 仓库到本地临时目录。
    如果提供了 checkout_before_datetime 和 default_branch，则会尝试检出到该日期之前的最新提交。
    返回 (克隆路径, 检出的 commit SHA) 或 None。
    """
    checked_out_commit_sha = None
    try:
        logger.info(f"开始克隆仓库 {repo_url} 到 {clone_to_path}")
        
        if os.path.exists(os.path.join(clone_to_path, ".git")):
            logger.warning(f"目录 {clone_to_path} 已存在 .git 目录。将尝试使用现有仓库。")
            cloned_repo = git.Repo(clone_to_path)
            # 对于复用仓库，需要确保它处于正确的状态或进行更新
            # origin = cloned_repo.remotes.origin
            # origin.fetch() # Fetch updates
            # logger.info(f"已拉取仓库 {repo_url} 的最新更改到 {clone_to_path}")
        else:
            os.makedirs(clone_to_path, exist_ok=True) # 确保目录存在
            cloned_repo = git.Repo.clone_from(repo_url, clone_to_path)
            logger.info(f"仓库 {repo_url} 成功克隆到 {clone_to_path}")

        if checkout_before_datetime and default_branch:
            logger.info(f"尝试检出到 {default_branch} 分支在 {checkout_before_datetime.isoformat()}之前的提交")
            try:
                until_date_str = checkout_before_datetime.strftime('%Y-%m-%dT%H:%M:%SZ')
                
                # 尝试从远程分支获取提交，如果本地分支不是最新的
                remote_branch_name = f"origin/{default_branch}"
                try:
                    cloned_repo.remotes.origin.fetch(default_branch)
                except git.GitCommandError as fetch_err:
                    logger.warning(f"获取远程分支 {default_branch} 失败: {fetch_err}. 将尝试使用本地分支。")


                commits = list(cloned_repo.iter_commits(remote_branch_name, max_count=1, until=until_date_str))
                if not commits: # 如果远程分支没有，尝试本地分支
                    commits = list(cloned_repo.iter_commits(default_branch, max_count=1, until=until_date_str))

                if commits:
                    commit_to_checkout = commits[0]
                    checked_out_commit_sha = commit_to_checkout.hexsha
                    logger.info(f"找到提交: {checked_out_commit_sha} (日期: {commit_to_checkout.committed_datetime})")
                    cloned_repo.git.checkout(checked_out_commit_sha)
                    logger.info(f"成功检出到提交: {checked_out_commit_sha}")
                else:
                    logger.warning(f"在 {default_branch} 分支上未找到 {checkout_before_datetime.isoformat()} 之前的提交。将尝试检出分支的最新状态。")
                    try:
                        cloned_repo.git.checkout(default_branch)
                        checked_out_commit_sha = cloned_repo.head.commit.hexsha
                        logger.info(f"已检出到 {default_branch} 的最新提交: {checked_out_commit_sha}")
                    except git.GitCommandError as e_checkout_latest:
                        logger.error(f"检出 {default_branch} 最新状态失败: {e_checkout_latest}")
                        # 如果检出默认分支也失败，可能仓库有问题
            except git.GitCommandError as e:
                logger.error(f"检出到特定提交失败: {e}")
            except Exception as e:
                logger.error(f"检出到特定提交时发生未知错误: {e}")
        else: # 没有指定日期，检出默认分支的最新
            try:
                if default_branch:
                    cloned_repo.git.checkout(default_branch)
                    checked_out_commit_sha = cloned_repo.head.commit.hexsha
                    logger.info(f"已检出到 {default_branch} 的最新提交: {checked_out_commit_sha}")
                else: # 如果连默认分支都没有，就停在克隆后的状态
                    checked_out_commit_sha = cloned_repo.head.commit.hexsha
                    logger.info(f"仓库保持在克隆后的默认提交: {checked_out_commit_sha}")
            except Exception as e_checkout_default:
                 logger.error(f"检出默认分支 {default_branch} 时出错: {e_checkout_default}")


        return clone_to_path, checked_out_commit_sha
    except git.GitCommandError as e:
        logger.error(f"克隆仓库 {repo_url} 失败: {e}")
        return None, None
    except Exception as e:
        logger.error(f"克隆仓库时发生未知错误: {e}")
        return None, None

def get_test_cases_for_issue(owner: str, repo_name: str, issue_number: int, token: str) -> List[Dict[str, str]]:
    """
    尝试获取与 GitHub Issue 关联的、在修复提交中添加或修改的测试用例。
    返回一个列表，每个元素是 {"name": "test_file.cpp", "content": "...", "description": "..."}
    """
    test_cases: List[Dict[str, str]] = []
    try:
        g = Github(token)
        repo_obj = g.get_repo(f"{owner}/{repo_name}")
        issue = repo_obj.get_issue(number=issue_number)

        fixing_commit_shas: Set[str] = set() # 使用 set 避免重复

        # 方案 1: 检查 Issue 的 Timeline 事件中是否有 'cross_referenced' 事件指向一个已合并的 PR
        # 或者 'closed' 事件是否由一个 PR 触发 (这部分 PyGithub 的具体属性可能需要细致探查)
        logger.info(f"检查 Issue #{issue_number} 的 timeline 事件...")
        for event in issue.get_timeline():
            # logger.debug(f"Timeline event: {event.event}, actor: {event.actor.login if event.actor else 'N/A'}, commit_id: {event.commit_id}")
            if event.event == "cross-referenced" and event.source and event.source.issue and event.source.issue.pull_request:
                # 这个事件表示 Issue 被一个 PR 引用了
                pr_data = event.source.issue # 这实际上是一个 Issue 对象，代表了那个 PR
                if pr_data.number: # 确保能拿到 PR 号
                    try:
                        pr = repo_obj.get_pull(pr_data.number)
                        if pr.merged and pr.merge_commit_sha:
                            logger.info(f"Timeline: Issue #{issue_number} 被已合并 PR #{pr.number} 引用。Merge SHA: {pr.merge_commit_sha}")
                            fixing_commit_shas.add(pr.merge_commit_sha)
                        # elif pr.closed_at and not pr.merged and pr.head.sha: # 如果PR关闭但未合并，有时修复在PR的分支提交中
                        #     logger.info(f"Timeline: Issue #{issue_number} 被已关闭但未合并的 PR #{pr.number} 引用。Head SHA: {pr.head.sha}")
                        #     # 这种情况比较复杂，修复可能在PR的多个提交中，merge_commit_sha 不存在
                        #     # fixing_commit_shas.add(pr.head.sha) # 考虑PR的最后一个提交
                    except GithubException as pr_err:
                        logger.warning(f"获取 PR #{pr_data.number} 详细信息失败: {pr_err}")
            elif event.event == "closed" and event.commit_id:
                # Issue 被一个 commit 直接关闭 (可能是直接提交，也可能是 PR 的一部分)
                # 这种方式获取的 commit_id 可能不是 merge commit，而是 PR 分支上的 commit
                # 如果是 PR merge 关闭的，通常 merge commit 更能代表最终状态
                logger.info(f"Timeline: Issue #{issue_number} 可能被提交 {event.commit_id} 直接关闭。")
                # fixing_commit_shas.add(event.commit_id) # 暂时优先PR的merge commit

        # 方案 2: 如果 Timeline 中没找到，或者想更全面，搜索已合并的 PR
        if not fixing_commit_shas:
            logger.info(f"Timeline 未明确找到修复提交/PR，尝试搜索已合并的 PRs 提及 Issue #{issue_number}...")
            # 关键词可以包括 "fixes #issue_number", "closes #issue_number", "resolves #issue_number"
            # GitHub 会自动链接这些关键词
            # 搜索所有已关闭的 PR (包括已合并的)
            for pr in repo_obj.get_pulls(state="closed", sort="updated", direction="desc", base=issue.repository.default_branch):
                if pr.merged and pr.merge_commit_sha:
                    # 检查 PR body, title, 或 commits 是否引用了 issue
                    # PyGithub 的 PR 对象没有直接的 'fixes_issues' 列表
                    # 我们需要自己检查文本
                    text_to_search = (pr.title or "") + " " + (pr.body or "")
                    # 也可以检查 PR 的提交信息，但这会增加 API 调用
                    # for pr_commit in pr.get_commits():
                    #    text_to_search += " " + pr_commit.commit.message

                    # 简单的关键词检查
                    issue_ref_patterns = [
                        rf"\bfixes\s+#{issue_number}\b",
                        rf"\bcloses\s+#{issue_number}\b",
                        rf"\bresolves\s+#{issue_number}\b",
                        rf"#{issue_number}\b" # 有时只是简单提及
                    ]
                    if any(re.search(pattern, text_to_search, re.IGNORECASE) for pattern in issue_ref_patterns):
                        logger.info(f"搜索: 找到已合并 PR #{pr.number} (Merge SHA: {pr.merge_commit_sha}) 提及 Issue #{issue_number}")
                        fixing_commit_shas.add(pr.merge_commit_sha)
                        # break # 可以找到第一个就停止，或者收集所有相关的

        if not fixing_commit_shas:
            logger.warning(f"未能找到明确修复 Issue #{issue_number} 的提交或已合并 PR。无法提取测试用例。")
            return test_cases
        
        logger.info(f"找到以下潜在的修复提交 SHAs for Issue #{issue_number}: {list(fixing_commit_shas)}")

        for commit_sha in fixing_commit_shas:
            try:
                commit: Commit = repo_obj.get_commit(sha=commit_sha) # Commit 对象
                logger.info(f"检查修复提交 {commit_sha} 中的文件变更...")
                for file_in_commit in commit.files: # file_in_commit 是 GithubFile 对象
                    filename = file_in_commit.filename # 保持原始大小写，因为 get_contents 需要
                    filename_lower = filename.lower()
                    
                    is_test = ("test/" in filename_lower or "tests/" in filename_lower or 
                               "example/" in filename_lower or "examples/" in filename_lower or
                               filename_lower.startswith("test_") or "_test." in filename_lower or
                               filename_lower.endswith("_test.cpp") or filename_lower.endswith("_test.h") or
                               filename_lower.endswith(".test") or filename_lower.endswith(".tst"))

                    # 我们需要的是添加或修改的测试文件
                    if is_test and (file_in_commit.status == 'added' or file_in_commit.status == 'modified'):
                        logger.info(f"在修复提交 {commit_sha} 中找到测试文件变更: {filename} (Status: {file_in_commit.status})")
                        try:
                            # 获取该 commit 下该文件的完整内容
                            content_obj = repo_obj.get_contents(filename, ref=commit_sha)
                            file_content_str = content_obj.decoded_content.decode('utf-8', errors='ignore')
                            
                            test_cases.append({
                                "name": filename,
                                "content": file_content_str,
                                "description": f"Test case from commit {commit_sha[:7]} for issue #{issue_number}"
                            })
                            logger.info(f"已提取测试用例: {filename}")
                        except GithubException as getContentErr:
                            # 有时文件可能在 commit 中被列为修改，但实际上是重命名或删除后重新添加，导致 get_contents 失败
                            logger.error(f"无法获取文件 {filename} 在提交 {commit_sha} 时的内容: {getContentErr}. 可能文件已重命名或模式更改。")
                        except Exception as e_file_content:
                             logger.error(f"处理文件 {filename} 内容时出错: {e_file_content}")
            except GithubException as getCommitErr:
                logger.error(f"无法获取提交 {commit_sha} 的详细信息: {getCommitErr}")
            except Exception as e_commit_proc:
                logger.error(f"处理提交 {commit_sha} 时发生未知错误: {e_commit_proc}")
                
    # ... (rest of the function) ...
    except GithubException as e:
        logger.error(f"获取 Issue #{issue_number} 的测试用例时发生 GitHub API 错误: {e}")
    except Exception as e: #捕获更广泛的异常
        logger.error(f"获取 Issue #{issue_number} 的测试用例时发生未知错误: {e}", exc_info=True) # 添加 exc_info=True
        
    if not test_cases:
        logger.info(f"未能为 Issue #{issue_number} 提取到任何测试用例。")
    return test_cases