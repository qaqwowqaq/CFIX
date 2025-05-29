import logging
import git
import re
from github import Github, GithubException
import datetime
import os

logger = logging.getLogger(__name__)

def parse_issue_url(issue_url: str) -> tuple[str, str, int] | None:
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

def get_issue_details(owner: str, repo_name: str, issue_number: int, token: str) -> dict | None:
    """获取 issue 的标题、描述、创建时间、评论和仓库默认分支等信息。"""
    try:
        g = Github(token)
        repo_obj = g.get_repo(f"{owner}/{repo_name}")
        issue = repo_obj.get_issue(number=issue_number)
        details = {
            "title": issue.title,
            "body": issue.body,
            "url": issue.html_url,
            "comments": [comment.body for comment in issue.get_comments()],
            "created_at": issue.created_at, # 获取 issue 创建时间
            "default_branch": repo_obj.default_branch # 获取仓库默认分支
        }
        logger.info(f"成功获取 Issue #{issue_number} 的详细信息来自 {owner}/{repo_name}")
        return details
    except GithubException as e:
        logger.error(f"获取 Issue 详细信息失败: {e}")
        return None
    except Exception as e:
        logger.error(f"获取 Issue 详细信息时发生未知错误: {e}")
        return None


def clone_repository(repo_url: str, clone_to_path: str, checkout_before_datetime: datetime.datetime | None = None, default_branch: str | None = None) -> str | None:
    """
    克隆指定的 GitHub 仓库到本地临时目录。
    如果提供了 checkout_before_datetime 和 default_branch，则会尝试检出到该日期之前的最新提交。
    """
    try:
        logger.info(f"开始克隆仓库 {repo_url} 到 {clone_to_path}")
        
        if os.path.exists(os.path.join(clone_to_path, ".git")):
            logger.warning(f"目录 {clone_to_path} 已存在 .git 目录。假设是之前的克隆，将尝试拉取。")
            # 对于检出到特定提交的场景，通常我们期望一个干净的克隆。
            # main.py 中的逻辑会为每次运行创建唯一目录，所以这里通常是新克隆。
            # 如果确实复用目录，并且需要检出到特定旧提交，则拉取后检出可能不是预期行为。
            # 为简化，这里保持原有逻辑，但依赖 main.py 提供干净路径。
            cloned_repo = git.Repo(clone_to_path)
            origin = cloned_repo.remotes.origin
            origin.pull()
            logger.info(f"已拉取仓库 {repo_url} 的最新更改到 {clone_to_path}")
        else:
            os.makedirs(os.path.dirname(clone_to_path), exist_ok=True)  # 确保父目录存在
            cloned_repo = git.Repo.clone_from(repo_url, clone_to_path)
            logger.info(f"仓库 {repo_url} 成功克隆到 {clone_to_path}")

        if checkout_before_datetime and default_branch:
            logger.info(f"尝试检出到 {default_branch} 分支在 {checkout_before_datetime.isoformat()}之前的提交")
            try:
                # iter_commits 的 'until' 参数是指在此时间点之前的提交 (不包括该时间点)
                # 我们需要找到的是严格小于 issue 创建时间的提交
                # GitPython 的 until 似乎是 exclusive upper bound.
                # format datetime to string that git understands
                until_date_str = checkout_before_datetime.strftime('%Y-%m-%dT%H:%M:%SZ')
                
                commits = list(cloned_repo.iter_commits(default_branch, max_count=1, until=until_date_str))
                if commits:
                    commit_to_checkout = commits[0]
                    logger.info(f"找到提交: {commit_to_checkout.hexsha} (日期: {commit_to_checkout.committed_datetime})")
                    cloned_repo.git.checkout(commit_to_checkout.hexsha)
                    logger.info(f"成功检出到提交: {commit_to_checkout.hexsha}")
                else:
                    logger.warning(f"在 {default_branch} 分支上未找到 {checkout_before_datetime.isoformat()} 之前的提交。仓库将保持在默认分支的最新状态。")
            except git.GitCommandError as e:
                logger.error(f"检出到特定提交失败: {e}")
                # 根据需求，这里可以决定是否返回 None 或让其停留在当前状态
                # return None 
            except Exception as e:
                logger.error(f"检出到特定提交时发生未知错误: {e}")

        return clone_to_path
    except git.GitCommandError as e:
        logger.error(f"克隆仓库 {repo_url} 失败: {e}")
        return None
    except Exception as e:
        logger.error(f"克隆仓库时发生未知错误: {e}")
        return None