import logging
import subprocess
import os

logger = logging.getLogger(__name__)

def apply_patch_to_file(original_file_path: str, patch_str: str, project_root: str) -> bool:
    """
    将 AI 生成的补丁字符串应用到文件。
    使用 `patch` 命令行工具。
    :param original_file_path: 相对于 project_root 的文件路径
    :param patch_str: unified diff 格式的补丁内容
    :param project_root: 项目的根目录，patch 命令将在此目录下执行
    :return: True 如果成功应用补丁，否则 False
    """
    if not patch_str.strip():
        logger.warning("补丁字符串为空，不执行任何操作。")
        return False # 或者 True，取决于如何定义空补丁

    patch_file_path = os.path.join(project_root, "temp_patch.diff")
    full_original_file_path = os.path.join(project_root, original_file_path)

    if not os.path.exists(full_original_file_path):
        logger.error(f"原始文件不存在: {full_original_file_path}")
        return False

    try:
        with open(patch_file_path, "w", encoding="utf-8") as f:
            f.write(patch_str)
        logger.info(f"补丁内容已写入: {patch_file_path}")

        # 使用 patch 命令
        # patch -p1 < patch_file_path (如果补丁中的路径是 a/file.c b/file.c)
        # patch original_file_path < patch_file_path (如果补丁中的路径是 --- file.c +++ file.c)
        # 需要根据 AI 生成的补丁格式调整 -p 参数或命令
        # 假设 AI 生成的补丁路径是相对于项目根目录的，例如：
        # --- a/src/main.c
        # +++ b/src/main.c
        # 或者
        # --- src/main.c
        # +++ src/main.c
        # 我们尝试使用 -p1，如果失败，可以尝试不带 -p 或 -p0

        # 确保 patch 命令在 project_root 中执行
        # 补丁中的文件路径应该是相对于 project_root 的
        # 例如，如果 original_file_path 是 "src/file.c"，补丁应该是针对 "src/file.c"
        
        # 尝试确定 patch level
        # 如果补丁以 "--- a/path/to/file" 开头，通常用 -p1
        # 如果补丁以 "--- path/to/file" 开头，通常用 -p0
        # 如果补丁以 "--- /abs/path/to/file" 开头，则需要更复杂的处理或确保AI不生成这种格式
        
        # 简单的 p-level 检测
        p_level = "1" # 默认
        first_diff_line = patch_str.splitlines()[0] if patch_str.splitlines() else ""
        if first_diff_line.startswith("--- ") and not first_diff_line.startswith("--- a/"):
             # 可能是 --- path/to/file 或者 --- /abs/path/to/file
             # 如果是 --- path/to/file (相对于 project_root)，则 p0
             # 我们假设AI生成的路径是相对于项目根目录的，如 "src/file.c"
             # 并且补丁格式是 "--- src/file.c" 或 "--- a/src/file.c"
             if not os.path.isabs(first_diff_line.split(' ')[1]): # 不是绝对路径
                p_level = "0"


        # command = ["patch", f"-p{p_level}", "-u", "-N", "--no-backup-if-mismatch", "-i", os.path.abspath(patch_file_path)]
        # `-N` 忽略已经应用的补丁
        # `-u` 表示 unified diff
        # `--no-backup-if-mismatch` 即使有不匹配也不创建 .rej 文件 (或者移除此选项以查看 .rej)
        # `-i patch_file_path` 指定输入文件
        # `original_file_path` (可选) 指定要打补丁的文件，如果补丁本身不包含足够信息
        
        # 更可靠的方式是让 patch 命令自动检测文件，如果补丁格式正确
        # command = ["patch", f"-p{p_level}", "-uN", "--no-backup-if-mismatch", f"< \"{os.path.abspath(patch_file_path)}\""]
        # 使用 subprocess.run 时，输入重定向需要特殊处理，或者直接将补丁内容通过 stdin 传递

        process = subprocess.Popen(
            ["patch", f"-p{p_level}", "-uN", "--no-backup-if-mismatch", original_file_path],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            cwd=project_root, # 在项目根目录执行
            text=True
        )
        stdout, stderr = process.communicate(input=patch_str)

        if process.returncode == 0:
            logger.info(f"成功将补丁应用到文件: {original_file_path}")
            logger.debug(f"Patch STDOUT:\n{stdout}")
            if stderr:
                 logger.warning(f"Patch STDERR (但返回码为0):\n{stderr}")
            return True
        else:
            logger.error(f"应用补丁到文件 {original_file_path} 失败。Return code: {process.returncode}")
            logger.error(f"Patch STDOUT:\n{stdout}")
            logger.error(f"Patch STDERR:\n{stderr}")
            # 尝试恢复原始文件 (如果需要且有备份机制)
            return False
    except FileNotFoundError:
        logger.error("`patch` 命令未找到。请确保它已安装并อยู่ใน PATH 中。")
        return False
    except Exception as e:
        logger.error(f"应用补丁时发生错误: {e}")
        return False
    finally:
        if os.path.exists(patch_file_path):
            os.remove(patch_file_path)