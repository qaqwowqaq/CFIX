import os
from dotenv import load_dotenv

load_dotenv()

GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
MAX_RETRIES = int(os.getenv("MAX_RETRIES", 3))
LOG_FILE = os.getenv("LOG_FILE", "logs/app.log")
DOCKERFILES_BASE_PATH = os.getenv("DOCKERFILES_BASE_PATH", "./test_dockerfiles") # 存储测试用 Dockerfile 的基础路径
TEMP_WORK_DIR = os.getenv("TEMP_WORK_DIR", "temp_work_dir") # 临时工作目录

# 确保日志目录存在
LOG_DIR = os.path.dirname(LOG_FILE)
if LOG_DIR and not os.path.exists(LOG_DIR):
    os.makedirs(LOG_DIR)

# 确保临时工作目录存在
if not os.path.exists(TEMP_WORK_DIR):
    os.makedirs(TEMP_WORK_DIR)
