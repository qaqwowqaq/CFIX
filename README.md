# CFix - AI辅助C/C++代码修复工具

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://python.org)
[![Docker](https://img.shields.io/badge/Docker-Required-blue.svg)](https://docker.com)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

CFix 是一个基于大语言模型的智能C/C++代码修复工具，能够自动定位问题文件并生成修复补丁。该工具结合了传统静态分析方法和先进的AI技术，为开源项目提供高质量的自动化代码修复服务。

## 🎯 主要特性

### 🤖 AI增强的文件定位
- **智能文件分析**：结合问题描述、测试补丁和项目结构进行多维度分析
- **传统方法融合**：将AI分析结果与基于关键词匹配的传统方法相结合
- **高准确率定位**：通过多阶段筛选确保定位到最相关的源文件

### 🔧 智能补丁生成
- **上下文感知**：收集相关头文件和依赖文件作为修复上下文
- **增强提示词**：基于问题分析、测试用例和文件内容构建高质量的修复提示
- **多重验证**：通过Docker容器进行隔离测试验证

### 📊 完整的实验追踪
- **对话历史记录**：详细记录每次AI交互的完整过程
- **多次尝试支持**：支持失败重试，累积学习和改进
- **丰富的日志**：包含文件定位、补丁生成、测试执行的全过程日志

### 🐳 Docker化测试环境
- **环境隔离**：每个修复任务在独立的Docker容器中执行
- **多种测试方法**：支持HEREDOC、拷贝等多种补丁应用方式
- **真实验证**：通过项目原生测试套件验证修复效果

## 🏗️ 系统架构

```
CFix/
├── dataset_runner.py      # 主要运行器，处理数据集和实验流程
├── file_locator.py        # 文件定位模块（传统+AI方法）
├── ai_handler.py          # AI API调用和响应处理
├── config.py             # 配置管理（API密钥等）
├── docker_manager.py     # Docker容器管理
├── dataset/              # 测试数据集
│   ├── test_c.jsonl     # C/C++项目问题数据集
│   └── ...
├── project_data/         # 实验数据和日志
│   ├── logs/            # 详细的执行日志
│   ├── repos/           # 克隆的代码仓库
│   └── results/         # 实验结果汇总
└── utils/               # 工具函数
    ├── git_utils.py     # Git操作
    ├── file_utils.py    # 文件处理
    └── docker_utils.py  # Docker工具
```

## 🚀 快速开始

### 环境要求

- **Python 3.8+**
- **Docker**（用于测试环境隔离）
- **Git**（用于代码仓库操作）
- **网络连接**（用于AI API调用）

### 安装步骤

1. **克隆项目**
```bash
git clone https://github.com/your-repo/CFix.git
cd CFix
```

2. **安装依赖**
```bash
pip install -r requirements.txt
```

3. **配置API密钥**
```bash
# 创建配置文件
cp config.py.example config.py

# 编辑配置文件，添加你的DeepSeek API密钥
vim config.py
```

4. **准备Docker环境**
```bash
# 确保Docker服务运行
sudo systemctl start docker

# 测试Docker访问权限
docker run hello-world
```

### 基本使用

#### 单个问题修复
```bash
# 修复特定实例
python dataset_runner.py --instance-id "redis__hiredis-427" --max-retries 3

# 修复并保存详细日志
python dataset_runner.py --instance-id "dunst-project__dunst-1215" --debug
```

#### 批量处理
```bash
# 处理数据集中的前10个问题
python dataset_runner.py --max-instances 10 --max-retries 2

# 并行处理（小心资源使用）
python dataset_runner.py --parallel 3 --max-instances 50
```

#### 自定义数据集
```bash
# 使用自定义数据集
python dataset_runner.py --dataset ./custom_dataset.jsonl --output-dir ./custom_results
```

## 📖 详细使用指南

### 数据集格式

CFix使用JSONL格式的数据集，每行包含一个问题实例：

```json
{
  "repo": "redis/hiredis",
  "pull_number": 427,
  "instance_id": "redis__hiredis-427",
  "issue_numbers": ["426"],
  "base_commit": "e93c05a7aa64736c91e455322c83181e8e67cd0e",
  "patch": "diff --git a/hiredis.c b/hiredis.c\n...",
  "test_patch": "diff --git a/test.c b/test.c\n...",
  "problem_statement": "Typo format in redisFormatSdsCommandArgv function...",
  "hints_text": "",
  "created_at": "2016-05-14T09:26:51Z"
}
```

### 配置选项

在 config.py 中可以配置：

```python
# API配置
DEEPSEEK_API_KEY = "your-api-key-here"
API_BASE_URL = "https://api.deepseek.com"

# Docker配置
DOCKER_TIMEOUT = 300  # 秒
MAX_CONTAINER_MEMORY = "2g"

# 文件定位配置
MAX_CONTEXT_FILES = 5
MAX_FILE_SIZE = 50000  # 字节

# 重试配置
MAX_API_RETRIES = 3
RETRY_DELAY = 5  # 秒
```

### 输出结果

每次运行会生成详细的结果文件：

```
project_data/logs/redis__hiredis-427_20250605_222642/attempt_1/
├── ai_conversation_history.json     # 完整的AI对话历史
├── localization_prompt.txt          # 文件定位的提示词
├── localization_response.txt        # 文件定位的AI响应
├── patch_generation_prompt.txt      # 补丁生成的提示词
├── patch_generation_response.txt    # 补丁生成的AI响应
├── generated_patch.diff             # AI生成的补丁
├── docker_test_heredoc.log          # Docker测试日志
└── test_results.json               # 测试结果汇总
```

## 🔬 核心技术

### 1. 智能文件定位算法

CFix采用多阶段文件定位策略：

#### 传统方法
- **关键词匹配**：从问题描述中提取关键词，在文件名和路径中匹配
- **测试文件分析**：分析测试补丁中涉及的文件，推断相关源文件
- **依赖关系追踪**：通过include关系找到相关文件
- **优先级打分**：结合多个因素为每个候选文件计算相关性分数

#### AI增强方法
```python
def ai_locate_target_file(issue_details, project_path, test_patch_diff, api_key):
    # 1. 扫描项目文件结构
    project_structure = scan_project_files_for_ai(project_path)
    
    # 2. 构建智能提示词
    prompt = build_file_localization_prompt(issue_details, test_patch_diff, project_structure)
    
    # 3. AI分析和推理
    ai_analysis = ai_handler.simple_api_call(api_key, prompt)
    
    # 4. 结果解析和验证
    return parse_and_validate_ai_response(ai_analysis)
```

#### 融合策略
```python
def merge_ai_and_traditional_results(ai_analysis, traditional_result, project_path):
    if ai_analysis and ai_analysis.get('confidence', 0) >= 7:
        return ai_analysis['target_file']  # 高置信度AI结果
    elif traditional_result:
        return traditional_result          # 传统方法备选
    else:
        return ai_analysis.get('target_file') if ai_analysis else None
```

### 2. 上下文感知的补丁生成

#### 上下文收集
```python
def collect_context_files(repo_path, target_file):
    context_files = {}
    
    # 1. 直接包含的头文件
    included_files = extract_includes_from_file(target_file)
    
    # 2. 被包含的相关文件
    depending_files = find_files_depending_on(target_file)
    
    # 3. 同目录相关文件
    related_files = find_related_files_in_directory(target_file)
    
    return context_files
```

#### 增强提示词构建
```python
def build_enhanced_repair_prompt(instance, target_file, file_content, context_files, test_patch_diff, ai_analysis):
    prompt = f"""
    ## 问题分析
    {ai_analysis.get('reasoning', '')}
    
    ## 文件内容
    ```c
    {file_content}
```

    ## 相关上下文文件
    {format_context_files(context_files)}
    
    ## 测试用例变更
    ```diff
    {test_patch_diff}
    ```
    
    ## 修复要求
    请生成精确的diff格式补丁...
    """
    return prompt
```

### 3. Docker化测试验证

#### 容器创建和配置
```python
def create_test_container(repo_url, base_commit, patch_content):
    dockerfile_content = f"""
    FROM ubuntu:20.04
    RUN apt-get update && apt-get install -y git build-essential
    WORKDIR /home
    RUN git clone {repo_url} .
    RUN git checkout {base_commit}
    """
    
    container = docker_client.containers.run(
        "temp_image", 
        command="sleep infinity",
        detach=True,
        mem_limit="2g",
        cpu_period=100000,
        cpu_quota=50000
    )
    return container
```

#### 测试执行流程
```python
def execute_patch_and_test(container, patch_content):
    # 1. 复制补丁到容器
    copy_patch_to_container(container, patch_content)
    
    # 2. 运行测试脚本（fix_run.sh会自动应用测试补丁）
    test_result = container.exec_run(["bash", "fix_run.sh"])
    
    # 3. 分析测试结果
    success = analyze_test_output(test_result.output)
    
    return {
        'success': success,
        'exit_code': test_result.exit_code,
        'output': test_result.output.decode('utf-8')
    }
```

## 📊 实验结果分析

### 成功案例示例

**redis/hiredis-427**：
- **问题**：空指针访问风险在 `strlen(argv[j])` 调用中
- **AI修复**：`len = argvlen ? argvlen[j] : (argv[j] ? strlen(argv[j]) : 0);`
- **结果**：所有79个测试用例通过
- **特点**：AI正确识别了防御性编程需求

### 挑战和局限性

1. **文件定位准确性**：在复杂项目中仍可能选择错误的文件
2. **补丁格式问题**：AI生成的补丁可能存在行号偏移
3. **上下文理解**：对于需要深度理解业务逻辑的问题仍有困难
4. **API依赖性**：依赖外部AI服务的稳定性和质量

## 🔧 故障排除

### 常见问题

#### 1. API调用失败
```bash
# 检查API密钥配置
python -c "import config; print('API Key:', config.DEEPSEEK_API_KEY[:10] + '...')"

# 测试网络连接
curl -H "Authorization: Bearer YOUR_API_KEY" https://api.deepseek.com/v1/models
```

#### 2. Docker权限问题
```bash
# 添加用户到docker组
sudo usermod -aG docker $USER
newgrp docker

# 测试Docker访问
docker ps
```

#### 3. 文件定位失败
```bash
# 启用详细日志
python dataset_runner.py --instance-id "example" --debug

# 检查项目克隆状态
ls -la project_data/repos/
```

#### 4. 内存不足
```bash
# 调整Docker内存限制
# 在config.py中修改 MAX_CONTAINER_MEMORY = "4g"

# 监控系统资源
htop
docker stats
```

## 🤝 贡献指南

### 开发环境设置

1. **创建开发分支**
```bash
git checkout -b feature/your-feature-name
```

2. **安装开发依赖**
```bash
pip install -r requirements-dev.txt
```

3. **运行测试**
```bash
python -m pytest tests/
```

### 代码规范

- **PEP 8**：遵循Python代码规范
- **类型注解**：为函数参数和返回值添加类型注解
- **文档字符串**：为所有公共函数添加详细的docstring
- **日志记录**：适当使用logging模块记录关键信息

### 提交规范

```bash
# 提交格式
git commit -m "feat: 添加新的文件定位算法"
git commit -m "fix: 修复Docker容器内存泄漏问题"
git commit -m "docs: 更新API使用文档"
```

## 📚 API参考

### 主要类和函数

#### DatasetRunner
```python
class DatasetRunner:
    def process_instance(self, instance: dict, max_retries: int = 1) -> dict:
        """处理单个问题实例"""
        
    def generate_ai_patch(self, instance: dict, repo_path: str, 
                          test_patch_diff: str, conversation_log_dir: str) -> tuple:
        """使用AI生成修复补丁"""
```

#### FileLocator
```python
def find_file_to_fix_with_ai(issue_details: dict, project_path: str, 
                           test_patch_diff: str, ai_api_key: str, 
                           enable_ai: bool = True) -> tuple:
    """AI增强的文件定位"""

def collect_context_files(repo_path: str, target_file: str) -> dict:
    """收集相关上下文文件"""
```

#### AIHandler
```python
def simple_api_call(api_key: str, prompt: str) -> dict:
    """简单的AI API调用"""
    
def generate_patch_with_enhanced_prompt(api_key: str, prompt: str, 
                                      stream: bool = False) -> dict:
    """使用增强提示词生成补丁"""
```

## 📈 性能优化

### 缓存策略
- **仓库缓存**：避免重复克隆相同的代码仓库
- **文件分析缓存**：缓存文件依赖关系分析结果
- **API响应缓存**：缓存相似问题的AI响应（可选）

### 并行处理
```python
# 并行处理多个实例（谨慎使用）
python dataset_runner.py --parallel 3 --max-instances 20
```

### 资源管理
- **容器清理**：及时清理测试完成的Docker容器
- **磁盘空间**：定期清理旧的日志和临时文件
- **内存控制**：限制单个容器的内存使用

## 🔮 未来改进方向

### 短期目标
1. **改进文件定位算法**：提高复杂项目中的定位准确率
2. **增强错误处理**：更好的API失败恢复机制
3. **优化性能**：减少不必要的重复计算
4. **扩展数据集**：支持更多类型的C/C++问题

### 中期目标
1. **多模型支持**：集成GPT-4、Claude等多种AI模型
2. **学习机制**：从成功和失败案例中学习改进
3. **可视化界面**：提供Web界面进行交互式修复
4. **增量修复**：支持对大型补丁的分步骤修复

### 长期愿景
1. **通用化**：扩展到Java、Python等其他语言
2. **IDE集成**：与主流IDE集成提供实时修复建议
3. **代码理解**：更深入的语义理解和推理能力
4. **社区生态**：建立开源社区和插件生态系统

## 📄 许可证

本项目采用 MIT 许可证。详见 LICENSE 文件。

## 🙏 致谢

- **DeepSeek** 提供高质量的AI API服务
- **Docker** 提供容器化技术支持
- **开源社区** 提供丰富的测试数据集和问题案例
- **所有贡献者** 为项目改进提供宝贵建议

## 📞 联系方式

- **项目主页**：https://github.com/qaqwowqaq/CFix
- **问题报告**：https://github.com/qaqwowqaq/CFix/issues
- **讨论组**：[加入讨论](https://your-discussion-link.com)

---

**CFix - 让AI帮助我们写出更好的代码！** 🚀