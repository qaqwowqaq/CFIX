import os
import sys
import unittest
from unittest.mock import patch, MagicMock
from dotenv import load_dotenv
import json
import logging
# 加载环境变量
load_dotenv()

# 导入 ai_handler 模块
import ai_handler

class TestAIHandler(unittest.TestCase):

    def setUp(self):
        self.api_key = os.getenv("DEEPSEEK_API_KEY")
        if not self.api_key:
            self.skipTest("未设置 DEEPSEEK_API_KEY 环境变量，跳过测试")

        self.file_content = """#include <iostream>

int main() {
    int array[5] = {1, 2, 3, 4, 5};
    // 数组越界访问
    for (int i = 0; i <= 5; i++) {
        std::cout << array[i] << std::endl;
    }
    return 0;
}"""

        self.issue_description = """
Title: Array Index Out of Bounds

Body:
The program has an array index out of bounds bug in main.cpp. 
The array is defined with size 5, but the loop iterates from 0 to 5 (inclusive),
which means the last iteration accesses array[5] which is beyond the array bounds.
"""

        self.file_path = "main.cpp"

    def test_extract_diff_from_response(self):
        """测试 diff 提取逻辑：Markdown 包裹和裸 diff"""
        response = """```diff
--- a/main.cpp
+++ b/main.cpp
@@ -3,7 +3,7 @@
 int main() {
     int array[5] = {1, 2, 3, 4, 5};
     // 数组越界访问
-    for (int i = 0; i <= 5; i++) {
+    for (int i = 0; i < 5; i++) {
         std::cout << array[i] << std::endl;
     }
     return 0;
}
```"""
        expected_start = "--- a/main.cpp"
        extracted = ai_handler.extract_diff_from_response(response)
        self.assertTrue(extracted.startswith(expected_start))
        self.assertIn("+    for (int i = 0; i < 5; i++) {", extracted)

    @patch('requests.post')
    def test_generate_patch_with_deepseek_mock(self, mock_post):
        """测试：使用模拟响应验证 API 调用逻辑"""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "choices": [{
                "message": {
                    "content": """```diff
--- a/main.cpp
+++ b/main.cpp
@@ -3,7 +3,7 @@
 int main() {
     int array[5] = {1, 2, 3, 4, 5};
     // 数组越界访问
-    for (int i = 0; i <= 5; i++) {
+    for (int i = 0; i < 5; i++) {
         std::cout << array[i] << std::endl;
     }
     return 0;
}
```"""
                }
            }]
        }
        mock_post.return_value = mock_response

        result = ai_handler.generate_patch_with_deepseek(
            self.api_key, self.file_content, self.issue_description, self.file_path
        )
        self.assertIsNotNone(result)
        self.assertIn("for (int i = 0; i < 5; i++)", result)
        self.assertIn("-    for (int i = 0; i <= 5; i++)", result)

    def test_generate_patch_with_deepseek_real_api(self):
        """真实 API 调用测试（需要 API KEY）"""
        print("\n--- 测试非流式 API 调用 ---")
        result = ai_handler.generate_patch_with_deepseek(
            self.api_key, self.file_content, self.issue_description, self.file_path
        )
        print("\n=== 实际生成的补丁 ===")
        print(result)
        print("======================\n")

        self.assertIsNotNone(result)
        self.assertTrue(
            ("-    for (int i = 0; i <= 5; i++) {" in result and "+    for (int i = 0; i < 5; i++) {" in result)
            or ("for (int i = 0; i <= 5;" in result and "for (int i = 0; i < 5;" in result)
        )

    def test_generate_patch_with_deepseek_stream(self):
        """测试流式 API 调用"""
        print("\n--- 测试流式 API 调用 ---")
        
        accumulated_chunks = []
        
        def stream_callback(chunk):
            accumulated_chunks.append(chunk)
            print(f"收到流式块 ({len(chunk)}字符): {chunk}", end='', flush=True)
        
        # 使用流式模式调用
        generator = ai_handler.generate_patch_with_deepseek(
            self.api_key, 
            self.file_content, 
            self.issue_description, 
            self.file_path,
            stream=True,
            stream_callback=stream_callback
        )
        
        # 消耗生成器
        result = None
        chunks_received = 0
        
        try:
            for chunk in generator:
                chunks_received += 1
                result = chunk  # 获取最终返回值
        except StopIteration as e:
            if e.value is not None:
                result = e.value
        
        print(f"\n\n共收到 {chunks_received} 个流式块")
        print("\n=== 流式模式最终结果 ===")
        print(result)
        print("=========================\n")
        
        self.assertIsNotNone(result)
        self.assertTrue(
            ("-    for (int i = 0; i <= 5; i++) {" in result and "+    for (int i = 0; i < 5; i++) {" in result)
            or ("for (int i = 0; i <= 5;" in result and "for (int i = 0; i < 5;" in result)
            or any("-    for (int i = 0; i <= 5; i++) {" in chunk and "+    for (int i = 0; i < 5; i++) {" in chunk 
                for chunk in accumulated_chunks)
        )

if __name__ == '__main__':
    # 设置日志级别为详细模式，方便查看 API 调用过程
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    unittest.main()