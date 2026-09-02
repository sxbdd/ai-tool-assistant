"""离线单元测试：不依赖网络/API Key，验证文件读取与表格整理两个工具。"""
import sys
import unittest
from pathlib import Path

# 保证可以 import 到项目内的 agent 包
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agent.tools import read_file, tidy_spreadsheet  # noqa: E402

BASE = Path(__file__).resolve().parent.parent


class TestReadFile(unittest.TestCase):
    def test_read_sample(self):
        text = read_file("data/示例-待办.txt")
        self.assertIn("今日待办", text)

    def test_not_found(self):
        self.assertIn("不存在", read_file("data/不存在的文件.txt"))


class TestTidySpreadsheet(unittest.TestCase):
    def test_clean_csv(self):
        result = tidy_spreadsheet("data/示例-销售数据.csv")
        self.assertIn("已整理并保存", result)
        out = BASE / "data" / "output" / "示例-销售数据_cleaned.xlsx"
        self.assertTrue(out.exists())
        # 7 行原始 -> 清理后应为 5 行（去掉 1 空行 + 1 重复行）
        self.assertIn("7 行", result)
        self.assertIn("5 行", result)


if __name__ == "__main__":
    unittest.main(verbosity=2)