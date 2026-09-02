"""graph/工具 注册与上下文裁剪 离线测试。"""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agent.config import MAX_CONTEXT_MESSAGES  # noqa: E402
from agent.graph import _trim_history  # noqa: E402
from agent.tools import TOOLS  # noqa: E402


class TestGraphHelpers(unittest.TestCase):
    def test_tools_registry_count(self):
        self.assertEqual(len(TOOLS), 7)

    def test_trim_history(self):
        h = [f"msg{i}" for i in range(20)]
        out = _trim_history(h)
        self.assertLessEqual(len(out), MAX_CONTEXT_MESSAGES)
        self.assertEqual(out[-1], "msg19")   # 保留最新的

    def test_trim_history_small(self):
        h = ["a", "b"]
        self.assertEqual(_trim_history(h), ["a", "b"])


if __name__ == "__main__":
    unittest.main(verbosity=2)