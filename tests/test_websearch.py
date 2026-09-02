"""联网搜索解析离线测试（不依赖网络）。"""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agent.websearch import _strip_tags, parse_bing_html  # noqa: E402

HTML = (
    '<html><body>'
    '<li class="b_algo"><h2><a href="https://example.com/a">标题A</a></h2><p>简介A</p></li>'
    '<li class="b_algo"><h2><a href="https://example.com/b">标题B</a></h2></li>'
    '</body></html>'
)


class TestBingParse(unittest.TestCase):
    def test_parse_two(self):
        items = parse_bing_html(HTML, max_results=5)
        self.assertEqual(len(items), 2)
        self.assertEqual(items[0]["title"], "标题A")
        self.assertIn("https://example.com/a", items[0]["url"])
        self.assertEqual(items[0]["snippet"], "简介A")

    def test_limit(self):
        self.assertEqual(len(parse_bing_html(HTML, max_results=1)), 1)

    def test_strip_tags(self):
        self.assertEqual(_strip_tags("<p>a <b>b</b></p>"), "a b")


if __name__ == "__main__":
    unittest.main(verbosity=2)