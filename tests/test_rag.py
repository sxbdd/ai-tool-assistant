"""RAG 离线测试：切分 + 入库 + 检索 + 带出处（不依赖网络/API Key）。"""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rag.ingest import chunk_text, ingest_text_pages  # noqa: E402
from rag.store import list_docs, remove_doc  # noqa: E402
from rag.search import search_knowledge_base  # noqa: E402

DOC_NAME = "rag-单元测试"


class TestChunk(unittest.TestCase):
    def test_no_empty(self):
        self.assertEqual(chunk_text("   \n  "), [])

    def test_short_text_single_chunk(self):
        chunks = chunk_text("RAG 是检索增强生成。")
        self.assertEqual(len(chunks), 1)

    def test_long_text_multiple_chunks(self):
        text = "句子。" * 400   # 800 字，应切成多块
        chunks = chunk_text(text)
        self.assertGreater(len(chunks), 1)
        self.assertTrue(all(len(c) <= 500 + 80 for c in chunks))


class TestRetrieval(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        text = (
            "RAG 是检索增强生成（Retrieval-Augmented Generation）。"
            "它的流程是：先把文档切分成小块，用向量模型转成向量存入向量数据库；"
            "提问时把问题也转成向量，检索最相关的片段；最后让大模型只依据这些片段生成回答，并标注来源。"
        )
        ingest_text_pages(DOC_NAME, [(text, None)])

    @classmethod
    def tearDownClass(cls):
        remove_doc(DOC_NAME)

    def test_doc_listed(self):
        names = [d["doc_name"] for d in list_docs()]
        self.assertIn(DOC_NAME, names)

    def test_query_hit(self):
        result = search_knowledge_base("RAG 的流程是什么")
        self.assertIn(DOC_NAME, result)          # 带出处
        self.assertIn("检索增强生成", result)     # 内容命中

    def test_query_out_of_scope(self):
        result = search_knowledge_base("今天股市行情怎么样")
        self.assertNotIn("【片段1】", result)


if __name__ == "__main__":
    unittest.main(verbosity=2)