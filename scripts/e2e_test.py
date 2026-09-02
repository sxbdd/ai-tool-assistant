"""端到端冒烟测试 v2.1：真实调用 DeepSeek，覆盖 工具调用 + RAG（可选识图）。

用法（项目根目录）：
    python scripts/e2e_test.py
需要 .env 已配置 DEEPSEEK_API_KEY；RAG 用例会自动把 data/示例-课程笔记.md 入库。
"""
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from agent.graph import ask  # noqa: E402
from rag.ingest import ingest_file  # noqa: E402

NOTE = PROJECT_ROOT / "data" / "示例-课程笔记.md"
IMG = PROJECT_ROOT / "data" / "uploads" / "测试天气图.png"


def main() -> None:
    print(">>> 预置：示例课程笔记入库")
    print("   ", ingest_file(NOTE))

    cases = [
        "帮我读一下 data/示例-待办.txt",
        "根据我的笔记，什么是 RAG？",
        "北京和上海今天天气怎么样？",
        "整理 data/示例-销售数据.csv，并告诉我数据里有什么问题",
    ]
    if IMG.exists():
        cases.append(
            f"请调用 analyze_image 工具分析 {IMG} 这张图片，告诉我内容"
        )

    for q in cases:
        print("=" * 70)
        print("问题：", q)
        try:
            answer, trace = ask(q)
        except Exception as exc:  # noqa: BLE001
            print("出错：", type(exc).__name__, exc)
            continue
        print("--- 工具调用轨迹 ---")
        for step in trace:
            print(step)
        print("--- 回答（前 300 字） ---")
        print(answer[:300])


if __name__ == "__main__":
    main()