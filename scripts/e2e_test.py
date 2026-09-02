"""端到端冒烟测试：真实调用 DeepSeek，验证 Agent 多步工具调用。

用法：在项目根目录执行
    python scripts/e2e_test.py
需要 .env 中已配置 DEEPSEEK_API_KEY。
"""
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
# 保证能 import 到项目根目录的 agent 包
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from agent.graph import ask

CASES = [
    "帮我读一下 data/示例-待办.txt",
    "北京和上海今天天气怎么样？",
    "整理 data/示例-销售数据.csv，并告诉我数据里有什么问题",
]

if __name__ == "__main__":
    for q in CASES:
        print("=" * 60)
        print("问题：", q)
        try:
            answer, trace = ask(q)
        except Exception as exc:
            print("出错：", type(exc).__name__, exc)
            continue
        print("--- 工具调用轨迹 ---")
        for step in trace:
            print(step)
        print("--- 回答 ---")
        print(answer)