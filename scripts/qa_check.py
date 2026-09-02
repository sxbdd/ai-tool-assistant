"""严格 QA：功能输出 与 真实数据 逐项核对（可重复运行）。

用法：python scripts/qa_check.py
离线项不依赖网络；在线项需要 .env 已配置 Key / 可联网。
任一项失败会打印 FAIL 并以非 0 退出。
"""
import sys, time, uuid
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

PASS, FAIL = 0, 0


def check(name, fn):
    global PASS, FAIL
    try:
        ok, msg = fn()
        if ok:
            PASS += 1
            print(f"  PASS  {name} — {msg}")
        else:
            FAIL += 1
            print(f"  FAIL  {name} — {msg}")
    except Exception as exc:  # noqa: BLE001
        FAIL += 1
        print(f"  FAIL  {name} — 异常 {type(exc).__name__}: {exc}")


def eq(got, exp, label=""):
    return (got == exp, f"{label} 应为 {exp!r}，实际 {got!r}" if got != exp else f"一致：{exp!r}")


# ---------------------------------------------------------------- 离线：工具与数据
def t_read_file():
    p = ROOT / "data" / "示例-待办.txt"
    text = p.read_text(encoding="utf-8")
    from agent.tools import read_file
    got = read_file("data/示例-待办.txt")
    return eq(got, text, "read_file 全文")


def t_read_truncate():
    from agent.tools import read_file
    got = read_file("data/示例-待办.txt", max_chars=10)
    return (len(got) <= 10 + 40 and "已截断" in got, f"截断长度 {len(got)}")


def t_tidy_data():
    import pandas as pd
    from agent.tools import tidy_spreadsheet
    res = tidy_spreadsheet("data/示例-销售数据.csv")
    out = ROOT / "data" / "output" / "示例-销售数据_cleaned.xlsx"
    if not out.exists():
        return (False, "清洗产物未生成")
    df = pd.read_excel(out)
    if df.shape[0] != 5:
        return (False, f"清洗后应为 5 行，实际 {df.shape[0]}")
    if df.duplicated().any():
        return (False, "仍存在重复行")
    if df.isna().all(axis=1).any():
        return (False, "仍存在整行空值")
    return (True, f"7→5 行正确；{out.name}")


def t_tool_registry():
    from agent.tools import TOOLS, run_tool
    names = [t["function"]["name"] for t in TOOLS]
    expect = {"read_file", "get_weather", "tidy_spreadsheet", "search_knowledge_base",
              "analyze_image", "web_search", "open_webpage"}
    if set(names) != expect:
        return (False, f"工具集合不符：{sorted(names)}")
    out = run_tool("read_file", {"raw_path": "data/示例-待办.txt"})
    return ("今日待办" in out, "run_tool 分发正常")


# ---------------------------------------------------------------- 离线：RAG 数据一致性
def t_rag_roundtrip():
    from rag.ingest import ingest_text_pages
    from rag.search import search_knowledge_base
    from rag.store import list_docs, remove_doc
    token = f"QA_TOKEN_{uuid.uuid4().hex[:8]}"
    doc = f"qa_{token}"
    text = f"本段为 QA 数据一致性测试：{token} 代表检索增强生成流程正确。"
    st = ingest_text_pages(doc, [(text, None)])
    if st.get("chunks", 0) < 1:
        return (False, "入库片段为 0")
    names = [d["doc_name"] for d in list_docs()]
    if doc not in names:
        return (False, "入库后未出现在文档列表")
    res = search_knowledge_base(token)
    if doc not in res or token not in res:
        return (False, "检索未带回对应文档/内容")
    removed = remove_doc(doc)
    names2 = [d["doc_name"] for d in list_docs()]
    if removed < 1 or doc in names2:
        return (False, "删除未生效或残留")
    return (True, f"入库→检索→删除闭环 OK（{st['chunks']} 片段）")


def t_rag_overwrite():
    from rag.ingest import ingest_text_pages
    from rag.store import list_docs, remove_doc
    doc = f"qa_over_{uuid.uuid4().hex[:6]}"
    ingest_text_pages(doc, [("v1 " * 400, None)])
    c1 = next((d["chunks"] for d in list_docs() if d["doc_name"] == doc), 0)
    ingest_text_pages(doc, [("v2 短内容", None)])
    c2 = next((d["chunks"] for d in list_docs() if d["doc_name"] == doc), 0)
    remove_doc(doc)
    return (c1 > 1 and c2 == 1, f"覆盖式入库 v1={c1} 块 → v2={c2} 块")


# ---------------------------------------------------------------- 在线项
def t_weather():
    from agent.tools import get_weather
    res = get_weather("北京")
    return ("北京" in res and "°C" in res, res[:80])


def t_vision():
    from agent.tools import analyze_image
    img = ROOT / "data" / "uploads" / "测试天气图.png"
    if not img.exists():
        from PIL import Image, ImageDraw, ImageFont
        img.parent.mkdir(parents=True, exist_ok=True)
        im = Image.new("RGB", (640, 360), "white")
        d = ImageDraw.Draw(im)
        try:
            font = ImageFont.truetype(r"C:\Windows\Fonts\msyh.ttc", 42)
        except Exception:  # noqa: BLE001
            font = ImageFont.load_default()
        d.text((40, 60), "今日天气：晴", fill="black", font=font)
        im.save(img)
    res = analyze_image(str(img))
    keys = ["今日天气", "晴", "25", "太阳", "适合"]
    hit = [k for k in keys if k in res]
    return (bool(hit), f"识别命中关键词 {hit}（输出前60字：{res[:60]}）" if hit else f"未命中，输出：{res[:120]}")


def t_web_search():
    from agent.tools import web_search
    res = web_search("LangGraph 教程", max_results=4)
    return ("LangGraph" in res and "http" in res, "Bing 返回含链接与关键词" if "LangGraph" in res else res[:120])


def t_llm_fact():
    from agent.graph import ask
    ans, tr = ask("用一句话回答：1+1等于几？")
    return ("2" in ans, f"回答含 2（trace={tr}，ans={ans[:80]}）")


def main():
    print("== 严格 QA 开始 ==")
    print("[离线]")
    check("read_file 内容一致", t_read_file)
    check("read_file 截断", t_read_truncate)
    check("tidy 数据正确(7→5)", t_tidy_data)
    check("工具注册表=7/分发", t_tool_registry)
    check("RAG 入库→检索→删除", t_rag_roundtrip)
    check("RAG 覆盖式入库", t_rag_overwrite)
    print("[在线]")
    check("天气-北京(数据含温度)", t_weather)
    check("识图-读出图中文字", t_vision)
    check("联网搜索-Bing", t_web_search)
    check("LLM 事实正确(1+1=2)", t_llm_fact)
    print(f"\n== 结果：PASS {PASS} / FAIL {FAIL} ==")
    sys.exit(0 if FAIL == 0 else 1)


if __name__ == "__main__":
    main()