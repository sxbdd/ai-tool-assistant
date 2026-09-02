"""网页对话界面（Streamlit）。

启动方式（在项目根目录）：
    streamlit run app.py
"""
from __future__ import annotations

import streamlit as st
from langchain_core.messages import AIMessage as LCAIMessage
from langchain_core.messages import HumanMessage

from agent.graph import ask

st.set_page_config(page_title="AI 工具调用小助手", page_icon="🤖", layout="centered")

st.title("🤖 AI 工具调用小助手")
st.caption("Agent + Function Calling：让 DeepSeek 自动调用工具，完成「读文件 / 查天气 / 整理表格」等多步任务")

with st.sidebar:
    st.header("🧰 可用工具")
    st.markdown(
        "- 📄 `read_file`：读取项目内文本文件\n"
        "- 🌤️ `get_weather`：查城市实时天气（Open-Meteo，免费）\n"
        "- 📊 `tidy_spreadsheet`：整理 CSV/XLSX 并输出报告"
    )
    st.header("💡 试试这样说")
    st.markdown(
        "- 帮我读一下 `data/示例-待办.txt`\n"
        "- 北京和上海今天天气怎么样？\n"
        "- 整理 `data/示例-销售数据.csv`，看看数据有什么问题"
    )

if "history" not in st.session_state:
    st.session_state.history = []   # [{"role", "content", "trace"}]


def render_history() -> None:
    for item in st.session_state.history:
        with st.chat_message(item["role"]):
            st.markdown(item["content"])
            if item.get("trace"):
                with st.expander("🧰 工具调用过程", expanded=False):
                    for step in item["trace"]:
                        st.markdown(step)


render_history()

if prompt := st.chat_input("输入你的任务，例如：整理 data/示例-销售数据.csv"):
    with st.chat_message("user"):
        st.markdown(prompt)
    st.session_state.history.append({"role": "user", "content": prompt, "trace": []})

    # 把历史转换为 LangChain 消息格式，作为上下文传给 Agent
    lc_history: list = []
    for item in st.session_state.history[:-1]:
        if item["role"] == "user":
            lc_history.append(HumanMessage(content=item["content"]))
        else:
            lc_history.append(LCAIMessage(content=item["content"]))

    with st.chat_message("assistant"):
        try:
            with st.spinner("Agent 正在思考并调用工具……"):
                answer, trace = ask(prompt, lc_history)
        except RuntimeError as exc:
            st.error(str(exc))
            answer, trace = None, []
        if answer:
            st.markdown(answer)
            if trace:
                with st.expander("🧰 工具调用过程", expanded=True):
                    for step in trace:
                        st.markdown(step)
            st.session_state.history.append(
                {"role": "assistant", "content": answer, "trace": trace}
            )