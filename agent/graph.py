"""LangGraph Agent 编排：模型 <-> 工具 的多步循环。

流程图（LangGraph 状态机）：
    START -> [agent 节点：大模型决定调用哪个工具] --有工具调用--> [tools 节点：本地执行工具]
                          |                                          |
                          +-- 没有工具调用 -> END                    +--> 回到 agent（直到出最终答案）
"""
from __future__ import annotations

import json
from operator import add
from typing import Annotated, TypedDict

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages

from .model_providers import create_chat_model
from .config import check_api_key
from .tools import TOOLS, run_tool

SYSTEM_PROMPT = """你是「AI 工具调用小助手」，一个会调用工具完成任务的智能体（Agent）。

你可以使用以下工具：
1. read_file：读取项目内的文本文件（如 data/示例-待办.txt）；
2. get_weather：查询任意城市的实时天气；
3. tidy_spreadsheet：整理 CSV / Excel 表格，清理空行和重复行，并保存为新文件；
4. search_knowledge_base：检索本地知识库（用户上传/导入的资料、笔记、文档），返回带来源的片段；
5. analyze_image：查看用户上传的图片内容；
6. web_search：搜索公开互联网信息；
7. open_webpage：打开网页读取正文。

工作要求：
- 先理解用户意图，再判断需要调用哪些工具、按什么顺序调用（多步任务要一步一步完成）；
- 用户询问资料、笔记、文档、知识库里的内容时，必须先调用 search_knowledge_base 检索；
- 回答只能基于检索片段，片段不足时明确说「知识库中没有找到相关内容」，不要编造；
- 引用资料时用（来源：文件名 · 页码）标注；
- 用户上传图片想让你看时，调用 analyze_image 获取图片描述后再回答；
- 用户询问新闻、最新动态、实时信息、或知识库里没有的内容时，调用 web_search；如需细读某篇文章再用 open_webpage；
- 联网结果要如实转述并附链接来源，不要凭空补充搜索结果之外的事实；
- 工具返回结果后，基于结果给用户完整、清晰的中文答复；
- 一次可并行调用多个互不依赖的工具（例如同时查两个城市的天气）；
- 不要编造工具结果，工具查不到就如实告诉用户。
"""


class AgentState(TypedDict):
    """Agent 运行状态：消息历史 + 工具调用轨迹。"""

    messages: Annotated[list, add_messages]   # 对话/工具消息，由 LangGraph 自动追加
    trace: Annotated[list[str], add]          # 工具调用轨迹（供网页端展示）


def create_agent(provider: str | None = None, model: str | None = None):
    """构建并编译 LangGraph Agent。"""
    check_api_key()

    # 模型走抽象层 model_providers：可在 .env / UI 切换服务商与模型
    llm = create_chat_model(provider=provider, model=model)
    llm_with_tools = llm.bind_tools(TOOLS)

    def call_model(state: AgentState) -> dict:
        """agent 节点：把系统提示 + 历史消息交给大模型，由它决定是否调用工具。"""
        messages = [SystemMessage(content=SYSTEM_PROMPT), *state["messages"]]
        response = llm_with_tools.invoke(messages)
        return {"messages": [response]}

    def call_tools(state: AgentState) -> dict:
        """tools 节点：执行模型请求的所有工具，把结果作为 ToolMessage 返回给模型。"""
        last: AIMessage = state["messages"][-1]
        tool_messages: list[ToolMessage] = []
        trace_steps: list[str] = []
        for call in last.tool_calls:
            name = call["name"]
            args = call.get("args", {})
            trace_steps.append(
                f"🔧 {name}({json.dumps(args, ensure_ascii=False)})"
            )
            result = run_tool(name, args)
            tool_messages.append(
                ToolMessage(content=result, tool_call_id=call["id"])
            )
        return {"messages": tool_messages, "trace": trace_steps}

    def route_after_agent(state: AgentState) -> str:
        """条件路由：模型还想调用工具就去 tools 节点，否则结束。"""
        last = state["messages"][-1]
        if isinstance(last, AIMessage) and last.tool_calls:
            return "tools"
        return END

    builder = StateGraph(AgentState)
    builder.add_node("agent", call_model)
    builder.add_node("tools", call_tools)
    builder.add_edge(START, "agent")
    builder.add_conditional_edges("agent", route_after_agent, {"tools": "tools", END: END})
    builder.add_edge("tools", "agent")
    return builder.compile()


def ask(question: str, history: list | None = None, provider: str | None = None, model: str | None = None) -> tuple[str, list[str]]:
    """对外接口：向 Agent 提问，返回 (最终回答, 工具调用轨迹)。"""
    graph = create_agent(provider=provider, model=model)
    init_messages: list = list(history or [])
    init_messages.append(HumanMessage(content=question))
    result = graph.invoke(
        {"messages": init_messages, "trace": []},
        config={"recursion_limit": 40},   # 防止极端情况下无限循环
    )
    answer = result["messages"][-1].content or "（模型未返回内容）"
    return str(answer), list(result.get("trace", []))