"""Agent 可调用工具集：read_file / get_weather / tidy_spreadsheet。

设计要点：
1. 每个工具 = 一份 OpenAI Function Calling 格式的 schema + 一个普通 Python 函数；
2. 大模型“看” schema 决定何时调用、传什么参数（这就是 Function Calling 的核心）；
3. 真实执行仍由本地 Python 完成，保证可控、可解释。
"""
from __future__ import annotations

import json
import urllib.parse
import urllib.request
from pathlib import Path

import pandas as pd

# 项目根目录：工具默认只允许读写项目目录内的文件，避免误碰系统/应用数据
BASE_DIR = Path(__file__).resolve().parent.parent


# ---------------------------------------------------------------- 工具实现

def _resolve(path_or_name: str) -> Path:
    """相对路径基于项目根目录解析，绝对路径按原样（仅限项目内）。"""
    p = Path(path_or_name).expanduser()
    if not p.is_absolute():
        p = BASE_DIR / p
    return p.resolve()


def read_file(raw_path: str, max_chars: int = 4000) -> str:
    """读取文本文件内容（示例路径：data/示例-待办.txt）。"""
    p = _resolve(raw_path)
    try:
        if not p.exists():
            return f"文件不存在：{p}。请先确认文件路径是否正确。"
        if p.is_dir():
            return f"{p} 是文件夹，不是文件。"
        text = p.read_text(encoding="utf-8", errors="replace")
    except Exception as exc:  # noqa: BLE001
        return f"读取文件失败：{exc}"
    if len(text) > max_chars:
        text = text[:max_chars] + f"\n……（内容过长，已截断，共 {len(text)} 字符）"
    return text


WEATHER_CODE_ZH = {
    0: "晴", 1: "基本晴朗", 2: "局部多云", 3: "阴",
    45: "雾", 48: "雾凇",
    51: "毛毛雨", 53: "小雨", 55: "中雨", 56: "冻毛毛雨", 57: "冻雨",
    61: "小雨", 63: "中雨", 65: "大雨", 66: "冻雨", 67: "强冻雨",
    71: "小雪", 73: "中雪", 75: "大雪", 77: "霰",
    80: "小阵雨", 81: "中阵雨", 82: "强阵雨",
    85: "小阵雪", 86: "大阵雪",
    95: "雷暴", 96: "雷暴伴冰雹", 99: "强雷暴伴冰雹",
}


def get_weather(city: str) -> str:
    """查询某城市实时天气（Open-Meteo 免费接口，无需额外 API Key）。"""
    try:
        # 第 1 步：城市名 -> 经纬度（地理编码）
        geo_url = ("https://geocoding-api.open-meteo.com/v1/search"
                   "?count=1&language=zh&format=json&name=" + urllib.parse.quote(city))
        with urllib.request.urlopen(geo_url, timeout=10) as resp:
            geo = json.loads(resp.read().decode("utf-8"))
        results = geo.get("results") or []
        if not results:
            return f"未找到城市：{city}，请换一个更常见的城市名再试。"
        lat, lon = results[0]["latitude"], results[0]["longitude"]
        name = results[0].get("name", city)

        # 第 2 步：按经纬度查天气
        w_url = (
            f"https://api.open-meteo.com/v1/forecast"
            f"?latitude={lat}&longitude={lon}"
            f"&current=temperature_2m,relative_humidity_2m,weather_code,wind_speed_10m"
            f"&timezone=auto"
        )
        with urllib.request.urlopen(w_url, timeout=10) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        cur = data.get("current", {})
        code = cur.get("weather_code")
        desc = WEATHER_CODE_ZH.get(code, f"代码 {code}")
        parts = [
            f"{name} 当前天气：{desc}",
            f"气温 {cur.get('temperature_2m')}°C",
            f"相对湿度 {cur.get('relative_humidity_2m')}%",
            f"风速 {cur.get('wind_speed_10m')} km/h",
        ]
        if cur.get("time"):
            parts.append(f"数据时间 {cur['time']}")
        return "，".join(parts) + "。"
    except Exception as exc:  # noqa: BLE001
        return f"查询天气失败：{exc}（请检查网络，或稍后重试）"


def tidy_spreadsheet(raw_path: str, output_name: str | None = None) -> str:
    """整理表格：读取 CSV/XLSX -> 清理（空行/重复）-> 保存到 data/output/ -> 返回摘要。"""
    p = _resolve(raw_path)
    if not p.exists():
        return f"文件不存在：{p}。"
    try:
        if p.suffix.lower() == ".csv":
            df = pd.read_csv(p)
        elif p.suffix.lower() in (".xlsx", ".xls"):
            df = pd.read_excel(p)
        else:
            return f"不支持的表格格式：{p.suffix}（目前支持 .csv / .xlsx）"
    except Exception as exc:  # noqa: BLE001
        return f"读取表格失败：{exc}"

    before = df.shape
    dropped_note = []
    df = df.dropna(how="all")          # 去掉整行都是空值的行
    if df.shape[0] != before[0]:
        dropped_note.append("空行")
    df = df.drop_duplicates()          # 去掉完全重复的行
    if df.shape[0] != before[0] - (1 if dropped_note else 0):
        dropped_note.append("重复行")

    out_dir = BASE_DIR / "data" / "output"
    out_dir.mkdir(parents=True, exist_ok=True)
    default_name = f"{p.stem}_cleaned.xlsx"
    out_file = out_dir / (output_name or default_name)
    try:
        df.to_excel(out_file, index=False)
    except Exception as exc:  # noqa: BLE001
        return f"保存整理结果失败：{exc}"

    lines = [
        f"✅ 已整理并保存：{out_file}",
        f"数据规模：{before[0]} 行 × {before[1]} 列 -> {df.shape[0]} 行 × {df.shape[1]} 列",
    ]
    if dropped_note:
        lines.append("清理动作：" + "、".join(dropped_note))
    lines.append("列名：" + "、".join(str(c) for c in df.columns))
    num = df.select_dtypes(include="number")
    if not num.empty:
        lines.append("\n数值列摘要：\n" + num.describe().to_string())
    return "\n".join(lines)


def search_knowledge_base(query: str, top_k: int = 4) -> str:
    """(tool) 检索本地知识库，返回带出处的相关片段。"""
    from rag.search import search_knowledge_base as _kb_search
    try:
        top_k = max(1, int(top_k))
    except (TypeError, ValueError):
        top_k = 4
    return _kb_search(query, top_k)


def analyze_image(raw_path: str, question: str | None = None) -> str:
    """(tool) 视觉桥：让免费视觉模型描述一张图片的内容。"""
    import base64

    from langchain_core.messages import HumanMessage

    from .model_providers import create_vision_model
    from rag.config import IMAGE_EXT

    p = _resolve(raw_path)
    if not p.exists():
        return f"图片不存在：{p}。请确认路径（上传的图片在 data/uploads/ 下）。"
    if p.suffix.lower() not in IMAGE_EXT:
        return f"不支持的图片格式：{p.suffix}（支持 {'/'.join(sorted(IMAGE_EXT))}）"
    if p.stat().st_size > 10 * 1024 * 1024:
        return "图片过大（超过 10MB），请压缩后重试。"
    mime = {
        ".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg",
        ".gif": "image/gif", ".webp": "image/webp",
    }[p.suffix.lower()]
    b64 = base64.b64encode(p.read_bytes()).decode("ascii")
    data_url = f"data:{mime};base64,{b64}"
    try:
        llm = create_vision_model()
        text = question or "请详细描述这张图片的内容，包括图中的文字、场景与关键信息。"
        msg = HumanMessage(content=[
            {"type": "text", "text": text},
            {"type": "image_url", "image_url": {"url": data_url}},
        ])
        resp = llm.invoke([msg])
        return str(resp.content or "（模型未返回内容）")
    except Exception as exc:  # noqa: BLE001
        return f"识图失败：{exc}"


def web_search(query: str, max_results: int = 5) -> str:
    """(tool) 搜索公开网页信息（Bing 免费 / Tavily 可选）。"""
    from .websearch import web_search as _ws
    return _ws(query, max_results)


def open_webpage(url: str, max_chars: int = 3000) -> str:
    """(tool) 打开网页读取正文文本。"""
    from .websearch import open_webpage as _ow
    return _ow(url, max_chars)


# ---------------------------------------------------------------- 注册表（Function Calling schema）

TOOLS: list[dict] = [
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "读取项目 data 目录内（或项目内）的文本文件内容。用于帮用户查看文件。",
            "parameters": {
                "type": "object",
                "properties": {
                    "raw_path": {
                        "type": "string",
                        "description": "文件路径，如 data/示例-待办.txt",
                    },
                    "max_chars": {
                        "type": "integer",
                        "description": "最多返回的字符数，默认 4000",
                    },
                },
                "required": ["raw_path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "查询某个城市的当前实时天气。用于回答“某地天气怎么样”之类的问题。",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {"type": "string", "description": "城市名，如 北京、上海、广州"},
                },
                "required": ["city"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "tidy_spreadsheet",
            "description": "整理表格文件（CSV/XLSX）：自动清理空行与重复行，把结果保存为新的 Excel 文件到 data/output/，并返回数据摘要。",
            "parameters": {
                "type": "object",
                "properties": {
                    "raw_path": {"type": "string", "description": "表格文件路径，如 data/示例-销售数据.csv"},
                    "output_name": {"type": "string", "description": "可选：输出文件名，如 销售数据_整理.xlsx"},
                },
                "required": ["raw_path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_knowledge_base",
            "description": "检索本地知识库（用户上传/导入的资料、笔记、文档），返回带出处的相关片段。当用户询问资料、知识库、笔记、文档中的内容时必须先用它。",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "要检索的问题或关键词"},
                    "top_k": {"type": "integer", "description": "返回片段数，默认 4"},
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "analyze_image",
            "description": "识别并描述一张图片的内容（图中文字、场景等）。当用户上传图片或询问图片内容时调用。",
            "parameters": {
                "type": "object",
                "properties": {
                    "raw_path": {"type": "string", "description": "图片路径，如 data/uploads/示例.png"},
                    "question": {"type": "string", "description": "可选：针对图片的具体问题"},
                },
                "required": ["raw_path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": "搜索公开互联网信息，返回标题/链接/摘要。当用户询问新闻、最新信息、知识库没有的内容或需要外部资料时使用。",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "搜索关键词，尽量具体，如：2026 年 AI Agent 最新进展"},
                    "max_results": {"type": "integer", "description": "返回条数，默认 5"},
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "open_webpage",
            "description": "打开一个网页并读取正文文本。当搜索结果摘要不够、需要细读某篇文章时使用。",
            "parameters": {
                "type": "object",
                "properties": {
                    "url": {"type": "string", "description": "网页完整链接（http/https）"},
                    "max_chars": {"type": "integer", "description": "最多读取字符数，默认 3000"},
                },
                "required": ["url"],
            },
        },
    },
]

# 工具名 -> 实现函数 的映射
TOOL_IMPL: dict[str, object] = {
    "read_file": read_file,
    "get_weather": get_weather,
    "tidy_spreadsheet": tidy_spreadsheet,
    "search_knowledge_base": search_knowledge_base,
    "analyze_image": analyze_image,
    "web_search": web_search,
    "open_webpage": open_webpage,
}


def run_tool(name: str, arguments: dict | str) -> str:
    """按名字调用工具实现，参数可能是 dict 或 JSON 字符串。"""
    if name not in TOOL_IMPL:
        return f"未知工具：{name}"
    if isinstance(arguments, str):
        arguments = json.loads(arguments or "{}")
    if not isinstance(arguments, dict):
        return f"工具参数格式错误：{arguments}"
    try:
        return str(TOOL_IMPL[name](**arguments))
    except Exception as exc:  # noqa: BLE001
        return f"工具执行出错：{exc}"