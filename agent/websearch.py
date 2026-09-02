"""联网搜索：web_search / open_webpage（Bing 免费解析 或 Tavily API）。

设计：后端可切换
- bing（默认）：解析 Bing 网页搜索结果，免费无需 Key，偶尔可能被反爬；
- tavily：更稳定、结果更干净，需在 .env 配置 TAVILY_API_KEY（tavily.com 免费额度）。
"""
from __future__ import annotations

import html as html_mod
import json
import re
import urllib.parse

import requests

from .config import TAVILY_API_KEY, WEB_SEARCH_BACKEND

UA = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
    )
}
_TIMEOUT = 12


# ---------------------------------------------------------------- Bing 解析
def _strip_tags(s: str) -> str:
    s = re.sub(r"<script.*?</script>", " ", s, flags=re.S | re.I)
    s = re.sub(r"<style.*?</style>", " ", s, flags=re.S | re.I)
    s = re.sub(r"<[^>]+>", " ", s)
    return html_mod.unescape(re.sub(r"\s+", " ", s)).strip()


def parse_bing_html(text: str, max_results: int = 5) -> list[dict]:
    """从 Bing 结果页 HTML 中抽取 [{title, url, snippet}]（含离线可测的解析函数）。"""
    items: list[dict] = []
    # 每个结果块 <li class="b_algo">…</li>
    blocks = re.split(r'<li class="b_algo"', text)[1:]
    for block in blocks:
        m = re.search(r'<h2[^>]*>\s*<a[^>]*href="([^"]+)"[^>]*>(.*?)</a>', block, re.S)
        if not m:
            continue
        url = html_mod.unescape(m.group(1))
        title = _strip_tags(m.group(2))
        snip = ""
        sm = re.search(r'<p[^>]*>(.*?)</p>', block, re.S)
        if sm:
            snip = _strip_tags(sm.group(1))
        items.append({"title": title, "url": url, "snippet": snip})
        if len(items) >= max_results:
            break
    return items


def _bing_search(query: str, max_results: int) -> list[dict]:
    params = {"q": query, "mkt": "zh-CN", "setlang": "zh-hans"}
    url = "https://www.bing.com/search?" + urllib.parse.urlencode(params)
    resp = requests.get(url, headers=UA, timeout=_TIMEOUT)
    resp.raise_for_status()
    return parse_bing_html(resp.text, max_results)


# ---------------------------------------------------------------- Tavily
def _tavily_search(query: str, max_results: int) -> list[dict]:
    resp = requests.post(
        "https://api.tavily.com/search",
        json={
            "api_key": TAVILY_API_KEY,
            "query": query,
            "max_results": max_results,
            "search_depth": "basic",
        },
        timeout=_TIMEOUT,
    )
    resp.raise_for_status()
    data = resp.json()
    out = []
    for r in (data.get("results") or [])[:max_results]:
        out.append(
            {
                "title": r.get("title", ""),
                "url": r.get("url", ""),
                "snippet": r.get("content", ""),
            }
        )
    return out


# ---------------------------------------------------------------- 对外工具
def web_search(query: str, max_results: int = 5) -> str:
    """(tool) 搜索公开网页信息，返回标题/链接/摘要。"""
    try:
        max_results = min(max(1, int(max_results)), 8)
    except (TypeError, ValueError):
        max_results = 5
    backend = WEB_SEARCH_BACKEND
    try:
        if backend == "tavily" and TAVILY_API_KEY:
            results = _tavily_search(query, max_results)
            source = "Tavily"
        else:
            results = _bing_search(query, max_results)
            source = "Bing"
    except Exception as exc:  # noqa: BLE001
        return f"联网搜索失败：{exc}（可稍后重试，或在 .env 配置 TAVILY_API_KEY 更稳定）"
    if not results:
        return f"没有搜索到与「{query}」相关的网页结果。"
    lines = [f"来自 {source} 的搜索结果（{len(results)} 条）：", ""]
    for i, r in enumerate(results, start=1):
        lines.append(f"{i}. {r['title']}")
        lines.append(f"   {r['url']}")
        if r.get("snippet"):
            lines.append(f"   {r['snippet'][:200]}")
        lines.append("")
    lines.append("回答时请综合以上结果并附上链接来源；如果信息不足可调用 open_webpage 打开链接细读。")
    return "\n".join(lines)


def open_webpage(url: str, max_chars: int = 3000) -> str:
    """(tool) 打开一个网页并读取正文文本（仅限 http/https）。"""
    if not url.lower().startswith(("http://", "https://")):
        return "仅支持 http/https 链接。"
    try:
        resp = requests.get(url, headers=UA, timeout=_TIMEOUT)
        resp.raise_for_status()
    except Exception as exc:  # noqa: BLE001
        return f"打开网页失败：{exc}"
    text = _strip_tags(resp.text)
    if len(text) > max_chars:
        text = text[:max_chars] + f"\n……（内容过长已截断，共 {len(text)} 字符）"
    return text or "（网页没有可读取的正文文本）"