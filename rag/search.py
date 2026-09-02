"""检索工具实现：search_knowledge_base（供 Agent 调用，返回带来源的片段）。"""
from __future__ import annotations

from .config import DEFAULT_TOP_K, MAX_DISTANCE
from .store import count, query


def search_knowledge_base(query_text: str, top_k: int = DEFAULT_TOP_K) -> str:
    """检索本地知识库，返回带出处的相关片段（供大模型引用回答）。"""
    try:
        total = count()
    except Exception as exc:  # noqa: BLE001
        return f"知识库不可用：{exc}"
    if total == 0:
        return "知识库目前为空。请先在左侧「知识库管理」上传文档或导入文件夹，我才能检索。"
    try:
        hits = query(query_text, top_k)
        hits = [h for h in hits if h[2] <= MAX_DISTANCE]   # 过滤不相关片段
    except Exception as exc:  # noqa: BLE001
        return f"检索失败：{exc}"
    if not hits:
        return f"未在知识库（共 {total} 个片段）中找到与「{query_text}」相关的内容，请换一种问法或先补充资料。"
    lines = [f"知识库中共 {total} 个片段，以下是与问题最相关的 {len(hits)} 个片段：", ""]
    for i, (doc, meta, _dist) in enumerate(hits, start=1):
        page = meta.get("page")
        src = meta.get("doc_name", "未知来源")
        if page:
            src = f"{src} · 第{page}页"
        lines.append(f"【片段{i}】（来源：{src}）")
        lines.append(doc)
        lines.append("")
    lines.append("请只依据以上片段回答，不足时明确说明，并用【来源】标注引用。")
    return "\n".join(lines)