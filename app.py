"""网页对话界面（Streamlit）v2.3 — 对齐 Codex 桌面版
多会话管理（本地持久化）+ 折叠面板 + 附件式上传 + 文件库 + 状态栏。
"""
from __future__ import annotations

import json
import os
import uuid
from datetime import datetime
from pathlib import Path

import streamlit as st
from langchain_core.messages import AIMessage as LCAIMessage
from langchain_core.messages import HumanMessage

from agent.graph import ask, ask_stream
from agent.model_providers import PROVIDERS, list_providers, provider_status
from rag.config import IMAGE_EXT, SUPPORTED_EXT, UPLOAD_DIR, ensure_dirs
from rag.ingest import ingest_file, ingest_folder
from rag.store import list_docs, remove_doc

BASE_DIR = Path(__file__).resolve().parent
SESSION_DIR = BASE_DIR / "data" / "sessions"
ensure_dirs()
SESSION_DIR.mkdir(parents=True, exist_ok=True)

# ---- 可选第三方组件（GitHub 免费开源）----
try:  # 中文文件上传组件（GracefulTabby, MIT）
    from streamlit_chunk_file_uploader import uploader as chunk_uploader
    HAS_CHUNK = True
except Exception:  # noqa: BLE001
    HAS_CHUNK = False

try:  # 图片点击缩放组件（vgilabert94, MIT）
    from streamlit_image_zoom import image_zoom
    HAS_ZOOM = True
except Exception:  # noqa: BLE001
    HAS_ZOOM = False

st.set_page_config(page_title="AI 工具调用小助手", page_icon="🤖", layout="wide")

# ============================================================ 中文化样式（尽力而为）
st.markdown('''
<style>
/* Streamlit 原生上传框中文化：选择器未命中时保持原样，不影响使用 */
[data-testid="stFileUploaderDropzone"] button { font-size: 0 !important; }
[data-testid="stFileUploaderDropzone"] button::after { content: "选择文件"; font-size: 1rem !important; font-weight: 600; }
[data-testid="stFileUploaderDropzoneInstructions"] { visibility: hidden; }
[data-testid="stFileUploaderDropzoneInstructions"]::after { content: "点击选择或将文件拖到此处"; visibility: visible; }
[data-testid="stWidgetLabel"] p { color: #334155; }
</style>
''', unsafe_allow_html=True)


# ============================================================ 会话存储
def _now() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def list_sessions() -> list[dict]:
    out = []
    for f in SESSION_DIR.glob("*.json"):
        try:
            data = json.loads(f.read_text(encoding="utf-8"))
            out.append({
                "id": data.get("id", f.stem),
                "title": data.get("title", "未命名"),
                "updated": data.get("updated", ""),
            })
        except Exception:  # noqa: BLE001
            continue
    out.sort(key=lambda s: s["updated"], reverse=True)
    return out


def read_session(sid: str) -> dict:
    p = SESSION_DIR / f"{sid}.json"
    if p.exists():
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            pass
    return {"id": sid, "title": "新会话", "created": _now(), "updated": _now(), "history": []}


def write_session(data: dict) -> None:
    data["updated"] = _now()
    (SESSION_DIR / f"{data['id']}.json").write_text(
        json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8"
    )


def ensure_session() -> None:
    if "cur_sid" not in st.session_state:
        sids = [s["id"] for s in list_sessions()]
        st.session_state["cur_sid"] = sids[0] if sids else new_session()
    _load_current_history()


def new_session() -> str:
    sid = uuid.uuid4().hex[:12]
    n = len(list_sessions()) + 1
    write_session({"id": sid, "title": f"会话 {n}", "created": _now(), "updated": _now(), "history": []})
    st.session_state["cur_sid"] = sid
    st.session_state["history"] = []
    return sid


def _load_current_history() -> None:
    data = read_session(st.session_state["cur_sid"])
    st.session_state["history"] = data.get("history", [])
    st.session_state["session_title"] = data.get("title", "未命名")


def _save_current() -> None:
    data = read_session(st.session_state["cur_sid"])
    data["history"] = st.session_state.get("history", [])
    data["title"] = st.session_state.get("session_title", data.get("title", "未命名"))
    write_session(data)


def delete_session(sid: str) -> None:
    p = SESSION_DIR / f"{sid}.json"
    if p.exists():
        p.unlink()
    if st.session_state.get("cur_sid") == sid:
        sids = [s["id"] for s in list_sessions()]
        st.session_state["cur_sid"] = sids[0] if sids else new_session()
        _load_current_history()


# ============================================================ 小工具


def export_history_md() -> str:
    """把当前会话历史导出为 Markdown。"""
    lines = [
        f"# {st.session_state.get('session_title', '会话')}",
        "",
        f"> 导出时间：{_now()}",
        "",
    ]
    for it in st.session_state.get("history", []):
        who = "😀 我" if it.get("role") == "user" else "🤖 AI"
        lines.append(f"### {who}\n\n{it.get('content', '')}\n")
        if it.get("trace"):
            lines.append("**工具调用：**\n")
            for s in it["trace"]:
                lines.append(f"- {s}\n")
    return "\n".join(lines)

def fmt_size(n: int) -> str:
    if n < 1024:
        return f"{n} B"
    if n < 1024 * 1024:
        return f"{n / 1024:.1f} KB"
    return f"{n / 1024 / 1024:.1f} MB"


EXT_ICON = {
    ".png": "🖼️", ".jpg": "🖼️", ".jpeg": "🖼️", ".gif": "🖼️", ".webp": "🖼️",
    ".pdf": "📕", ".txt": "📄", ".md": "📝", ".csv": "📊", ".xlsx": "📊",
    ".xls": "📊", ".docx": "📘", ".py": "🐍",
}


def icon_of(p: Path) -> str:
    return EXT_ICON.get(p.suffix.lower(), "📎")


def is_image(p: Path) -> bool:
    return p.suffix.lower() in IMAGE_EXT


def list_upload_files() -> list[Path]:
    if not UPLOAD_DIR.exists():
        return []
    files = [p for p in UPLOAD_DIR.iterdir() if p.is_file()]
    files.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return files


def save_upload(name: str, data: bytes) -> Path:
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    p = UPLOAD_DIR / name
    i = 1
    while p.exists():
        p = UPLOAD_DIR / f"{Path(name).stem}({i}){Path(name).suffix}"
        i += 1
    p.write_bytes(data)
    return p


def ensure_defaults() -> None:
    st.session_state.setdefault("attachments", [])
    st.session_state.setdefault("main_provider", "deepseek")
    st.session_state.setdefault("main_model", "deepseek-chat")
    st.session_state.setdefault("preview_file", None)
    st.session_state.setdefault("_seen_files", set())


def _clipboard_html(text: str) -> str:
    """返回一个'复制'HTML 按钮（点击把内容写入剪贴板，本地 localhost 可用）。"""
    import json as _json
    payload = _json.dumps(text, ensure_ascii=False)
    onclick = f"navigator.clipboard.writeText({payload});this.textContent='✅ 已复制'"
    return (
        '<button style="border:1px solid #cbd5e1;border-radius:8px;background:#f8fafc;'
        'color:#475569;font-size:12px;padding:1px 10px;cursor:pointer" '
        f'onclick="{onclick}">📋 复制</button>'
    )


def _message_actions(i: int, item: dict) -> None:
    """在消息正文之后渲染操作按钮（复制/重写/删除/编辑），逻辑集中在一处。"""
    if item["role"] == "user":
        c1, c2, _ = st.columns([1, 1, 5])
        with c1:
            if st.button("✏️ 编辑", key=f"edit_msg_{i}", use_container_width=True):
                st.session_state["_edit"] = {"idx": i, "text": item["content"]}
                st.rerun()
        with c2:
            st.markdown(_clipboard_html(item["content"]), unsafe_allow_html=True)
    else:
        c1, c2, c3, _ = st.columns([1, 1, 1, 4])
        with c1:
            st.markdown(_clipboard_html(item["content"]), unsafe_allow_html=True)
        with c2:
            if st.button("🔄 重写", key=f"regen_msg_{i}", use_container_width=True):
                st.session_state["_action"] = {"type": "regen", "idx": i}
                st.rerun()
        with c3:
            if st.button("🗑", key=f"del_msg_{i}", use_container_width=True, help="删除本条回答"):
                st.session_state["_action"] = {"type": "del", "idx": i}
                st.rerun()


def _render_files(files: list) -> None:
    """在消息里展示文件引用：图片小缩略图，其余仅文件名（不展开内容）。"""
    for f in files:
        if f.get("kind") == "img" and Path(f.get("path", "")).exists():
            st.image(f["path"], width=90)
        else:
            st.caption(f"📎 {f.get('name', '')}")


def _save_attachment(name: str, data: bytes) -> None:
    """静默保存上传文件并把引用加入当前消息（不弹提示、不自动处理）。"""
    fp = save_upload(name, data)
    st.session_state.setdefault("attachments", []).append(
        {"name": fp.name, "path": str(fp), "kind": "img" if fp.suffix.lower() in IMAGE_EXT else "doc"}
    )


def _refs_text(files: list) -> str:
    """生成发给模型的精简引用说明（用户界面不展示这堆文字）。"""
    lines = []
    for f in files:
        tip = "图片→用 analyze_image 查看" if f.get("kind") == "img" else "文档→用 read_file/相关工具读取"
        lines.append(f"- {f.get('name', '')}（{tip}）：{f.get('path', '')}")
    return "\n".join(lines)


def render_messages() -> None:
    """渲染历史消息；操作按钮统一走 _message_actions。"""
    hist = st.session_state.get("history", [])
    for i, item in enumerate(hist):
        with st.chat_message(item["role"]):
            if item.get("img_path") and Path(item["img_path"]).exists():
                st.image(str(item["img_path"]), width=120)
            st.markdown(item.get("display") or item["content"])
            if item["role"] == "user" and item.get("files"):
                _render_files(item["files"])
            if item.get("trace"):
                with st.expander("🧰 工具调用过程"):
                    for s in item["trace"]:
                        st.markdown(s)
            _message_actions(i, item)


def run_ask(prompt: str, img_path: str | None = None, attachments: list | None = None) -> None:
    with st.chat_message("user"):
        if img_path and Path(img_path).exists():
            st.image(img_path, width=120)
        if attachments:
            _render_files(attachments)
        st.markdown(prompt)
    model_prompt = prompt
    if attachments:
        model_prompt = prompt + "\n\n（本消息附带了文件引用，请按需查看/处理）\n" + _refs_text(attachments)
    item = {"role": "user", "content": model_prompt, "display": prompt, "trace": []}
    if img_path:
        item["img_path"] = img_path
    if attachments:
        item["files"] = [{"name": a["name"], "path": a["path"], "kind": a["kind"]} for a in attachments]
    st.session_state["history"].append(item)

    lc_history: list = []
    for it in st.session_state["history"][:-1]:
        if it["role"] == "user":
            lc_history.append(HumanMessage(content=it["content"]))
        else:
            lc_history.append(LCAIMessage(content=it["content"]))

    st.session_state["_retry"] = None
    with st.chat_message("assistant"):
        ans_box = st.empty()
        tool_box = st.empty()
        trace_lines: list[str] = []
        buf = ""
        failed = None
        try:
            for ev in ask_stream(
                model_prompt, lc_history,
                provider=st.session_state["main_provider"],
                model=st.session_state["main_model"],
            ):
                if ev["type"] == "token":
                    buf += ev["text"]
                    ans_box.markdown(buf + "▍")
                elif ev["type"] == "tool":
                    trace_lines.append(ev["text"])
                    tool_box.caption("🧰 " + " ｜ ".join(trace_lines[-3:]))
                elif ev["type"] == "error":
                    failed = ev["text"]
                elif ev["type"] == "answer":
                    buf = ev["text"]
                    ans_box.markdown(buf)
                elif ev["type"] == "trace":
                    trace_lines = ev["lines"]
        except Exception as exc:  # noqa: BLE001
            failed = str(exc)
        if failed:
            st.error(f"出错了：{failed}")
            st.session_state["_retry"] = (prompt, img_path, attachments)
        else:
            if buf:
                ans_box.markdown(buf)
            if trace_lines:
                with st.expander("🧰 工具调用过程", expanded=len(trace_lines) > 1):
                    for s in trace_lines:
                        st.markdown(s)
            st.session_state["history"].append(
                {"role": "assistant", "content": buf, "trace": trace_lines}
            )
    _save_current()


# ============================================================ 侧栏
ensure_defaults()
ensure_session()

with st.sidebar:
    st.header("🤖 AI 工具调用小助手")

    with st.expander("💬 会话管理", expanded=False):
        sessions = list_sessions()
        cur_id = st.session_state["cur_sid"]
        _q = st.text_input("🔍 搜索会话", key="sess_search", label_visibility="collapsed", placeholder="🔍 搜索会话标题…")
        if _q.strip():
            sessions = [s for s in sessions if _q.strip().lower() in s["title"].lower()]
            if cur_id not in {s["id"] for s in sessions}:
                cur = read_session(cur_id)
                sessions.insert(0, {"id": cur["id"], "title": cur["title"] + "（当前）", "updated": cur.get("updated", "")})
        if sessions:
            options = {s["id"]: s["title"] for s in sessions}
            if cur_id not in options:
                cur_id = sessions[0]["id"]
                st.session_state["cur_sid"] = cur_id
                _load_current_history()
            sel = st.selectbox(
                "当前会话", list(options), index=list(options).index(cur_id),
                format_func=lambda i: options[i],
            )
            if sel != cur_id:
                st.session_state["cur_sid"] = sel
                _load_current_history()
                st.rerun()
        b1, b2, b3 = st.columns(3)
        if b1.button("🆕 新建", key="btn_new_session", use_container_width=True):
            new_session()
            st.rerun()
        if b2.button("✏️ 改名", key="btn_rename_session", use_container_width=True):
            st.session_state["renaming"] = True
        if b3.button("🗑 删除", key="btn_del_session", use_container_width=True):
            delete_session(cur_id)
            st.rerun()
        if st.session_state.get("renaming"):
            new_title = st.text_input("新标题", value=st.session_state.get("session_title", ""))
            if st.button("确定改名", key="btn_confirm_rename", use_container_width=True):
                st.session_state["session_title"] = new_title.strip() or "未命名"
                st.session_state["renaming"] = False
                _save_current()
                st.rerun()
        st.divider()
        cc1, cc2 = st.columns(2)
        if cc1.button("🧹 清空对话", key="btn_clear_chat", use_container_width=True):
            st.session_state["history"] = []
            _save_current()
            st.rerun()
        cc2.download_button(
            "⬇ 导出对话", data=export_history_md(),
            file_name=f"{st.session_state.get('session_title', '会话')}.md",
            mime="text/markdown", key="btn_export_chat", use_container_width=True,
        )

    with st.expander("⚙️ 模型设置", expanded=True):
        status = provider_status()
        main_provider = st.selectbox(
            "主力：服务商", list_providers(), index=0,
            format_func=lambda n: f"{n} · {status[n]}",
            help="未配置 Key 会报错，请在 .env 补充",
        )
        main_models = PROVIDERS[main_provider]["models"]
        _env_main = os.getenv(PROVIDERS[main_provider]["model_env"], "")
        main_model = st.selectbox(
            "主力：具体模型", main_models,
            index=main_models.index(_env_main) if _env_main in main_models else 0,
        )
        st.session_state["main_provider"] = main_provider
        st.session_state["main_model"] = main_model
        os.environ["DEFAULT_PROVIDER"] = main_provider

        vis_providers = [n for n in list_providers() if PROVIDERS[n]["supports_vision"]]
        if vis_providers:
            vision_provider = st.selectbox(
                "识图：服务商", vis_providers,
                index=vis_providers.index("zhipu") if "zhipu" in vis_providers else 0,
                format_func=lambda n: f"{n} · {status[n]}",
            )
            vis_models = PROVIDERS[vision_provider]["models"]
            _env_vis = os.getenv(PROVIDERS[vision_provider]["model_env"], "")
            vision_model = st.selectbox(
                "识图：具体模型", vis_models,
                index=vis_models.index(_env_vis) if _env_vis in vis_models else 0,
            )
            os.environ["VISION_PROVIDER"] = vision_provider
            os.environ[PROVIDERS[vision_provider]["model_env"]] = vision_model

    files = list_upload_files()
    with st.expander(f"📁 文件库（{len(files)}）", expanded=False):
        st.caption("文件库 = 上传文件的存放与管理（预览 / 入库知识库 / 删除）。要让 AI 处理文件，请在对话区用「➕ 附件」上传，会自动进入当前对话。")
        if not files:
            st.caption("还没有上传文件。在对话区点「➕ 附件」上传。")
        else:
            names = [f.name for f in files]
            sel_name = st.selectbox("选择文件", names, key="fl_sel")
            sel_path = UPLOAD_DIR / sel_name
            st.caption(
                f"{icon_of(sel_path)} {sel_name} ｜ {fmt_size(sel_path.stat().st_size)}"
                f" ｜ {datetime.fromtimestamp(sel_path.stat().st_mtime):%m-%d %H:%M}"
            )
            c1, c2, c3 = st.columns(3)
            if c1.button("👁 预览", key="btn_preview_file", use_container_width=True):
                st.session_state["preview_file"] = str(sel_path) if st.session_state.get("preview_file") != str(sel_path) else None
            if c2.button("📚 入库", key="btn_ingest_file", use_container_width=True):
                r = ingest_file(sel_path)
                if "error" in r:
                    st.error(r["error"])
                else:
                    st.success(f"已入库：{r.get('chunks', 0)} 个片段")
            if c3.button("🗑 删除", key="btn_del_file", use_container_width=True):
                sel_path.unlink()
                st.session_state["preview_file"] = None
                st.rerun()

            preview = st.session_state.get("preview_file")
            if preview and Path(preview).exists():
                pp = Path(preview)
                st.divider()
                st.markdown(f"**预览：{pp.name}**")
                if is_image(pp) and HAS_ZOOM:
                    try:
                        from PIL import Image as PILImage
                        image_zoom(PILImage.open(pp), mode="scroll", size=(640, 420))
                        st.caption("滚动缩放 / 拖动查看")
                    except Exception:  # noqa: BLE001
                        st.image(str(pp), width=520)
                elif is_image(pp):
                    st.image(str(pp), width=520)
                elif pp.suffix.lower() in {".txt", ".md", ".csv"}:
                    st.code(pp.read_text(encoding="utf-8", errors="replace")[:2000], language=None)
                else:
                    st.caption("该类型不支持直接预览，可用「发送」或「入库」。")

    with st.expander("📚 知识库管理", expanded=False):
        st.caption("支持：" + " / ".join(sorted(SUPPORTED_EXT)) + "；单文件 ≤20MB；文档复制进沙盒 data/kb/")
        kb_up = st.file_uploader(
            "上传文档入库", type=[e.lstrip(".") for e in sorted(SUPPORTED_EXT)],
            accept_multiple_files=True,
        )
        if kb_up and st.button("上传并入库", key="btn_upload_kb", use_container_width=True):
            for f in kb_up:
                if f.size > 20 * 1024 * 1024:
                    st.error(f"{f.name} 超过 20MB，跳过")
                    continue
                dest = save_upload(f.name, f.getvalue())
                r = ingest_file(dest)
                st.success(f"{f.name} -> {r.get('chunks', 0)} 片段" if "error" not in r else f"{f.name}：{r['error']}")
        docs = list_docs()
        if not docs and (BASE_DIR / "data" / "示例-课程笔记.md").exists():
            st.caption("知识库为空。可先载入示例笔记体验问答：")
            if st.button("📥 一键载入示例笔记", key="btn_load_sample", use_container_width=True):
                r = ingest_file(BASE_DIR / "data" / "示例-课程笔记.md")
                st.success(f"已载入：{r.get('chunks', 0)} 个片段")
        st.markdown("**导入本地文件夹（绝对路径）**")
        folder_path = st.text_input("文件夹路径", placeholder="如 D:\\我的笔记", label_visibility="collapsed")
        if folder_path.strip() and st.button("扫描并导入", key="btn_scan_import", use_container_width=True):
            folder = Path(folder_path.strip())
            if not folder.is_dir():
                st.error(f"不是有效文件夹：{folder}")
            else:
                n = len([x for x in folder.rglob("*") if x.is_file() and x.suffix.lower() in SUPPORTED_EXT])
                st.info(f"扫描到 {n} 个支持文件，正在入库……")
                results = ingest_folder(folder)
                chunks = sum(r.get("chunks", 0) for r in results if "error" not in r)
                st.success(f"完成：入库 {chunks} 片段 / {sum(1 for r in results if 'error' not in r)} 个文件")
        if docs:
            st.divider()
            st.markdown(f"**已入库（{len(docs)}）**")
            st.dataframe(docs, use_container_width=True, hide_index=True)
            sel_del = st.multiselect("删除（仅移除索引）", [d["doc_name"] for d in docs])
            if sel_del and st.button("删除选中", key="btn_del_docs", use_container_width=True):
                for n in sel_del:
                    remove_doc(n)
                st.rerun()

# ============================================================ 主区
st.title("🤖 AI 工具调用小助手")
st.caption("查资料 · 看图 · 联网搜 · 调工具 —— 让大模型帮你完成多步任务")

# ---- 消息操作处理（删除 / 重写） ----
_action = st.session_state.pop("_action", None)
if _action:
    h = st.session_state.get("history", [])
    kind, i = _action.get("type"), _action.get("idx", -1)
    if kind == "del" and 0 <= i < len(h):
        h.pop(i)
        st.session_state["history"] = h
        _save_current()
    elif kind == "regen" and 0 < i < len(h) and h[i]["role"] == "assistant":
        # 从历史里反查产生这条回答的用户提问，避免在消息里重复存 prompt
        ui = next((j for j in range(i - 1, -1, -1) if h[j]["role"] == "user"), None)
        if ui is not None:
            prompt = h[ui]["display"] if "display" in h[ui] else h[ui]["content"]
            img = h[ui].get("img_path")
            files = h[ui].get("files")
            del h[i]
            del h[ui]
            st.session_state["history"] = h
            _save_current()
            st.session_state["_run_queue"] = st.session_state.get("_run_queue", []) + [(prompt, img, files)]
        else:
            st.warning("找不到这条回答对应的提问，无法重写。")

render_messages()

# ---- 空会话快捷示例（点击即体验） ----
if not st.session_state.get("history"):
    SUGGESTIONS = [
        ("📚 知识库问答", "根据我的笔记，什么是 RAG？"),
        ("📊 整理表格", "整理 data/示例-销售数据.csv，并告诉我数据有什么问题"),
        ("🌤 两地天气", "北京和上海今天天气怎么样？"),
        ("🌐 联网查资料", "联网搜索：LangGraph 是什么？简单介绍"),
    ]
    st.caption("👋 欢迎！点下面的示例直接体验，或自己输入任务：")
    scol = st.columns(len(SUGGESTIONS))
    for col, (label, q) in zip(scol, SUGGESTIONS):
        if col.button(label, key=f"sug_{label[:2]}", use_container_width=True):
            st.session_state["_run_queue"] = st.session_state.get("_run_queue", []) + [(q, None)]

# ---- 编辑历史消息并从此处重发 ----
_edit = st.session_state.get("_edit")
if _edit:
    with st.container(border=True):
        st.caption("✏️ 编辑消息并重发（该消息之后的内容会被替换）")
        new_text = st.text_area("消息内容", value=_edit.get("text", ""), key="edit_text_area", height=90)
        ec1, ec2 = st.columns(2)
        if ec1.button("💾 保存并重发", key="btn_edit_save", use_container_width=True):
            idx = _edit.get("idx", 0)
            h = st.session_state.get("history", [])[:idx]
            st.session_state["history"] = h
            st.session_state.pop("_edit", None)
            st.session_state["_run_queue"] = st.session_state.get("_run_queue", []) + [(new_text, None)]
            st.rerun()
        if ec2.button("取消", key="btn_edit_cancel", use_container_width=True):
            st.session_state.pop("_edit", None)
            st.rerun()

# ---- 队列：统一在主对话区执行（保证气泡不跑到侧栏） ----
_runq = st.session_state.pop("_run_queue", None)
if _runq:
    for _entry in _runq:
        _p = _entry[0]
        _ip = _entry[1] if len(_entry) > 1 else None
        _af = _entry[2] if len(_entry) > 2 else None
        run_ask(_p, _ip, _af)

# ---- 附件（引用）区：只加入待发送，不自动处理 ----
with st.popover("📎 附件", use_container_width=False):
    st.caption("添加文件作为引用：不会立刻处理，输入文字回车后一起发送。")
    if HAS_CHUNK:
        try:
            _picked = chunk_uploader(
                label="选择文件", type=None, key="attach_chunk",
                uploader_msg="点击选择文件，或拖拽到此处",
            )
            if _picked is not None and getattr(_picked, "name", None):
                _fid = getattr(_picked, "file_id", None) or f"{_picked.name}:{len(_picked.getvalue())}"
                if _fid not in st.session_state["_seen_files"]:
                    st.session_state["_seen_files"].add(_fid)
                    _save_attachment(_picked.name, _picked.getvalue())
        except Exception:  # noqa: BLE001
            pass
    with st.expander("备用上传框", expanded=False):
        fb = st.file_uploader("文件", type=None, accept_multiple_files=True, label_visibility="collapsed")
        if fb:
            for f in fb:
                _fid2 = getattr(f, "file_id", None) or f"{f.name}:{f.size}"
                if _fid2 not in st.session_state["_seen_files"]:
                    st.session_state["_seen_files"].add(_fid2)
                    _save_attachment(f.name, f.getvalue())

_attachments = st.session_state.get("attachments") or []
if _attachments:
    for _idx, _a in enumerate(_attachments):
        _ic = "🖼️" if _a["kind"] == "img" else "📎"
        _r1, _r2 = st.columns([10, 1])
        _r1.caption(f"{_ic} {_a['name']}")
        if _r2.button("✖", key=f"att_del_{_idx}", help="移除该引用"):
            _attachments.pop(_idx)
            st.session_state["attachments"] = _attachments
            st.rerun()

# ---- 失败重试 ----
if st.session_state.get("_retry"):
    rp, rimg, ratt = st.session_state["_retry"]
    if st.button("🔄 重试上一条", key="btn_retry_last", use_container_width=True):
        h = st.session_state.get("history", [])
        if h and h[-1].get("role") == "user":
            h.pop()
        st.session_state["_retry"] = None
        run_ask(rp, rimg, ratt)

# ---- 对话输入 ----
prompt = st.chat_input("输入任务，如：根据我的笔记解释 RAG｜北京天气｜联网查最新 AI 新闻")
if prompt:
    _send_atts = st.session_state.get("attachments") or []
    run_ask(prompt, attachments=_send_atts)
    st.session_state["attachments"] = []

# ---- 底部状态栏 ----
st.divider()
st.caption(
    f"📌 当前会话：{st.session_state.get('session_title', '')} ｜ 主力模型：{st.session_state['main_provider']}/{st.session_state['main_model']}"
    f" ｜ 知识库 {len(list_docs())} 个文档 ｜ 文件 {len(files)} 个"
    + (" ｜ 💾 会话自动保存到 data/sessions/" if True else "")
)