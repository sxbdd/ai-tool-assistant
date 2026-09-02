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
    st.session_state.setdefault("pending", [])
    st.session_state.setdefault("main_provider", "deepseek")
    st.session_state.setdefault("main_model", "deepseek-chat")
    st.session_state.setdefault("preview_file", None)


def render_messages() -> None:
    for item in st.session_state.get("history", []):
        with st.chat_message(item["role"]):
            if item.get("img_path") and Path(item["img_path"]).exists():
                st.image(str(item["img_path"]), width=120)
            st.markdown(item["content"])
            if item.get("trace"):
                with st.expander("🧰 工具调用过程"):
                    for s in item["trace"]:
                        st.markdown(s)


def run_ask(prompt: str, img_path: str | None = None) -> None:
    with st.chat_message("user"):
        if img_path and Path(img_path).exists():
            st.image(img_path, width=120)
        st.markdown(prompt)
    item = {"role": "user", "content": prompt, "trace": []}
    if img_path:
        item["img_path"] = img_path
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
                prompt, lc_history,
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
            st.session_state["_retry"] = (prompt, img_path)
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
        if not files:
            st.caption("还没有上传文件。在下方输入框点「➕ 附件」上传。")
        else:
            names = [f.name for f in files]
            sel_name = st.selectbox("选择文件", names, key="fl_sel")
            sel_path = UPLOAD_DIR / sel_name
            st.caption(
                f"{icon_of(sel_path)} {sel_name} ｜ {fmt_size(sel_path.stat().st_size)}"
                f" ｜ {datetime.fromtimestamp(sel_path.stat().st_mtime):%m-%d %H:%M}"
            )
            c1, c2, c3, c4 = st.columns(4)
            if c1.button("👁 预览", key="btn_preview_file", use_container_width=True):
                st.session_state["preview_file"] = str(sel_path) if st.session_state.get("preview_file") != str(sel_path) else None
            if c2.button("📤 发送", key="btn_send_file", use_container_width=True):
                if is_image(sel_path):
                    run_ask(
                        f"请调用 analyze_image 工具分析 data/uploads/{sel_name} 这张图片，告诉我内容。",
                        img_path=str(sel_path),
                    )
                else:
                    run_ask(f"请用合适工具读取并处理 data/uploads/{sel_name}，告诉我结果。")
            if c3.button("📚 入库", key="btn_ingest_file", use_container_width=True):
                r = ingest_file(sel_path)
                if "error" in r:
                    st.error(r["error"])
                else:
                    st.success(f"已入库：{r.get('chunks', 0)} 个片段")
            if c4.button("🗑 删除", key="btn_del_file", use_container_width=True):
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

if not st.session_state.get("history"):
    st.info(
        "👋 欢迎！试试：\n\n"
        "1. **知识库问答**：左侧「知识库管理」一键载入示例笔记，然后问「根据我的笔记，什么是 RAG？」\n"
        "2. **识图**：点下方「➕ 附件」上传一张图片，AI 会自动识别\n"
        "3. **联网**：直接问「联网查一下今天 AI Agent 的新进展」\n"
        "4. **调工具**：「北京和上海今天天气怎么样？」"
    )

render_messages()

# ---- 附件区 ----
with st.popover("➕ 附件", use_container_width=False):
    st.caption("可上传图片 / 文档；文件存入 data/uploads/（侧栏「文件库」管理）。")
    picked = None
    if HAS_CHUNK:
        try:
            picked = chunk_uploader(
                label="选择文件", type=None, key="attach_chunk",
                uploader_msg="点击选择文件，或拖拽到此处",
            )
        except Exception:  # noqa: BLE001
            st.warning("中文组件异常，请用下方备用上传框")
    if picked is not None and getattr(picked, "name", None):
        st.session_state["pending"].append((picked.name, picked.getvalue()))
        st.success(f"已加入待发送：{picked.name}")
    with st.expander("备用上传框（原生）", expanded=False):
        fb = st.file_uploader("备用", type=None, accept_multiple_files=True, label_visibility="collapsed")
        if fb:
            for f in fb:
                st.session_state["pending"].append((f.name, f.getvalue()))
            st.success(f"已加入 {len(fb)} 个文件")

pending = st.session_state.get("pending", [])
if pending:
    st.markdown("**待发送附件：**")
    for i, (nm, _data) in enumerate(pending):
        cc1, cc2 = st.columns([8, 1])
        cc1.caption(f"📎 {nm}")
        if cc2.button("✖", key=f"rm_{i}"):
            st.session_state["pending"].pop(i)
            st.rerun()
    if st.button(f"📤 发送 {len(pending)} 个附件给 AI", key="btn_send_pending", use_container_width=True):
        lines, first_img_path = [], None
        for nm, data in pending:
            p = save_upload(nm, data)
            if p.suffix.lower() in IMAGE_EXT and first_img_path is None:
                first_img_path = str(p)
            lines.append(
                f"- {nm}（{'图片 → 请调用 analyze_image 分析' if p.suffix.lower() in IMAGE_EXT else '文档 → 请用 read_file / 相关工具处理'} data/uploads/{p.name}）"
            )
        prompt = "我上传了以下文件（已保存到 data/uploads/）：\n" + "\n".join(lines) + "\n\n请逐一查看并处理：图片请识图告诉我内容，文档请读取并总结/按需整理。"
        run_ask(prompt, img_path=first_img_path)
        st.session_state["pending"] = []

# ---- 失败重试 ----
if st.session_state.get("_retry"):
    rp, rimg = st.session_state["_retry"]
    if st.button("🔄 重试上一条", key="btn_retry_last", use_container_width=True):
        h = st.session_state.get("history", [])
        if h and h[-1].get("role") == "user":
            h.pop()
        st.session_state["_retry"] = None
        run_ask(rp, rimg)

# ---- 对话输入 ----
prompt = st.chat_input("输入任务，如：根据我的笔记解释 RAG｜北京天气｜联网查最新 AI 新闻")
if prompt:
    run_ask(prompt)

# ---- 底部状态栏 ----
st.divider()
st.caption(
    f"📌 当前会话：{st.session_state.get('session_title', '')} ｜ 主力模型：{st.session_state['main_provider']}/{st.session_state['main_model']}"
    f" ｜ 知识库 {len(list_docs())} 个文档 ｜ 文件 {len(files)} 个"
    + (" ｜ 💾 会话自动保存到 data/sessions/" if True else "")
)