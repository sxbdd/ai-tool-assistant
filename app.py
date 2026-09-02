"""网页对话界面（Streamlit）v2.2 — 对齐 Codex 桌面版体验
折叠式侧栏（模型/文件库/知识库）+ 附件式上传（中文组件优先）+ 文件库点击预览大图。
"""
from __future__ import annotations

import os
from datetime import datetime
from pathlib import Path

import streamlit as st
from langchain_core.messages import AIMessage as LCAIMessage
from langchain_core.messages import HumanMessage

from agent.graph import ask
from agent.model_providers import PROVIDERS, list_providers, provider_status
from rag.config import IMAGE_EXT, SUPPORTED_EXT, UPLOAD_DIR, ensure_dirs
from rag.ingest import ingest_file, ingest_folder
from rag.store import list_docs, remove_doc

ensure_dirs()

# ---- 可选第三方组件（都来自 GitHub 免费开源）----
try:  # 中文文件上传组件 streamlit-chunk-file-uploader（GracefulTabby, MIT）
    from streamlit_chunk_file_uploader import uploader as chunk_uploader
    HAS_CHUNK = True
except Exception:  # noqa: BLE001
    HAS_CHUNK = False

try:  # 图片点击缩放组件 streamlit-image-zoom（vgilabert94, MIT）
    from streamlit_image_zoom import image_zoom
    HAS_ZOOM = True
except Exception:  # noqa: BLE001
    HAS_ZOOM = False

st.set_page_config(page_title="AI 工具调用小助手", page_icon="🤖", layout="wide")

# ============================================================ 小工具
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
    """保存上传文件，重名时自动加序号，返回最终路径。"""
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    p = UPLOAD_DIR / name
    i = 1
    while p.exists():
        p = UPLOAD_DIR / f"{Path(name).stem}({i}){Path(name).suffix}"
        i += 1
    p.write_bytes(data)
    return p


def ensure_defaults() -> None:
    st.session_state.setdefault("history", [])
    st.session_state.setdefault("pending", [])
    st.session_state.setdefault("main_provider", "deepseek")
    st.session_state.setdefault("main_model", "deepseek-chat")
    st.session_state.setdefault("preview_file", None)


ensure_defaults()


def render_messages() -> None:
    for item in st.session_state.history:
        with st.chat_message(item["role"]):
            if item.get("img"):
                st.image(item["img"], width=120)
            st.markdown(item["content"])
            if item.get("trace"):
                with st.expander("🧰 工具调用过程"):
                    for s in item["trace"]:
                        st.markdown(s)


def run_ask(prompt: str, img_bytes: bytes | None = None) -> None:
    with st.chat_message("user"):
        if img_bytes:
            st.image(img_bytes, width=120)
        st.markdown(prompt)
    st.session_state.history.append({"role": "user", "content": prompt})

    lc_history: list = []
    for item in st.session_state.history[:-1]:
        if item["role"] == "user":
            lc_history.append(HumanMessage(content=item["content"]))
        else:
            lc_history.append(LCAIMessage(content=item["content"]))

    with st.chat_message("assistant"):
        try:
            with st.spinner("Agent 正在思考并调用工具……"):
                answer, trace = ask(
                    prompt, lc_history,
                    provider=st.session_state["main_provider"],
                    model=st.session_state["main_model"],
                )
        except RuntimeError as exc:
            st.error(str(exc))
            answer, trace = None, []
        except Exception as exc:  # noqa: BLE001
            st.error(f"出错了：{exc}")
            answer, trace = None, []
        if answer:
            st.markdown(answer)
            if trace:
                with st.expander("🧰 工具调用过程", expanded=len(trace) > 1):
                    for s in trace:
                        st.markdown(s)
            st.session_state.history.append(
                {"role": "assistant", "content": answer, "trace": trace}
            )


# ============================================================ 侧栏
with st.sidebar:
    st.header("🤖 AI 工具调用小助手")
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
            meta = st.caption(
                f"{icon_of(sel_path)} {sel_name} ｜ {fmt_size(sel_path.stat().st_size)}"
                f" ｜ {datetime.fromtimestamp(sel_path.stat().st_mtime):%m-%d %H:%M}"
            )
            c1, c2, c3, c4 = st.columns(4)
            if c1.button("👁 预览", use_container_width=True):
                st.session_state["preview_file"] = str(sel_path) if st.session_state.get("preview_file") != str(sel_path) else None
            if c2.button("📤 发送", use_container_width=True):
                data = sel_path.read_bytes()
                if is_image(sel_path):
                    run_ask(
                        f"请调用 analyze_image 工具分析 data/uploads/{sel_name} 这张图片，告诉我内容。",
                        img_bytes=data,
                    )
                else:
                    run_ask(f"请用合适工具读取并处理 data/uploads/{sel_name}，告诉我结果。")
            if c3.button("📚 入库", use_container_width=True):
                r = ingest_file(sel_path)
                if "error" in r:
                    st.error(r["error"])
                else:
                    st.success(f"已入库：{r.get('chunks', 0)} 个片段")
            if c4.button("🗑 删除", use_container_width=True):
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
                        st.caption("滚动鼠标可缩放 / 拖动查看")
                    except Exception:  # noqa: BLE001
                        st.image(str(pp), width=520)
                elif is_image(pp):
                    st.image(str(pp), width=520)
                elif pp.suffix.lower() in {".txt", ".md", ".csv"}:
                    st.code(pp.read_text(encoding="utf-8", errors="replace")[:2000], language=None)
                else:
                    st.caption("该类型不支持直接预览，可用「发送」让 AI 处理或「入库」进知识库。")

    with st.expander("📚 知识库管理", expanded=False):
        st.caption("支持：" + " / ".join(sorted(SUPPORTED_EXT)) + "；单文件 ≤20MB；文档复制进沙盒 data/kb/")
        kb_up = st.file_uploader(
            "上传文档入库", type=[e.lstrip(".") for e in sorted(SUPPORTED_EXT)],
            accept_multiple_files=True,
        )
        if kb_up and st.button("上传并入库", use_container_width=True):
            for f in kb_up:
                if f.size > 20 * 1024 * 1024:
                    st.error(f"{f.name} 超过 20MB，跳过")
                    continue
                dest = save_upload(f.name, f.getvalue())
                r = ingest_file(dest)
                st.success(f"{f.name} -> {r.get('chunks', 0)} 片段" if "error" not in r else f"{f.name}：{r['error']}")
        st.markdown("**导入本地文件夹（绝对路径）**")
        folder_path = st.text_input("文件夹路径", placeholder="如 D:\\我的笔记", label_visibility="collapsed")
        if folder_path.strip() and st.button("扫描并导入", use_container_width=True):
            folder = Path(folder_path.strip())
            if not folder.is_dir():
                st.error(f"不是有效文件夹：{folder}")
            else:
                n = len([x for x in folder.rglob("*") if x.is_file() and x.suffix.lower() in SUPPORTED_EXT])
                st.info(f"扫描到 {n} 个支持文件，正在入库……")
                results = ingest_folder(folder)
                chunks = sum(r.get("chunks", 0) for r in results if "error" not in r)
                st.success(f"完成：入库 {chunks} 片段 / {sum(1 for r in results if 'error' not in r)} 个文件")
        docs = list_docs()
        if docs:
            st.divider()
            st.markdown(f"**已入库（{len(docs)}）**")
            st.dataframe(docs, use_container_width=True, hide_index=True)
            sel_del = st.multiselect("删除（仅移除索引）", [d["doc_name"] for d in docs])
            if sel_del and st.button("删除选中", use_container_width=True):
                for n in sel_del:
                    remove_doc(n)
                st.rerun()
        else:
            st.caption("知识库为空：上传文件或导入本地文件夹后可提问。")

# ============================================================ 主区
st.title("🤖 AI 工具调用小助手")
st.caption("查资料 · 看图 · 联网搜 · 调工具 —— 让大模型帮你完成多步任务")

render_messages()

# ---- 附件区（弹出式，不再占大框）----
with st.popover("➕ 附件", use_container_width=False):
    st.caption("可上传图片 / 文档；文件会存入 data/uploads/（侧栏「文件库」可管理）。")
    picked = None
    if HAS_CHUNK:
        try:
            picked = chunk_uploader(
                label="选择文件", type=None, key="attach_chunk",
                uploader_msg="点击选择文件，或拖拽到此处",
            )
            st.caption("（中文上传组件）")
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
    if st.button(f"📤 发送 {len(pending)} 个附件给 AI", use_container_width=True):
        lines, first_img = [], None
        for nm, data in pending:
            p = save_upload(nm, data)
            if p.suffix.lower() in IMAGE_EXT and first_img is None:
                first_img = data
                lines.append(f"- {nm}（图片 → 请调用 analyze_image 分析 data/uploads/{p.name}）")
            elif p.suffix.lower() in IMAGE_EXT:
                lines.append(f"- {nm}（图片 → 请调用 analyze_image 分析 data/uploads/{p.name}）")
            else:
                lines.append(f"- {nm}（文档 → 请用 read_file / 相关工具处理 data/uploads/{p.name}）")
        prompt = "我上传了以下文件（已保存到 data/uploads/）：\n" + "\n".join(lines) + "\n\n请逐一查看并处理：图片请识图告诉我内容，文档请读取并总结/按需整理。"
        run_ask(prompt, img_bytes=first_img)
        st.session_state["pending"] = []

# ---- 对话输入 ----
prompt = st.chat_input("输入任务，如：根据我的笔记解释 RAG｜北京天气｜联网查最新 AI 新闻")
if prompt:
    run_ask(prompt)