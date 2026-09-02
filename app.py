"""网页对话界面（Streamlit）v2.1
mini-Codex：多模型切换 + 文件上传 + 本地文件夹导入 + RAG 知识库 + 免费识图。

启动：python -m streamlit run app.py
"""
from __future__ import annotations

import os
from pathlib import Path

import streamlit as st
from langchain_core.messages import AIMessage as LCAIMessage
from langchain_core.messages import HumanMessage

from agent.graph import ask
from agent.model_providers import PROVIDERS, list_providers, provider_status
from rag.config import RAW_DIR, SUPPORTED_EXT, UPLOAD_DIR, ensure_dirs
from rag.ingest import ingest_file, ingest_folder
from rag.store import list_docs, remove_doc

ensure_dirs()

st.set_page_config(page_title="AI 工具调用小助手", page_icon="🤖", layout="wide")
st.title("🤖 AI 工具调用小助手")
st.caption("Agent + Function Calling + RAG 知识库 + 识图 + 联网搜索：让大模型「查资料、看图、查网、调工具」完成多步任务")

# ============================================================ 侧栏
with st.sidebar:
    st.header("⚙️ 模型设置")
    status = provider_status()
    main_provider = st.selectbox(
        "① 主力对话：服务商", list_providers(), index=0,
        format_func=lambda n: f"{n} · {status[n]}",
        help="未配置 Key 的选项会报错，请在 .env 补充对应 Key",
    )
    main_models = PROVIDERS[main_provider]["models"]
    _env_main_model = os.getenv(PROVIDERS[main_provider]["model_env"], "")
    main_model = st.selectbox(
        "② 主力对话：具体模型", main_models,
        index=main_models.index(_env_main_model) if _env_main_model in main_models else 0,
        help="以该服务商实际可用型号为准；想用列表外的型号可直接改 .env 对应变量",
    )
    os.environ["DEFAULT_PROVIDER"] = main_provider
    vis_providers = [n for n in list_providers() if PROVIDERS[n]["supports_vision"]]
    if vis_providers:
        vision_provider = st.selectbox(
            "③ 识图：服务商", vis_providers,
            index=vis_providers.index("zhipu") if "zhipu" in vis_providers else 0,
            format_func=lambda n: f"{n} · {status[n]}",
        )
        vis_models = PROVIDERS[vision_provider]["models"]
        _env_vis_model = os.getenv(PROVIDERS[vision_provider]["model_env"], "")
        vision_model = st.selectbox(
            "④ 识图：具体模型", vis_models,
            index=vis_models.index(_env_vis_model) if _env_vis_model in vis_models else 0,
        )
        os.environ["VISION_PROVIDER"] = vision_provider
        os.environ[PROVIDERS[vision_provider]["model_env"]] = vision_model

    st.divider()
    st.header("📚 知识库管理")
    st.caption("支持：" + " / ".join(sorted(SUPPORTED_EXT)) + "；单文件 ≤20MB；文档复制进沙盒 data/kb/ 后入库")

    up_files = st.file_uploader(
        "① 上传文件入库", type=[e.lstrip(".") for e in sorted(SUPPORTED_EXT)],
        accept_multiple_files=True,
    )
    if up_files:
        if st.button("② 上传并入库", use_container_width=True):
            ok_cnt = err_cnt = 0
            for f in up_files:
                if f.size > 20 * 1024 * 1024:
                    st.error(f"{f.name} 超过 20MB，已跳过")
                    err_cnt += 1
                    continue
                dest = UPLOAD_DIR / f.name
                dest.write_bytes(f.getvalue())
                r = ingest_file(dest)
                if "error" in r:
                    st.error(f"{f.name}：{r['error']}")
                    err_cnt += 1
                else:
                    st.success(f"{f.name} -> {r.get('chunks', 0)} 个片段")
                    ok_cnt += 1
            st.caption(f"本次：成功 {ok_cnt}，失败 {err_cnt}")

    st.markdown("**③ 导入本地文件夹（绝对路径）**")
    folder_path = st.text_input("文件夹路径", placeholder="如 D:\\我的笔记", label_visibility="collapsed")
    if folder_path.strip() and st.button("④ 扫描并导入", use_container_width=True):
        folder = Path(folder_path.strip())
        if not folder.is_dir():
            st.error(f"不是有效文件夹：{folder}")
        else:
            files = [x for x in folder.rglob("*") if x.is_file() and x.suffix.lower() in SUPPORTED_EXT]
            st.info(f"共扫描到 {len(files)} 个支持的文件，正在复制并入库……")
            results = ingest_folder(folder)
            chunks = sum(r.get("chunks", 0) for r in results if "error" not in r)
            errors = [r for r in results if "error" in r]
            st.success(f"完成：入库 {chunks} 个片段 / {len(results) - len(errors)} 个文件")
            for r in errors:
                st.warning(f"{r.get('doc_name', '')}：{r['error']}")
    st.caption("安全说明：只读你的原文件夹，文件会复制到项目沙盒 data/kb/raw/，不改动原目录。")

    docs = list_docs()
    if docs:
        st.divider()
        st.markdown(f"**已入库文档（{len(docs)}）**")
        st.dataframe(docs, use_container_width=True, hide_index=True)
        sel = st.multiselect("选择要删除的文档（仅移除向量索引）", [d["doc_name"] for d in docs])
        if sel and st.button("删除选中", use_container_width=True):
            for n in sel:
                remove_doc(n)
            st.rerun()
    else:
        st.caption("知识库为空：请上传文件或导入本地文件夹。")

# ============================================================ 主区
if "history" not in st.session_state:
    st.session_state.history = []  # [{role, content, trace}]


def render_messages() -> None:
    for item in st.session_state.history:
        with st.chat_message(item["role"]):
            if item.get("img"):
                st.image(item["img"], width=180)
            st.markdown(item["content"])
            if item.get("trace"):
                with st.expander("🧰 工具调用过程"):
                    for s in item["trace"]:
                        st.markdown(s)


def run_ask(prompt: str, img_bytes: bytes | None = None) -> None:
    """把用户消息交给 Agent，展示最终回答与工具轨迹。"""
    with st.chat_message("user"):
        if img_bytes:
            st.image(img_bytes, width=180)
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
                answer, trace = ask(prompt, lc_history, provider=main_provider, model=main_model)
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


render_messages()

# ---- 图片识图 ----
with st.container(border=True):
    img_file = st.file_uploader(
        "🖼️ 上传图片让 AI 识别（可选）", type=["png", "jpg", "jpeg", "gif", "webp"], key="img_upload"
    )
    if img_file is not None:
        col_prev, col_btn = st.columns([1, 2])
        with col_prev:
            st.image(img_file.getvalue(), width=180)
        with col_btn:
            if st.button("🔍 让 AI 识别这张图片", use_container_width=True):
                dest = UPLOAD_DIR / img_file.name
                dest.write_bytes(img_file.getvalue())
                run_ask(
                    f"请调用 analyze_image 工具分析 data/uploads/{img_file.name} 这张图片，"
                    "告诉我图片里有什么，包括所有文字。",
                    img_bytes=img_file.getvalue(),
                )

# ---- 对话输入 ----
prompt = st.chat_input(
    "试试：根据我的笔记解释 RAG 是什么｜北京今天天气怎么样？｜整理 data/示例-销售数据.csv"
)
if prompt:
    run_ask(prompt)