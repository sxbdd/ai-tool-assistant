"""文档摄取：解析(PDF按页/文本) -> 切分(递归+重叠) -> 向量化 -> 入库。

安全约定：文档统一复制到 data/kb/raw/ 沙盒后再入库，不直接引用用户原目录文件。
"""
from __future__ import annotations

import re
import shutil
from pathlib import Path

from pypdf import PdfReader

from .config import CHUNK_OVERLAP, CHUNK_SIZE, RAW_DIR, SUPPORTED_EXT, ensure_dirs
from .store import add_chunks, remove_doc


# ---------------------------------------------------------------- 切分
def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    """递归切分：优先按空行/段落/句号，中文友好；相邻块重叠 overlap 字。"""
    seps = ["\n\n", "\n", "。", "！", "？", "；", ". ", "! ", "? ", " "]
    text = text.replace("\r\n", "\n").strip()
    if not text:
        return []

    def _split(t: str, level: int) -> list[str]:
        if len(t) <= chunk_size or level >= len(seps):
            return [t] if t.strip() else []
        sep = seps[level]
        parts = t.split(sep)
        out: list[str] = []
        cur = ""
        for p in parts:
            piece = p if not cur else sep + p
            if cur and len(cur) + len(piece) <= chunk_size:
                cur += piece
            else:
                if cur:
                    out.append(cur)
                if len(p) > chunk_size:
                    out.extend(_split(p, level + 1))
                else:
                    cur = p
        if cur:
            out.append(cur)
        return out

    pieces = _split(text, 0)
    chunks: list[str] = []
    prev = ""
    for p in pieces:
        p = p.strip()
        if not p:
            continue
        if prev and overlap:
            p = prev[-overlap:] + p
        chunks.append(p)
        prev = p
    return chunks


# ---------------------------------------------------------------- 解析
def extract_pages(path: Path) -> list[tuple[str, int | None]]:
    """返回 [(文本, 页码或None)]。PDF 按页返回，其余整个文件一页。"""
    ext = path.suffix.lower()
    if ext == ".pdf":
        reader = PdfReader(str(path))
        pages = []
        for i, page in enumerate(reader.pages, start=1):
            text = (page.extract_text() or "").strip()
            if text:
                pages.append((text, i))
        return pages
    text = path.read_text(encoding="utf-8", errors="replace")
    return [(text, None)] if text.strip() else []


# ---------------------------------------------------------------- 入库
def ingest_text_pages(
    doc_name: str, pages: list[tuple[str, int | None]]
) -> dict:
    """把 [(文本,页码)] 切分后写入向量库；doc_name 相同会先清旧再写（覆盖式）。"""
    remove_doc(doc_name)  # 覆盖旧版本，避免重复
    ids: list[str] = []
    docs: list[str] = []
    metas: list[dict] = []
    seq = 0
    for text, page in pages:
        for c in chunk_text(text):
            ids.append(f"{doc_name}__{seq:05d}")
            docs.append(c)
            meta = {"doc_name": doc_name, "chunk_index": seq}
            if page is not None:
                meta["page"] = int(page)
            metas.append(meta)
            seq += 1
    if docs:
        add_chunks(ids, docs, metas)
    return {"doc_name": doc_name, "chunks": len(docs)}


def ingest_file(path: Path, doc_name: str | None = None) -> dict:
    """把单个文件复制进沙盒后入库。doc_name 默认取文件名（不含扩展名）。"""
    ensure_dirs()
    path = Path(path)
    if not path.exists():
        return {"error": f"文件不存在：{path}"}
    if path.suffix.lower() not in SUPPORTED_EXT:
        return {"error": f"不支持的格式：{path.suffix}（支持 {'/'.join(sorted(SUPPORTED_EXT))}）"}
    name = doc_name or path.stem
    # 复制进沙盒（已在沙盒内的不重复复制）
    if not str(path.resolve()).startswith(str(RAW_DIR.resolve())):
        dest_dir = RAW_DIR / name
        dest_dir.mkdir(parents=True, exist_ok=True)
        dest = dest_dir / path.name
        shutil.copy2(path, dest)
        path = dest
    pages = extract_pages(path)
    if not pages:
        return {"doc_name": name, "chunks": 0, "warning": "未能从文件中提取到文字（可能是扫描版 PDF）"}
    stats = ingest_text_pages(name, pages)
    stats["source"] = str(path)
    return stats


def ingest_folder(folder: Path, group: str | None = None) -> list[dict]:
    """把文件夹内所有受支持文档复制进沙盒并批量入库。"""
    ensure_dirs()
    folder = Path(folder)
    if not folder.is_dir():
        return [{"error": f"文件夹不存在：{folder}"}]
    group = group or folder.name
    results: list[dict] = []
    files = [
        p for p in folder.rglob("*")
        if p.is_file() and p.suffix.lower() in SUPPORTED_EXT
    ]
    for f in files:
        rel = f.relative_to(folder)
        doc_name = str(rel).replace("\\", "/")
        dest_dir = RAW_DIR / group / rel.parent
        dest_dir.mkdir(parents=True, exist_ok=True)
        dest = dest_dir / f.name
        shutil.copy2(f, dest)
        try:
            pages = extract_pages(dest)
            if not pages:
                results.append({"doc_name": doc_name, "chunks": 0, "warning": "无可提取文字"})
                continue
            stats = ingest_text_pages(doc_name, pages)
            stats["doc_name"] = doc_name
            results.append(stats)
        except Exception as exc:  # noqa: BLE001
            results.append({"doc_name": doc_name, "error": str(exc)})
    return results