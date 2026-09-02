"""Chroma 向量库封装：增删查 + embedding 统一由本地模型完成。"""
from __future__ import annotations

import chromadb
from sentence_transformers import SentenceTransformer

from .config import CHROMA_DIR, CHUNK_SIZE, CHUNK_OVERLAP, DEFAULT_TOP_K, EMBED_MODEL

COLLECTION_NAME = "kb_docs"

_embedder = None
_client = None
_collection = None


def get_embedder() -> SentenceTransformer:
    global _embedder
    if _embedder is None:
        _embedder = SentenceTransformer(EMBED_MODEL)
    return _embedder


def get_collection():
    global _client, _collection
    if _collection is None:
        CHROMA_DIR.mkdir(parents=True, exist_ok=True)
        _client = chromadb.PersistentClient(path=str(CHROMA_DIR))
        _collection = _client.get_or_create_collection(
            name=COLLECTION_NAME, metadata={"hnsw:space": "cosine"}
        )
    return _collection


def _embed(texts: list[str]) -> list[list[float]]:
    return get_embedder().encode(texts, normalize_embeddings=True).tolist()


def add_chunks(ids: list[str], docs: list[str], metadatas: list[dict]) -> None:
    get_collection().add(
        ids=ids, embeddings=_embed(docs), documents=docs, metadatas=metadatas
    )


def query(query_text: str, top_k: int = DEFAULT_TOP_K) -> list[tuple[str, dict, float]]:
    """返回 [(片段文本, 元数据, 距离)]，按相关度排序。"""
    col = get_collection()
    if col.count() == 0:
        return []
    n = min(top_k, col.count())
    res = col.query(
        query_embeddings=_embed([query_text]),
        n_results=n,
        include=["documents", "metadatas", "distances"],
    )
    out = []
    docs = (res.get("documents") or [[]])[0]
    metas = (res.get("metadatas") or [[]])[0]
    dists = (res.get("distances") or [[]])[0]
    for d, m, dist in zip(docs, metas, dists):
        out.append((d, m or {}, float(dist)))
    return out


def count() -> int:
    return get_collection().count()


def list_docs() -> list[dict]:
    """按 doc_name 汇总：返回 [{doc_name, chunks}]。"""
    col = get_collection()
    if col.count() == 0:
        return []
    res = col.get(include=["metadatas"])
    stats: dict[str, int] = {}
    for m in res.get("metadatas") or []:
        name = (m or {}).get("doc_name", "未知")
        stats[name] = stats.get(name, 0) + 1
    return [
        {"doc_name": name, "chunks": n} for name, n in sorted(stats.items())
    ]


def remove_doc(doc_name: str) -> int:
    """从向量库删除某个文档的全部片段，返回删除条数。"""
    col = get_collection()
    before = col.count()
    col.delete(where={"doc_name": doc_name})
    return before - col.count()


def reset() -> None:
    """清空向量库（测试用）。"""
    get_collection().delete(where={})