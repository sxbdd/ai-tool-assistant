"""RAG 配置：目录、切分参数、embedding 模型。"""
from __future__ import annotations

import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

KB_DIR = BASE_DIR / "data" / "kb"
RAW_DIR = KB_DIR / "raw"        # 原始文档副本（沙盒内）
CHROMA_DIR = KB_DIR / "chroma"  # 向量库持久化
UPLOAD_DIR = BASE_DIR / "data" / "uploads"

EMBED_MODEL = os.getenv("EMBED_MODEL", "BAAI/bge-small-zh-v1.5").strip()
CHUNK_SIZE = int(os.getenv("RAG_CHUNK_SIZE", "500"))
CHUNK_OVERLAP = int(os.getenv("RAG_CHUNK_OVERLAP", "80"))
DEFAULT_TOP_K = int(os.getenv("RAG_TOP_K", "4"))
# 余弦距离阈值：距离大于该值视为"不相关"（cos 相似度约 < 0.35）
MAX_DISTANCE = float(os.getenv("RAG_MAX_DISTANCE", "0.65"))
MAX_FILE_MB = 20

SUPPORTED_EXT = {".pdf", ".txt", ".md", ".csv"}
IMAGE_EXT = {".png", ".jpg", ".jpeg", ".gif", ".webp"}


def ensure_dirs() -> None:
    for d in (RAW_DIR, CHROMA_DIR, UPLOAD_DIR):
        d.mkdir(parents=True, exist_ok=True)