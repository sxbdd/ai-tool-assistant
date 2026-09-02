"""全局配置：从 .env / 环境变量读取 DeepSeek 相关配置。

密钥只放 .env（已被 .gitignore 忽略），绝不写死在代码里。
"""
from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

# 项目根目录
BASE_DIR = Path(__file__).resolve().parent.parent

# 加载项目根目录下的 .env（若存在）
load_dotenv(BASE_DIR / ".env")

DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "").strip()
DEEPSEEK_BASE_URL = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com").strip()
DEEPSEEK_MODEL = os.getenv("DEEPSEEK_MODEL", "deepseek-chat").strip()
MAX_ITERATIONS = int(os.getenv("AGENT_MAX_ITERATIONS", "10"))


def check_api_key() -> None:
    """启动前校验 API Key，未配置时给出清晰提示。"""
    if not DEEPSEEK_API_KEY or DEEPSEEK_API_KEY.startswith("sk-你的"):
        raise RuntimeError(
            "未检测到有效的 DEEPSEEK_API_KEY。\n"
            "请把项目根目录的 .env.example 复制为 .env，"
            "并填入你在 https://platform.deepseek.com 申请的 Key。"
        )