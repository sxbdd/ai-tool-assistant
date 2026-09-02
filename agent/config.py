"""全局配置：从 .env / 环境变量读取各模型服务商配置。

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

# ---- DeepSeek（默认主力对话）----
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "").strip()
DEEPSEEK_BASE_URL = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com").strip()
DEEPSEEK_MODEL = os.getenv("DEEPSEEK_MODEL", "deepseek-chat").strip()

# ---- 默认主力服务商 / 识图服务商 ----
DEFAULT_PROVIDER = os.getenv("DEFAULT_PROVIDER", "deepseek").strip().lower()
VISION_PROVIDER = os.getenv("VISION_PROVIDER", "zhipu").strip().lower()

# ---- 智谱 GLM（识图免费档）----
ZHIPU_API_KEY = os.getenv("ZHIPU_API_KEY", "").strip()
ZHIPU_VISION_MODEL = os.getenv("ZHIPU_VISION_MODEL", "glm-4v-flash").strip()

# ---- 通义 Qwen（备选识图）----
DASHSCOPE_API_KEY = os.getenv("DASHSCOPE_API_KEY", "").strip()
QWEN_VISION_MODEL = os.getenv("QWEN_VISION_MODEL", "qwen-vl-plus").strip()

# ---- OpenAI（可选付费）----
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "").strip()
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini").strip()

# ---- 联网搜索 ----
WEB_SEARCH_BACKEND = os.getenv("WEB_SEARCH_BACKEND", "bing").strip().lower()  # bing / tavily
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY", "").strip()  # 可选：更稳的搜索 API

MAX_ITERATIONS = int(os.getenv("AGENT_MAX_ITERATIONS", "10"))
# 每次发给模型的历史消息条数上限（超出裁掉最旧的，控制上下文长度）
MAX_CONTEXT_MESSAGES = int(os.getenv("MAX_CONTEXT_MESSAGES", "12"))
# 网络/瞬时错误自动重试次数
ASK_RETRIES = int(os.getenv("ASK_RETRIES", "2"))


def check_api_key() -> None:
    """启动前校验"主力模型"的 Key，未配置时给出清晰提示。"""
    key = {
        "deepseek": DEEPSEEK_API_KEY,
        "zhipu": ZHIPU_API_KEY,
        "qwen": DASHSCOPE_API_KEY,
        "openai": OPENAI_API_KEY,
    }.get(DEFAULT_PROVIDER, "")
    if not key:
        raise RuntimeError(
            f"未检测到主力模型（{DEFAULT_PROVIDER}）的 API Key。\n"
            "请在项目根目录 .env 中正确配置，参考 .env.example。"
        )