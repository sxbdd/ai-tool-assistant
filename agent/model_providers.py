"""模型抽象层：多 Provider 可切换（全部走 OpenAI 兼容协议）。

设计：
- PROVIDERS 注册表描述每个服务商的 base_url / Key 环境变量 / 默认模型 / 是否支持图片；
- 换模型 = 换 provider + model，代码不用改；
- 视觉桥(analyze_image) 用 VISION_PROVIDER 指定的"识图模型"，主对话用 DEFAULT_PROVIDER。
"""
from __future__ import annotations

import os

from langchain_openai import ChatOpenAI

from . import config

# 各服务商注册表（Key 一律从环境变量/.env 读取，绝不写死）
PROVIDERS: dict[str, dict] = {
    "deepseek": {
        "label": "DeepSeek",
        "base_url": "https://api.deepseek.com",
        "api_key_env": "DEEPSEEK_API_KEY",
        "model_env": "DEEPSEEK_MODEL",
        "default_model": "deepseek-chat",
        "supports_vision": False,
        "models": ["deepseek-chat", "deepseek-reasoner", "deepseek-v4-flash-vision-exp"],
    },
    "zhipu": {
        "label": "智谱 GLM",
        "base_url": "https://open.bigmodel.cn/api/paas/v4",
        "api_key_env": "ZHIPU_API_KEY",
        "model_env": "ZHIPU_VISION_MODEL",
        "default_model": "glm-4v-flash",
        "supports_vision": True,
        "models": ["glm-4v-flash", "glm-4.6v-flash", "glm-4.1v-thinking-flash", "glm-4v-plus"],
    },
    "qwen": {
        "label": "通义 Qwen",
        "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
        "api_key_env": "DASHSCOPE_API_KEY",
        "model_env": "QWEN_VISION_MODEL",
        "default_model": "qwen-vl-plus",
        "supports_vision": True,
        "models": ["qwen-vl-plus", "qwen-vl-max", "qwen3-vl-flash"],
    },
    "openai": {
        "label": "OpenAI GPT",
        "base_url": "https://api.openai.com/v1",
        "api_key_env": "OPENAI_API_KEY",
        "model_env": "OPENAI_MODEL",
        "default_model": "gpt-4o-mini",
        "supports_vision": True,
        "models": ["gpt-4o-mini", "gpt-4o", "gpt-4.1-mini"],
    },
}


def list_providers() -> list[str]:
    return list(PROVIDERS)


def provider_status() -> dict[str, str]:
    """给 UI 用：返回每个 provider 是否已配置 Key。"""
    out = {}
    for name, p in PROVIDERS.items():
        key = os.getenv(p["api_key_env"], "").strip()
        out[name] = "已配置" if key else "未配置"
    return out


def _resolve(provider: str | None, model: str | None) -> tuple[dict, str, str]:
    name = (provider or config.DEFAULT_PROVIDER or "deepseek").strip().lower()
    if name not in PROVIDERS:
        raise ValueError(f"未知模型服务商：{name}，可选：{', '.join(PROVIDERS)}")
    p = PROVIDERS[name]
    api_key = os.getenv(p["api_key_env"], "").strip()
    if not api_key:
        raise RuntimeError(
            f"[{p['label']}] 未配置 API Key：请在项目根目录 .env 中设置 {p['api_key_env']}=sk-xxx"
        )
    model_name = (model or os.getenv(p["model_env"], "") or p["default_model"]).strip()
    return p, api_key, model_name


def create_chat_model(
    provider: str | None = None,
    model: str | None = None,
    temperature: float = 0.2,
    timeout: int = 90,
) -> ChatOpenAI:
    """创建主对话/任意 OpenAI 兼容模型客户端。"""
    p, api_key, model_name = _resolve(provider, model)
    return ChatOpenAI(
        model=model_name,
        api_key=api_key,
        base_url=p["base_url"],
        temperature=temperature,
        timeout=timeout,
    )


def create_vision_model(timeout: int = 90) -> ChatOpenAI:
    """创建视觉桥专用模型。

    优先读取运行时环境变量 VISION_PROVIDER（网页端切换即时生效），
    其次取 .env 中的 VISION_PROVIDER，最后默认 zhipu。
    """
    vision_provider = os.getenv("VISION_PROVIDER") or config.VISION_PROVIDER or "zhipu"
    return create_chat_model(provider=vision_provider, temperature=0.1, timeout=timeout)