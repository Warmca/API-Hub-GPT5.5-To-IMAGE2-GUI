#!/usr/bin/env python3
"""API_GPT5_5_to_IMAGE2.py

个人 Windows 工具。通过 OpenAI 兼容中转的 Responses API 调 gpt-5.5 + image_generation
工具，产出 gpt-image-2 质量的图。支持 freemodel.dev / OpenAI 官方 / laozhang.ai /
aimlapi / 自定义中转。

公开接口（GUI 直接 import）：
    generate, refine_prompt_fields_with_gpt5,
    image_to_data_url, image_size,
    dedupe_prompt_fields, read_text_file, repair_mojibake,
    configure_stdio,
    PROVIDERS, DEFAULT_PROVIDER, REF_IMAGE_DEFAULT_MAX_EDGE,
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import io
import json
import mimetypes
import os
import random
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable, Iterable

import requests


# ─────────────────────────────────────────────────────────────────────────────
# 1. 常量与上游配置
# ─────────────────────────────────────────────────────────────────────────────

class ImageLimits:
    """gpt-image-2 输入端硬约束。"""
    MAX_EDGE   = 3840
    MAX_PIXELS = 8_294_400
    MIN_PIXELS = 655_360
    MAX_RATIO  = 3.0
    QUALITIES  = ("auto", "low", "medium", "high")
    BG_CHOICES = ("auto", "opaque")
    MOD_CHOICES = ("auto", "low")


class RefImage:
    """参考图本地压缩参数。"""
    DEFAULT_MAX_EDGE = 1536
    JPEG_QUALITY     = 90


REF_IMAGE_DEFAULT_MAX_EDGE = RefImage.DEFAULT_MAX_EDGE  # 公开常量给 GUI

# HTTP 状态码视作可重试
TRANSIENT_HTTP = frozenset({408, 409, 425, 429, 500, 502, 503, 504, 520, 522, 524})

# 响应里可能装最终图 b64 的 key
_FINAL_B64_KEYS = frozenset({"result", "image_b64", "image_base64", "b64_json"})
# 响应里可能装预览帧 b64 的 key
_PARTIAL_B64_KEYS = frozenset({"partial_image_b64", "partial_image_base64"})


@dataclass(frozen=True)
class ProviderConfig:
    """单个中转/官方端点的配置。"""
    key: str
    label: str
    base_url: str
    env_keys: tuple[str, ...]
    env_list_key: str
    # 文本/润色调用走哪种 API 形态。
    # "responses"         → 走 /v1/responses（OpenAI 新版，gpt-5.5 / gpt-image-2 必须）
    # "chat_completions"  → 走 /v1/chat/completions（DeepSeek / 智谱 / Kimi / Qwen / Grok 等大多数 OpenAI 兼容厂商）
    # "anthropic_messages"→ 走 /v1/messages（Anthropic 官方，需 x-api-key + anthropic-version 头）
    text_api: str = "responses"
    # 没找到本 provider 的 token 时，是否兜底用 FREEMODEL_TOKEN。
    # 对于和 freemodel 不同源的厂商（如 DeepSeek），用错 token 比报错更糟。
    fallback_to_freemodel: bool = True
    # 该 provider 上典型可用的文本/润色模型；第一个是默认值，其余是 GUI 下拉的可选项。
    # 切换 polish provider 时 GUI 会自动把"润色模型"框填成 default_text_model（=第 0 项）。
    text_models: tuple[str, ...] = ()

    @property
    def is_custom(self) -> bool:
        return self.key == "custom"

    @property
    def default_text_model(self) -> str:
        return self.text_models[0] if self.text_models else ""


# 旧的 freemodel 多 token 命名，单独抽出来避免逐个写
_FM_TOKEN_KEYS = tuple(
    "FREEMODEL_TOKEN" if i == 1 else f"FREEMODEL_TOKEN_{i}"
    for i in range(1, 11)
)

PROVIDERS: dict[str, ProviderConfig] = {
    cfg.key: cfg
    for cfg in (
        ProviderConfig(
            key="freemodel", label="freemodel.dev",
            base_url="https://api.freemodel.dev/v1",
            env_keys=_FM_TOKEN_KEYS,
            env_list_key="FREEMODEL_TOKENS",
            text_models=("gpt-5.5", "gpt-5", "gpt-5-mini", "gpt-4.1", "gpt-4o"),
        ),
        ProviderConfig(
            key="openai", label="OpenAI Official",
            base_url="https://api.openai.com/v1",
            env_keys=("OPENAI_API_KEY", "OPENAI_API_KEY_2", "OPENAI_API_KEY_3"),
            env_list_key="OPENAI_API_KEYS",
            text_models=("gpt-5.5", "gpt-5", "gpt-5-mini", "gpt-4.1", "gpt-4o"),
        ),
        ProviderConfig(
            key="laozhang", label="laozhang.ai",
            base_url="https://api.laozhang.ai/v1",
            env_keys=("LAOZHANG_API_KEY", "LAOZHANG_TOKEN"),
            env_list_key="LAOZHANG_TOKENS",
            text_models=("gpt-5.5", "gpt-5", "gpt-4o"),
        ),
        ProviderConfig(
            key="aiml", label="AI/ML API",
            base_url="https://api.aimlapi.com/v1",
            env_keys=("AIML_API_KEY", "AIMLAPI_KEY"),
            env_list_key="AIML_API_KEYS",
            text_models=("gpt-5.5", "gpt-5", "gpt-4o"),
        ),
        ProviderConfig(
            key="deepseek", label="DeepSeek（仅润色）",
            base_url="https://api.deepseek.com/v1",
            env_keys=("DEEPSEEK_API_KEY", "DEEPSEEK_TOKEN"),
            env_list_key="DEEPSEEK_API_KEYS",
            text_api="chat_completions",
            fallback_to_freemodel=False,
            text_models=("deepseek-chat", "deepseek-reasoner"),
        ),
        ProviderConfig(
            key="anthropic", label="Anthropic Claude（仅润色）",
            base_url="https://api.anthropic.com/v1",
            env_keys=("ANTHROPIC_API_KEY", "ANTHROPIC_TOKEN", "CLAUDE_API_KEY"),
            env_list_key="ANTHROPIC_API_KEYS",
            text_api="anthropic_messages",
            fallback_to_freemodel=False,
            text_models=(
                "claude-opus-4-7", "claude-sonnet-4-6", "claude-haiku-4-5",
                "claude-opus-4-5", "claude-sonnet-4-5",
            ),
        ),
        ProviderConfig(
            key="xai", label="xAI Grok（仅润色）",
            base_url="https://api.x.ai/v1",
            env_keys=("XAI_API_KEY", "GROK_API_KEY"),
            env_list_key="XAI_API_KEYS",
            text_api="chat_completions",
            fallback_to_freemodel=False,
            text_models=(
                "grok-4-0709", "grok-4-fast-reasoning", "grok-4-fast-non-reasoning",
                "grok-3", "grok-code-fast-1",
            ),
        ),
        ProviderConfig(
            key="zhipu", label="智谱 GLM（仅润色）",
            base_url="https://open.bigmodel.cn/api/paas/v4",
            env_keys=("ZHIPU_API_KEY", "GLM_API_KEY", "BIGMODEL_API_KEY"),
            env_list_key="ZHIPU_API_KEYS",
            text_api="chat_completions",
            fallback_to_freemodel=False,
            text_models=("glm-4.5", "glm-4.5-air", "glm-4-plus", "glm-4-flash"),
        ),
        ProviderConfig(
            key="moonshot", label="月之暗面 Kimi（仅润色）",
            base_url="https://api.moonshot.cn/v1",
            env_keys=("MOONSHOT_API_KEY", "KIMI_API_KEY"),
            env_list_key="MOONSHOT_API_KEYS",
            text_api="chat_completions",
            fallback_to_freemodel=False,
            text_models=(
                "kimi-k2-turbo-preview", "moonshot-v1-128k", "moonshot-v1-32k", "moonshot-v1-8k",
            ),
        ),
        ProviderConfig(
            key="dashscope", label="阿里通义千问 Qwen（仅润色）",
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
            env_keys=("DASHSCOPE_API_KEY", "QWEN_API_KEY", "ALIYUN_API_KEY"),
            env_list_key="DASHSCOPE_API_KEYS",
            text_api="chat_completions",
            fallback_to_freemodel=False,
            text_models=(
                "qwen-max", "qwen-plus", "qwen-turbo",
                "qwen3-max", "qwen3-coder-plus",
            ),
        ),
        ProviderConfig(
            key="custom", label="自定义",
            base_url="",
            env_keys=("OPENAI_API_KEY", "API_KEY"),
            env_list_key="API_KEYS",
            text_models=("gpt-5.5", "gpt-5", "gpt-4o"),
        ),
    )
}
DEFAULT_PROVIDER = "freemodel"


# 所有 provider 的典型模型合并 —— GUI 把它当成"通用模型预设清单"用。
def _build_polish_model_presets() -> tuple[str, ...]:
    seen: list[str] = []
    for cfg in PROVIDERS.values():
        for model in cfg.text_models:
            if model and model not in seen:
                seen.append(model)
    return tuple(seen)


POLISH_MODEL_PRESETS: tuple[str, ...] = _build_polish_model_presets()


# ─────────────────────────────────────────────────────────────────────────────
# 2. 基础 I/O / 编码工具
# ─────────────────────────────────────────────────────────────────────────────

def configure_stdio() -> None:
    """让 Windows 控制台尽量按 UTF-8 输出，旧流则忽略。"""
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            try:
                stream.reconfigure(encoding="utf-8", errors="replace")
            except Exception:
                pass


def read_text_file(path: Path) -> str:
    """优先 UTF-8，遇到中文 Windows 旧文件回退 GB18030。"""
    for encoding in ("utf-8-sig", "utf-8", "gb18030"):
        try:
            return path.read_text(encoding=encoding)
        except UnicodeDecodeError:
            continue
    return path.read_text(errors="replace")


def _decode_body(response: requests.Response) -> str:
    raw = response.content
    for encoding in ("utf-8-sig", "utf-8"):
        try:
            return raw.decode(encoding)
        except UnicodeDecodeError:
            continue
    return response.text


# ─────────────────────────────────────────────────────────────────────────────
# 3. .env 读取 + 端点 / token 解析
# ─────────────────────────────────────────────────────────────────────────────

def _bootstrap_env() -> None:
    """优先用 python-dotenv，没装就用极简的内置解析器。"""
    try:
        from dotenv import load_dotenv  # type: ignore
        load_dotenv()
        return
    except ImportError:
        pass

    env_path = Path(".env")
    if not env_path.is_file():
        return
    for raw in read_text_file(env_path).splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        if not key or key in os.environ:
            continue
        os.environ[key] = value.strip().strip('"').strip("'")


def _split_token_list(value: str) -> list[str]:
    return [token.strip() for token in re.split(r"[,\s;]+", value) if token.strip()]


def _provider(provider_key: str | None) -> ProviderConfig:
    name = (provider_key or DEFAULT_PROVIDER).strip().lower()
    return PROVIDERS.get(name, PROVIDERS[DEFAULT_PROVIDER])


def resolve_endpoint(provider: str | None = None, base_url: str | None = None) -> str:
    """根据 provider/base_url 拼出 Responses 端点。"""
    candidate = (base_url or "").strip()
    if not candidate:
        candidate = _provider(provider).base_url
    if not candidate:
        candidate = os.environ.get("OPENAI_BASE_URL", "").strip()
    if not candidate:
        raise SystemExit(
            f"Provider '{provider or DEFAULT_PROVIDER}' needs a base URL. "
            "Set OPENAI_BASE_URL or pass --base-url."
        )
    return candidate.rstrip("/") + "/responses"


def _resolve_text_endpoint(
    provider: str | None, base_url: str | None, cfg: ProviderConfig,
) -> str:
    """文本/润色端点：按 cfg.text_api 拼 /responses 或 /chat/completions。"""
    candidate = (base_url or "").strip()
    if not candidate:
        candidate = cfg.base_url
    if not candidate:
        candidate = os.environ.get("OPENAI_BASE_URL", "").strip()
    if not candidate:
        raise SystemExit(
            f"Provider '{provider or cfg.key}' needs a base URL. "
            "Set OPENAI_BASE_URL or pass --base-url."
        )
    if cfg.text_api == "chat_completions":
        path = "/chat/completions"
    elif cfg.text_api == "anthropic_messages":
        path = "/messages"
    else:
        path = "/responses"
    return candidate.rstrip("/") + path


def load_tokens(provider: str | None = None) -> list[str]:
    """读取该 provider 对应的所有可用 API token。"""
    _bootstrap_env()
    cfg = _provider(provider)

    tokens: list[str] = []
    seen: set[str] = set()

    def absorb(value: str) -> None:
        if value and value not in seen:
            tokens.append(value)
            seen.add(value)

    if cfg.env_list_key:
        for token in _split_token_list(os.environ.get(cfg.env_list_key, "")):
            absorb(token)
    for key in cfg.env_keys:
        absorb(os.environ.get(key, "").strip())

    # 最后兜底：让没配 OpenAI key 的人能继续用 freemodel 的旧 .env
    if not tokens and cfg.key != "freemodel" and cfg.fallback_to_freemodel:
        fm = PROVIDERS["freemodel"]
        for token in _split_token_list(os.environ.get(fm.env_list_key, "")):
            absorb(token)
        for key in fm.env_keys:
            absorb(os.environ.get(key, "").strip())

    if not tokens:
        hint = ", ".join(cfg.env_keys[:3]) or cfg.env_list_key or "API_KEY"
        raise SystemExit(
            f"No API token found for provider '{cfg.key}'. Set {hint} in environment or .env."
        )
    return tokens


# ─────────────────────────────────────────────────────────────────────────────
# 4. 文本工具：mojibake 修复、片段去重
# ─────────────────────────────────────────────────────────────────────────────

_MOJIBAKE_MARKERS = "ÃÂâæèçåäéð�"


def _has_cjk(text: str) -> bool:
    return any(
        ("㐀" <= c <= "鿿") or ("豈" <= c <= "﫿")
        for c in text
    )


def _mojibake_quality(text: str) -> int:
    cjk = sum(1 for c in text if ("㐀" <= c <= "鿿") or ("豈" <= c <= "﫿"))
    bad = sum(text.count(marker) for marker in _MOJIBAKE_MARKERS)
    return cjk * 4 - bad * 2 - text.count("�") * 8


def repair_mojibake(value: str) -> str:
    """尝试把 cp1252/latin1 误解码回来的中文修复。"""
    if not value or not any(marker in value for marker in _MOJIBAKE_MARKERS):
        return value

    best = value
    best_quality = _mojibake_quality(value)
    for fallback in ("cp1252", "latin1"):
        try:
            candidate = value.encode(fallback).decode("utf-8")
        except UnicodeError:
            continue
        candidate_quality = _mojibake_quality(candidate)
        if candidate_quality > best_quality + 2:
            best = candidate
            best_quality = candidate_quality
    return best


_FRAGMENT_DELIMITER_RE = re.compile(r"[\n,，;；]+")
_ASCII_SLUG_RE = re.compile(r"[A-Za-z0-9 .'\-/_]+")


def _normalize_fragment(fragment: str) -> str:
    fragment = re.sub(r"\s+", " ", repair_mojibake(fragment).strip())
    fragment = fragment.strip(" \t\r\n,，;；。.!！？?:：")
    return fragment.casefold() if _ASCII_SLUG_RE.fullmatch(fragment) else fragment


def _fragment_joiner(language: str) -> str:
    return "，" if language in {"简体中文", "繁體中文", "日本語", "한국어", "保持原文"} else ", "


def dedupe_prompt_fields(
    fields: dict[str, str],
    target_language: str = "English",
    exempt: set[str] | None = None,
) -> dict[str, str]:
    """按字段顺序逐句去重，全局只保留一份。

    exempt 里的字段不参与去重 —— 原文一字不改；但它们的片段会先"占用"全局
    `seen` 集合，所以其它字段如果出现同一片段会被剃掉。这样 locked 字段不会
    丢内容，但仍能压制重复。
    """
    joiner = _fragment_joiner(target_language)
    seen: set[str] = set()
    exempt = exempt or set()
    ordered = (
        "general_prompt", "main_prompt", "person_prompt",
        "style_prompt", "scene_prompt", "avoid_prompt",
    )
    # 先吃掉 exempt 字段的所有片段（占位但不动文本），再按顺序处理其它
    for key in ordered:
        if key not in exempt:
            continue
        text = fields.get(key, "")
        for raw in (p.strip() for p in _FRAGMENT_DELIMITER_RE.split(text) if p.strip()):
            normal = _normalize_fragment(raw)
            if normal:
                seen.add(normal)
    result: dict[str, str] = {}
    for key in ordered:
        text = fields.get(key, "")
        if key in exempt:
            result[key] = text
            continue
        kept: list[str] = []
        for raw in (p.strip() for p in _FRAGMENT_DELIMITER_RE.split(text) if p.strip()):
            normal = _normalize_fragment(raw)
            if not normal or normal in seen:
                continue
            seen.add(normal)
            kept.append(raw)
        result[key] = joiner.join(kept).strip()
    for key, value in fields.items():
        result.setdefault(key, value)
    return result


# ─────────────────────────────────────────────────────────────────────────────
# 5. SSE / JSON 响应解析（合并成一遍 visitor 扫描）
# ─────────────────────────────────────────────────────────────────────────────

def _looks_like_b64_image(value: str) -> bool:
    if value.startswith("data:") and "," in value:
        value = value.split(",", 1)[1]
    value = "".join(value.split())
    if len(value) <= 500:
        return False
    return bool(re.fullmatch(r"[A-Za-z0-9+/=_-]+", value))


def _walk_json(
    obj: object,
    on_image: Callable[[str, str], None],
    on_text: Callable[[str], None],
) -> None:
    """同时收集 b64 图和 output_text 文本。"""
    if isinstance(obj, dict):
        for key, value in obj.items():
            if not isinstance(value, str):
                continue
            if key in _FINAL_B64_KEYS and _looks_like_b64_image(value):
                on_image("final", value)
            elif key in _PARTIAL_B64_KEYS and _looks_like_b64_image(value):
                on_image("partial", value)
            elif key in {"output_text", "text"} and value.strip() and not _looks_like_b64_image(value):
                on_text(repair_mojibake(value))
        for value in obj.values():
            _walk_json(value, on_image, on_text)
    elif isinstance(obj, list):
        for value in obj:
            _walk_json(value, on_image, on_text)


def _iter_payloads(body: str) -> Iterable[object]:
    """从 SSE 或单条 JSON 响应里逐个 yield JSON 对象。"""
    stripped = body.strip()
    if stripped[:1] in ("{", "["):
        try:
            yield json.loads(stripped)
            return
        except json.JSONDecodeError:
            pass

    for raw in body.splitlines():
        line = raw.strip()
        if not line.startswith("data:"):
            continue
        chunk = line[5:].strip()
        if not chunk or chunk == "[DONE]":
            continue
        try:
            yield json.loads(chunk)
        except json.JSONDecodeError:
            continue


def extract_images_from_response(body: str) -> list[str]:
    finals: list[str] = []
    partials: list[str] = []

    def on_image(kind: str, value: str) -> None:
        bucket = finals if kind == "final" else partials
        if value not in bucket:
            bucket.append(value)

    def on_text(_text: str) -> None:
        pass

    for payload in _iter_payloads(body):
        _walk_json(payload, on_image, on_text)
    return finals or partials


def extract_text_from_response(body: str) -> str:
    chunks: list[str] = []

    def on_image(_kind: str, _value: str) -> None:
        pass

    def on_text(text: str) -> None:
        if not chunks or chunks[-1] != text:
            chunks.append(text)

    for payload in _iter_payloads(body):
        _walk_json(payload, on_image, on_text)
    return repair_mojibake("".join(chunks).strip())


def extract_text_from_anthropic(body: str) -> str:
    """Anthropic /v1/messages 响应：content[*].text 拼起来。"""
    try:
        data = json.loads(body)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", body, re.S)
        if not match:
            return ""
        try:
            data = json.loads(match.group(0))
        except json.JSONDecodeError:
            return ""
    content = data.get("content") if isinstance(data, dict) else None
    if isinstance(content, str):
        return repair_mojibake(content.strip())
    if not isinstance(content, list):
        return ""
    parts: list[str] = []
    for item in content:
        if isinstance(item, dict) and isinstance(item.get("text"), str):
            parts.append(item["text"])
        elif isinstance(item, str):
            parts.append(item)
    return repair_mojibake("".join(parts).strip())


def extract_text_from_chat_completions(body: str) -> str:
    """OpenAI 兼容 chat/completions 响应：choices[0].message.content。"""
    try:
        data = json.loads(body)
    except json.JSONDecodeError:
        # 兼容某些返回 NDJSON 或带前后噪声的中转：找第一个 { 到最后一个 }
        match = re.search(r"\{.*\}", body, re.S)
        if not match:
            return ""
        try:
            data = json.loads(match.group(0))
        except json.JSONDecodeError:
            return ""
    choices = data.get("choices") if isinstance(data, dict) else None
    if not isinstance(choices, list) or not choices:
        return ""
    first = choices[0]
    if not isinstance(first, dict):
        return ""
    message = first.get("message")
    if isinstance(message, dict):
        content = message.get("content", "")
    else:
        content = first.get("text", "")
    if isinstance(content, list):
        # 某些厂商把 content 拆成 parts；拼起来
        parts = []
        for item in content:
            if isinstance(item, dict) and isinstance(item.get("text"), str):
                parts.append(item["text"])
            elif isinstance(item, str):
                parts.append(item)
        content = "".join(parts)
    return repair_mojibake(str(content).strip())


# 旧名字保留 alias（万一其它脚本引用）
parse_response_images = extract_images_from_response
parse_response_text = extract_text_from_response


def _short_body_preview(body: str, limit: int = 600) -> str:
    flattened = " ".join(body.split())
    return flattened[:limit] + ("..." if len(flattened) > limit else "")


def _now_hms() -> str:
    """`[HH:MM:SS]` 时间戳，给日志开头用。"""
    return datetime.now().strftime("[%H:%M:%S]")


def log(message: str) -> None:
    """统一日志出口：自动加时间戳 + flush。"""
    print(f"{_now_hms()} {message}", flush=True)


def _diagnose_empty_response(body: str) -> str:
    """从响应体推断为什么没拿到图，返回一行中文诊断 + 简短证据。"""
    stripped = body.strip()
    if not stripped:
        return "中转返回空响应（连接被中断或上游静默断开）"

    events_seen: list[str] = []
    last_status: str | None = None
    error_msgs: list[str] = []
    has_image_call = False
    saw_output_text = False

    for payload in _iter_payloads(body):
        if not isinstance(payload, dict):
            continue
        evt = payload.get("type")
        if isinstance(evt, str):
            events_seen.append(evt)
        response = payload.get("response") if isinstance(payload.get("response"), dict) else None
        if response:
            st = response.get("status")
            if isinstance(st, str):
                last_status = st
            err = response.get("error")
            if isinstance(err, dict):
                msg = err.get("message") or err.get("code") or json.dumps(err, ensure_ascii=False)
                if isinstance(msg, str):
                    error_msgs.append(msg)
            for item in response.get("output", []) or []:
                if isinstance(item, dict):
                    t = item.get("type")
                    if isinstance(t, str) and "image" in t:
                        has_image_call = True
                    if t == "message":
                        saw_output_text = True

    last_event = events_seen[-1] if events_seen else None

    if error_msgs:
        return f"上游报错：{error_msgs[-1][:200]}"

    if last_status == "completed" and not has_image_call:
        if saw_output_text:
            return "中转把请求当成了普通聊天，没触发 image_generation 工具（可能不支持该工具或 tool_choice 被丢弃）"
        return "上游标记完成但响应里没有图（中转剥掉了 image_generation 输出）"

    if last_status in {"failed", "cancelled", "incomplete"}:
        return f"上游状态 = {last_status}"

    if last_event == "response.created" and last_status == "in_progress":
        return "SSE 在生成开始后被中转切断（中转代理的 SSE timeout 太短，或不支持长连接）"

    if not events_seen and stripped.startswith("{"):
        return "中转返回了非流式 JSON 但没有图字段（可能 stream=true 被中转无视，或它只在错误时回 JSON）"

    if last_event:
        return f"未完成的流式响应，最后事件 = {last_event} (status={last_status or '未知'})"

    return "响应无法识别"


# ─────────────────────────────────────────────────────────────────────────────
# 6. 取消 / 重试
# ─────────────────────────────────────────────────────────────────────────────

class CancelledError(RuntimeError):
    """调用方通过 cancel_event 主动停止时抛出。"""


def _retry_after(response: requests.Response) -> float | None:
    raw = response.headers.get("Retry-After", "").strip()
    if not raw:
        return None
    try:
        return max(1.0, min(300.0, float(raw)))
    except ValueError:
        return None


def _backoff_seconds(
    attempt: int,
    status: int | None = None,
    retry_after: float | None = None,
) -> float:
    if retry_after is not None:
        return retry_after + random.uniform(0, 1.5)
    base = 5 if status in TRANSIENT_HTTP else 3
    return min(120, base * (2 ** (attempt - 1))) + random.uniform(0, 1.5)


def _sleep_or_cancel(seconds: float, cancel_event: threading.Event | None) -> bool:
    """睡眠最多 seconds 秒。中途被取消则返回 True。"""
    if cancel_event is None:
        time.sleep(seconds)
        return False
    deadline = time.monotonic() + seconds
    while True:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            return False
        if cancel_event.wait(min(0.5, remaining)):
            return True


# ─────────────────────────────────────────────────────────────────────────────
# 7. 文件名 slug
# ─────────────────────────────────────────────────────────────────────────────

_PROMPT_LABEL_RE = re.compile(
    r"^(?:subject|person details?|shooting style|scene|avoid|composition requirement"
    r"|主体|主體|人物.*?|拍摄.*?|拍攝.*?|场景|場景|避免.*?|構図要件|구도 요구사항)\s*[:：]\s*",
    re.IGNORECASE,
)


def slug_from_prompt(prompt: str, max_len: int = 16) -> str:
    """从 prompt 抽 ASCII slug 给文件名用，全 CJK 时回退到短 MD5。"""
    for raw_line in (prompt or "").splitlines():
        line = _PROMPT_LABEL_RE.sub("", raw_line.strip())
        tokens = re.findall(r"[A-Za-z0-9]+", line[:120].lower())
        if not tokens:
            continue
        slug = "-".join(tokens)[:max_len].strip("-")
        if slug:
            return slug
    digest = hashlib.md5((prompt or "").encode("utf-8")).hexdigest()[:8]
    return f"img-{digest}"


# ─────────────────────────────────────────────────────────────────────────────
# 8. 参考图本地压缩
# ─────────────────────────────────────────────────────────────────────────────

_pillow_warned = False


def _shrink_local_image(
    path: Path,
    max_edge: int,
    jpeg_quality: int,
) -> tuple[bytes, str, tuple[int, int], tuple[int, int]] | None:
    """用 Pillow 缩到 max_edge 并按透明性自动选 JPEG/PNG。没装 Pillow 返回 None。"""
    global _pillow_warned
    try:
        from PIL import Image, ImageOps  # type: ignore
    except ImportError:
        if not _pillow_warned:
            print(
                "Pillow not installed; reference images sent at original size. "
                "`pip install Pillow` to enable auto-resize."
            )
            _pillow_warned = True
        return None

    with Image.open(path) as raw:
        image = ImageOps.exif_transpose(raw)
        original_size = image.size
        w, h = original_size
        long_edge = max(w, h)
        if max_edge > 0 and long_edge > max_edge:
            scale = max_edge / float(long_edge)
            new_size = (max(1, int(round(w * scale))), max(1, int(round(h * scale))))
            image = image.resize(new_size, Image.LANCZOS)
        else:
            new_size = original_size

        has_alpha = image.mode in {"RGBA", "LA"} or (
            image.mode == "P" and "transparency" in image.info
        )
        buf = io.BytesIO()
        if has_alpha:
            if image.mode != "RGBA":
                image = image.convert("RGBA")
            image.save(buf, "PNG", optimize=True)
            return buf.getvalue(), "image/png", original_size, new_size

        if image.mode != "RGB":
            image = image.convert("RGB")
        image.save(buf, "JPEG", quality=jpeg_quality, optimize=True, progressive=True)
        return buf.getvalue(), "image/jpeg", original_size, new_size


def image_to_data_url(
    value: str,
    max_edge: int = RefImage.DEFAULT_MAX_EDGE,
    jpeg_quality: int = RefImage.JPEG_QUALITY,
) -> str:
    """把本地图编成 data: URL；URL / data: 原样透传。"""
    if value.startswith(("http://", "https://", "data:")):
        return value

    path = Path(value).expanduser()
    if not path.is_file():
        raise SystemExit(f"Reference image not found: {value}")

    raw_bytes = path.read_bytes()

    if max_edge and max_edge > 0:
        shrunk = _shrink_local_image(path, max_edge, jpeg_quality)
        if shrunk is not None:
            data, mime, original_size, new_size = shrunk
            if new_size != original_size or len(data) < len(raw_bytes):
                log(
                    f"  ref: {path.name} {original_size[0]}x{original_size[1]} -> "
                    f"{new_size[0]}x{new_size[1]} "
                    f"({len(raw_bytes):,} -> {len(data):,} bytes, {mime})"
                )
                encoded = base64.b64encode(data).decode("ascii")
                return f"data:{mime};base64,{encoded}"

    mime = mimetypes.guess_type(path.name)[0] or "image/png"
    return f"data:{mime};base64,{base64.b64encode(raw_bytes).decode('ascii')}"


# ─────────────────────────────────────────────────────────────────────────────
# 9. 尺寸校验
# ─────────────────────────────────────────────────────────────────────────────

_SIZE_RE = re.compile(r"(\d{2,5})x(\d{2,5})")


def image_size(value: str) -> str:
    """argparse type；返回规范化的 'WxH' 或 'auto'。"""
    text = value.strip().lower()
    if text == "auto":
        return text
    match = _SIZE_RE.fullmatch(text)
    if not match:
        raise argparse.ArgumentTypeError("must be auto or WIDTHxHEIGHT, e.g. 1024x1024")

    width, height = int(match.group(1)), int(match.group(2))
    pixels = width * height
    if width % 16 or height % 16:
        raise argparse.ArgumentTypeError("width and height must both be multiples of 16")
    if width > ImageLimits.MAX_EDGE or height > ImageLimits.MAX_EDGE:
        raise argparse.ArgumentTypeError(f"width and height must both be <= {ImageLimits.MAX_EDGE}")
    if pixels > ImageLimits.MAX_PIXELS:
        raise argparse.ArgumentTypeError(f"total pixels must be <= {ImageLimits.MAX_PIXELS:,}")
    if pixels < ImageLimits.MIN_PIXELS:
        raise argparse.ArgumentTypeError(f"total pixels must be >= {ImageLimits.MIN_PIXELS:,}")
    if max(width / height, height / width) > ImageLimits.MAX_RATIO:
        raise argparse.ArgumentTypeError(
            f"aspect ratio must be no more than {ImageLimits.MAX_RATIO:g}:1"
        )
    return text


# ─────────────────────────────────────────────────────────────────────────────
# 10. base64 解码 / payload 组装
# ─────────────────────────────────────────────────────────────────────────────

def decode_image_b64(value: str) -> bytes:
    cleaned = value.strip()
    if cleaned.startswith("data:") and "," in cleaned:
        cleaned = cleaned.split(",", 1)[1]
    cleaned = "".join(cleaned.split())
    cleaned += "=" * (-len(cleaned) % 4)
    try:
        return base64.b64decode(cleaned, validate=True)
    except Exception:
        return base64.urlsafe_b64decode(cleaned)


def build_image_request(
    prompt: str,
    *,
    quality: str,
    size: str,
    output_format: str,
    ref_images: list[str] | None,
    model: str,
    background: str = "auto",
    output_compression: int | None = None,
    moderation: str = "auto",
    partial_images: int = 0,
    stream: bool = True,
) -> dict:
    """组装 Responses API 请求体（含 image_generation tool）。"""
    chosen_quality = (quality or "auto").strip().lower()
    if chosen_quality not in ImageLimits.QUALITIES:
        chosen_quality = "auto"

    if ref_images:
        body_content: list[dict] = [{"type": "input_text", "text": prompt}]
        body_content.extend({"type": "input_image", "image_url": url} for url in ref_images)
        body_input: object = [{"role": "user", "content": body_content}]
    else:
        body_input = prompt

    tool: dict = {
        "type": "image_generation",
        "action": "generate",
        "quality": chosen_quality,
        "size": size,
        "output_format": output_format,
    }
    if background and background != "auto":
        tool["background"] = background
    if moderation and moderation != "auto":
        tool["moderation"] = moderation
    if output_compression is not None and output_format in {"jpeg", "webp"}:
        tool["output_compression"] = output_compression
    if partial_images:
        tool["partial_images"] = partial_images

    return {
        "model": model,
        "input": body_input,
        "tools": [tool],
        "tool_choice": "required",
        "stream": bool(stream),
    }


# 老名字保留 alias
build_payload = build_image_request


# ─────────────────────────────────────────────────────────────────────────────
# 11. 单次 image_generation 调用
# ─────────────────────────────────────────────────────────────────────────────

def _post_image_request(
    *,
    token: str,
    payload: dict,
    endpoint: str,
    retries: int,
    timeout: int,
    session: requests.Session,
    call_id: int,
    cancel_event: threading.Event | None,
) -> str:
    """对端点做一次 POST 并按 SSE/JSON 提图。失败按 _backoff_seconds 重试。"""
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
    }

    def maybe_wait(delay: float) -> None:
        if _sleep_or_cancel(delay, cancel_event):
            log(f"  [#{call_id}] 终止信号已生效（等待重试时）")
            raise CancelledError(f"#{call_id} cancelled")

    for attempt in range(1, retries + 1):
        if cancel_event is not None and cancel_event.is_set():
            log(f"  [#{call_id}] 终止信号已生效（开始下一次请求前）")
            raise CancelledError(f"#{call_id} cancelled before attempt")

        attempt_started = time.time()
        log(f"  [#{call_id}] → POST {endpoint} (尝试 {attempt}/{retries})")
        try:
            response = session.post(endpoint, json=payload, headers=headers, timeout=timeout)
        except requests.exceptions.RequestException as exc:
            elapsed = time.time() - attempt_started
            if cancel_event is not None and cancel_event.is_set():
                log(f"  [#{call_id}] 终止信号已生效（在飞 HTTP 已切断，{elapsed:.1f}s）")
                raise CancelledError(f"#{call_id} cancelled mid-flight") from exc
            log(f"  [#{call_id}] HTTP 异常（{elapsed:.1f}s 后）: {exc}")
            if attempt < retries:
                delay = _backoff_seconds(attempt)
                log(f"  [#{call_id}] {delay:.1f}s 后重试...")
                maybe_wait(delay)
            continue

        elapsed = time.time() - attempt_started
        if response.status_code in TRANSIENT_HTTP and attempt < retries:
            delay = _backoff_seconds(attempt, response.status_code, _retry_after(response))
            log(
                f"  [#{call_id}] HTTP {response.status_code} 临时错误（{elapsed:.1f}s），"
                f"{delay:.1f}s 后重试..."
            )
            maybe_wait(delay)
            continue
        if response.status_code in (401, 403):
            log(f"  [#{call_id}] HTTP {response.status_code} 鉴权失败 — 请检查 token")
        response.raise_for_status()

        body = _decode_body(response)
        images = extract_images_from_response(body)
        if images:
            best = images[0]
            log(f"  [#{call_id}] ✓ 拿到图（{elapsed:.1f}s，{len(best):,} chars b64）")
            return best

        reason = _diagnose_empty_response(body)
        log(
            f"  [#{call_id}] ✗ 无图（{elapsed:.1f}s，尝试 {attempt}/{retries}）"
            f"\n     原因: {reason}"
            f"\n     原文片段: {_short_body_preview(body, limit=300)}"
        )
        if attempt < retries:
            delay = _backoff_seconds(attempt)
            log(f"  [#{call_id}] {delay:.1f}s 后重试...")
            maybe_wait(delay)

    raise RuntimeError(f"#{call_id} {retries} 次尝试后仍失败")


# 旧 API 兼容
def generate_one(
    token: str,
    payload: dict,
    call_index: int,
    retries: int,
    timeout: int,
    session: requests.Session | None = None,
    cancel_event: threading.Event | None = None,
    endpoint: str | None = None,
) -> str:
    if endpoint is None:
        endpoint = resolve_endpoint()
    if session is not None:
        return _post_image_request(
            token=token, payload=payload, endpoint=endpoint,
            retries=retries, timeout=timeout, session=session,
            call_id=call_index, cancel_event=cancel_event,
        )
    with requests.Session() as owned:
        return _post_image_request(
            token=token, payload=payload, endpoint=endpoint,
            retries=retries, timeout=timeout, session=owned,
            call_id=call_index, cancel_event=cancel_event,
        )


# ─────────────────────────────────────────────────────────────────────────────
# 12. 主入口 generate（多 token 分摊 + 并发 + 取消）
# ─────────────────────────────────────────────────────────────────────────────

def generate(
    prompt: str,
    n: int = 1,
    quality: str = "auto",
    size: str = "1024x1024",
    output_format: str = "png",
    ref_images: list[str] | None = None,
    out_dir: str = "output",
    model: str = "gpt-5.5",
    retries: int = 5,
    timeout: int = 180,
    action: str = "generate",
    background: str = "auto",
    output_compression: int | None = None,
    moderation: str = "auto",
    partial_images: int = 0,
    max_concurrency: int = 1,
    cancel_event: threading.Event | None = None,
    session: requests.Session | None = None,
    provider: str | None = None,
    base_url: str | None = None,
    stream: bool = True,
) -> list[str]:
    """生成 n 张图，多 token 自动轮询，可取消，返回所有成功落盘的路径。"""
    endpoint = resolve_endpoint(provider, base_url)
    tokens = load_tokens(provider)
    payload = build_image_request(
        prompt,
        quality=quality, size=size, output_format=output_format,
        ref_images=ref_images, model=model,
        background=background, output_compression=output_compression,
        moderation=moderation, partial_images=partial_images, stream=stream,
    )

    # 把 n 张图按 round-robin 分给每个 token
    per_token = [n // len(tokens)] * len(tokens)
    for i in range(n % len(tokens)):
        per_token[i] += 1

    workload: list[tuple[str, list[int]]] = []
    cursor = 1
    for token, count in zip(tokens, per_token):
        if count <= 0:
            continue
        workload.append((token, list(range(cursor, cursor + count))))
        cursor += count

    if not workload:
        log("没有任务可做。")
        return []

    requested = len(workload) if max_concurrency <= 0 else max_concurrency
    workers = max(1, min(requested, len(workload)))

    log(
        f"开始生成 {n} 张 | {size} {quality} | {len(tokens)} 个 token | "
        f"stream {'on' if stream else 'off'} | 并发 {workers} | endpoint {endpoint}"
    )

    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    slug = slug_from_prompt(prompt)
    ext = output_format if output_format in {"jpeg", "webp"} else "png"

    started = time.time()
    saved: list[str] = []
    save_lock = threading.Lock()

    def save_image(call_id: int, b64_image: str) -> None:
        """先写 .part 再 rename，做到原子落盘——崩溃 / 终止永远不会留半张图。"""
        final = out_path / f"{stamp}_{slug}_{call_id}.{ext}"
        tmp = final.with_suffix(final.suffix + ".part")
        tmp.write_bytes(decode_image_b64(b64_image))
        tmp.replace(final)
        with save_lock:
            saved.append(str(final))
            progress = f"[{len(saved)}/{n}]"
        log(f"  ✓ 已落盘 {progress}: {final} ({final.stat().st_size:,} bytes)")

    owned_session = session is None
    session = session or requests.Session()

    def is_cancelled() -> bool:
        return cancel_event is not None and cancel_event.is_set()

    def consume_queue(token: str, indices: list[int]) -> None:
        for idx in indices:
            if is_cancelled():
                with save_lock:
                    kept = len(saved)
                log(f"  [#{idx}] 收到终止信号，跳过此任务（已保留 {kept}/{n}）")
                return
            try:
                b64_image = _post_image_request(
                    token=token, payload=payload, endpoint=endpoint,
                    retries=retries, timeout=timeout, session=session,
                    call_id=idx, cancel_event=cancel_event,
                )
            except CancelledError:
                with save_lock:
                    kept = len(saved)
                log(f"  [#{idx}] 已中断（已保留 {kept}/{n}）")
                return
            except Exception as exc:
                log(f"  [#{idx}] 失败: {exc}")
                continue
            try:
                save_image(idx, b64_image)
            except Exception as exc:
                log(f"  [#{idx}] 落盘失败: {exc}")

    try:
        if len(workload) == 1:
            consume_queue(*workload[0])
        else:
            with ThreadPoolExecutor(max_workers=workers) as pool:
                futures = [pool.submit(consume_queue, t, i) for t, i in workload]
                for fut in as_completed(futures):
                    if is_cancelled():
                        for f in futures:
                            f.cancel()
                    try:
                        fut.result()
                    except CancelledError:
                        pass
    finally:
        if owned_session:
            try:
                session.close()
            except Exception:
                pass

    elapsed = time.time() - started
    if is_cancelled():
        log(f"已终止：保留了 {len(saved)}/{n} 张已落盘图像，用时 {elapsed:.1f}s")
    else:
        log(f"完成：{len(saved)}/{n} 张图像，用时 {elapsed:.1f}s")
    return saved


# ─────────────────────────────────────────────────────────────────────────────
# 13. 调一次 gpt-5.5 文本（给润色用）
# ─────────────────────────────────────────────────────────────────────────────

def call_text_model(
    request_text: str,
    model: str = "gpt-5.5",
    retries: int = 5,
    timeout: int = 300,
    provider: str | None = None,
    base_url: str | None = None,
    system_prompt: str | None = None,
) -> str:
    cfg = _provider(provider)
    endpoint = _resolve_text_endpoint(provider, base_url, cfg)

    if cfg.text_api == "chat_completions":
        messages: list[dict[str, str]] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": request_text})
        payload: dict[str, object] = {
            "model": model, "messages": messages, "stream": False,
        }
    elif cfg.text_api == "anthropic_messages":
        # Anthropic 把 system 放顶层；max_tokens 必填。
        payload = {
            "model": model,
            "max_tokens": 4096,
            "messages": [{"role": "user", "content": request_text}],
            "stream": False,
        }
        if system_prompt:
            payload["system"] = system_prompt
    else:
        if system_prompt:
            body_input: object = [
                {"role": "system", "content": [{"type": "input_text", "text": system_prompt}]},
                {"role": "user",   "content": [{"type": "input_text", "text": request_text}]},
            ]
        else:
            body_input = request_text
        payload = {"model": model, "input": body_input, "stream": False}

    tokens = load_tokens(provider)
    rotation = tokens[:]
    random.shuffle(rotation)

    for attempt in range(1, retries + 1):
        token = rotation[(attempt - 1) % len(rotation)]
        if cfg.text_api == "anthropic_messages":
            headers = {
                "x-api-key": token,
                "anthropic-version": "2023-06-01",
                "Content-Type": "application/json",
            }
        else:
            headers = {
                "Authorization": f"Bearer {token}",
                "Content-Type": "application/json",
            }
        try:
            response = requests.post(endpoint, json=payload, headers=headers, timeout=timeout)
        except requests.exceptions.RequestException as exc:
            if attempt >= retries:
                raise
            delay = _backoff_seconds(attempt)
            log(f"文本润色 {exc}；{delay:.1f}s 后重试...")
            time.sleep(delay)
            continue

        if response.status_code in TRANSIENT_HTTP and attempt < retries:
            delay = _backoff_seconds(attempt, response.status_code, _retry_after(response))
            log(
                f"文本润色 HTTP {response.status_code} "
                f"(尝试 {attempt}/{retries})；{delay:.1f}s 后重试..."
            )
            time.sleep(delay)
            continue
        response.raise_for_status()

        body = _decode_body(response)
        if cfg.text_api == "chat_completions":
            text = extract_text_from_chat_completions(body)
        elif cfg.text_api == "anthropic_messages":
            text = extract_text_from_anthropic(body)
        else:
            text = extract_text_from_response(body)
        if text:
            return text

        if attempt < retries:
            delay = _backoff_seconds(attempt)
            log(f"文本润色无返回；{delay:.1f}s 后重试...")
            time.sleep(delay)

    raise RuntimeError("文本润色请求多次失败")


# ─────────────────────────────────────────────────────────────────────────────
# 14. 润色：摄影词汇库 + few-shot 反 AI 系统提示词
# ─────────────────────────────────────────────────────────────────────────────

# 给模型当"心法"的摄影领域词汇库
_PHOTO_DICTIONARY = """\
camera_bodies: Canon EOS R6, Canon EOS R5, Sony A7 IV, Sony A7C II, Fujifilm X-T5,
    Fujifilm GFX100S, Leica Q3, Nikon Z6 II; (vintage) Canon AE-1, Pentax K1000,
    Contax T2, Yashica T4, Olympus OM-1, Hasselblad 500C; (digital nostalgia) early
    2000s Sony Cyber-shot CCD, Casio EX-Z series, Canon PowerShot G-series.
lenses: 35mm f/1.4 or f/1.8, 50mm f/1.2/1.4/1.8, 85mm f/1.4/1.8, 24-70 f/2.8,
    70-200 f/2.8, 28mm pancake; vintage Helios 44-2, Pentax SMC 50mm.
films: Kodak Portra 400, Kodak Portra 800, Kodak Gold 200, Cinestill 800T,
    Fujifilm Pro 400H, Fuji Superia 400, Ilford HP5+ B&W, Tri-X 400 B&W.
lighting_setups: Rembrandt key from camera-left, butterfly key from above,
    split light from window, broad-side fill, short-side fill, hair/rim light from
    behind, low-key with one bare bulb, high-key softbox bank, on-camera direct
    flash with cool ambient mix, off-camera strobe + umbrella, golden hour back-rim,
    overcast diffused, blue-hour neon mix, single window soft light from camera-right.
exposure_hints: ISO 100-200 daylight, ISO 400-800 dim interior, ISO 1600-3200 night,
    1/200s sync for flash, 1/60s window light, 1/15s slight motion blur.
grading_words: warm cream highlights, cool blue shadows, muted teal-orange,
    pastel film fade, contrasty B&W, faded magenta-leaning vintage, neutral
    editorial, kodak gold warm.
realism_hooks: natural pores at close range, faint freckles, asymmetric eye creases,
    micro flyaway hair strands, very mild lens distortion, slight motion blur on
    one finger, gentle bokeh balls (not perfectly round), real fabric weave,
    visible seam stitches, scuffed shoes, candid eye-line, off-center framing.
"""

# 反 AI 感的硬规则
_ANTI_AI_RULES = """\
Forbidden positive words (delete or convert to concrete visuals):
    perfect, stunning, flawless, masterpiece, amazing, gorgeous, breathtaking,
    "8K", "ultra HD", "ultra-detailed", "hyperrealistic", "highly detailed",
    cinematic, dreamy, magical, ethereal, "best quality", "trending on artstation",
    "octane render", "unreal engine", "photorealistic" (it ironically reads as AI).
Forbidden positive structural moves:
    perfectly symmetrical face, glass-like skin, porcelain skin, doll-like features,
    sparkling eyes, "soulful gaze" without anchor, three-paragraph praise.
Always push into avoid_prompt instead:
    "no plastic skin, no waxy highlights, no oversharpening, no over-smoothing,
    no excessive HDR, no fake film LUT, no AI artifacts, no extra fingers,
    no warped hands, no merged limbs, no asymmetric pupils, no text, no watermark,
    no logo, no signature, no border, no collage."
"""

# 翻译/润色都基于这个 system prompt，按 intensity 微调
_BASE_SYSTEM_PROMPT = f"""\
YOUR ROLE: senior gpt-image-2 prompt-polish expert (with the brain of an
editorial portrait photographer). Every JSON field you output is sent
DIRECTLY into a gpt-image-2 image_generation tool call — there is no
human review between your output and the image model. Treat every field
as a final, executable instruction to the image model, not as
description for a reader.

Primary objective: take Chinese (or messy mixed-language) prompt fields
and produce clean English prompt fields that read like a real-shoot
brief. The resulting image must look like a real photograph from a real
camera, NOT an AI generation.

Faithfulness to user intent (HIGH PRIORITY):
- When a PRESET BASE / PRESET DIRECTIVE is present below (wardrobe /
  scene / style / framing / pose), it is the user's deliberate creative
  choice. Your job is to REALIZE that direction in concrete photographic
  language, not to soften, hedge, or replace it with a tamer default.
- Editorial register only. Even bold poses / coverage stay phrased as
  adult fashion editorial work, never as crude or pornographic prose.

Domain vocabulary you should rely on instead of vague adjectives:
{_PHOTO_DICTIONARY}

{_ANTI_AI_RULES}

Field responsibility map (mutually exclusive — never repeat info across fields):
- general_prompt: one short overall concept or assignment brief (≤1 line)
- main_prompt: subject identity anchor (age, ethnicity, body type, who they are)
- person_prompt: face / hair / makeup / clothing / pose / expression specifics
- style_prompt: real camera body + lens + ISO + lighting setup + grading
- scene_prompt: location + props + light source direction + background context
- avoid_prompt: only the negatives; never put positive content here

Subject safety:
- Adult subjects only. If age is ambiguous, write the subject as an adult.
- If a brief mixes youthful age cues with sexualized styling, rewrite it as a
  clearly adult fashion/editorial concept rather than explicit content.

Two examples of GOOD polish (you must imitate this density and concreteness):

Example A
INPUT:
{{"general_prompt": "美少女在咖啡店", "main_prompt": "完美的19岁女孩",
  "person_prompt": "大眼睛、白皙皮肤、完美面孔、迷人微笑",
  "style_prompt": "8K超高清、完美光线", "scene_prompt": "舒适的咖啡店",
  "avoid_prompt": ""}}
OUTPUT:
{{"general_prompt": "candid afternoon coffee-shop portrait",
  "main_prompt": "early-20s Korean woman sitting by a window seat",
  "person_prompt": "slightly upturned almond eyes with subtle monolid crease, light-warm skin with visible pores at close range, faint blush, casual no-makeup-makeup look, mid-length wavy dark brown hair tucked behind one ear, beige rib-knit cardigan over a plain white tee, both hands wrapped around a ceramic latte cup, faint warm half-smile, slight side glance toward camera",
  "style_prompt": "Canon EOS R6 with RF 50mm f/1.8 STM, ISO 800, 1/125s, single soft window light from camera-right with a faint white reflector camera-left, mild film-like grain, warm cream highlights, slightly cool shadow rolloff",
  "scene_prompt": "small independent cafe interior in mid-afternoon, walnut wood counter blurred behind her, ceramic mugs and an open paper notebook on the table, a single brass pendant lamp out of focus in the upper-left corner, no signage, no readable text on cups",
  "avoid_prompt": "no plastic skin, no waxy highlights, no oversharpening, no over-smoothed face, no doll symmetry, no glowing eyes, no HDR halo, no fake film LUT, no AI artifacts, no extra fingers, no merged hands, no text, no watermark, no logo, no border, no collage"}}

Example B
INPUT:
{{"general_prompt": "夜晚街头直闪 千禧年CCD感",
  "main_prompt": "亚洲女孩 朋克cool girl",
  "person_prompt": "短发 黑色机车皮夹克 暗色口红",
  "style_prompt": "CCD质感 强直闪 颗粒",
  "scene_prompt": "夜晚街头", "avoid_prompt": ""}}
OUTPUT:
{{"general_prompt": "Y2K-era CCD digital snapshot, late-night street",
  "main_prompt": "early-20s East Asian woman with a cool-edged street look",
  "person_prompt": "chin-length straight black hair with blunt micro-fringe, faint smoky liner with deep berry matte lipstick, fitted black leather biker jacket with chrome zippers, plain dark tee underneath, a thin silver chain, one hand half in the jacket pocket, neutral closed-mouth expression, looking straight into the lens",
  "style_prompt": "early-2000s Sony Cyber-shot CCD sensor look, on-camera direct flash dominating, harsh frontal illumination, cool ambient streetlight rolloff behind her, heavy luminance noise, slight motion blur on the jacket edge, low dynamic range with crushed shadows, slight chromatic fringing on high-contrast edges, mild blown highlights on cheekbones",
  "scene_prompt": "narrow city sidewalk at night, blurred neon shop signs behind her with unreadable kanji-like letters, wet asphalt reflections, a couple of out-of-focus pedestrians far in the background, light drizzle in the air catching the flash",
  "avoid_prompt": "no clean studio look, no glossy retouch, no perfect skin, no smooth bokeh, no modern mirrorless sharpness, no HDR sky, no readable text, no watermark, no logo, no extra fingers, no warped jacket geometry, no merged limbs"}}

OUTPUT FORMAT
- Return ONE valid JSON object.
- Keys must exactly match the input keys.
- Values are plain strings (no nested objects, no arrays).
- No markdown fence, no commentary, no key not in the input.
"""

# intensity 块只追加在 base system prompt 之后
_INTENSITY_NOTES = {
    "conservative": (
        "INTENSITY=CONSERVATIVE. Keep every concrete subject/wardrobe/scene element "
        "as given. Only translate accurately, fix grammar, remove duplication, swap "
        "vague adjectives for concrete photography vocabulary. Do not invent new "
        "outfits, new poses, or new locations."
    ),
    "open": (
        "INTENSITY=OPEN. Subject identity, garment category, scene type and mood "
        "stay; every other detail you SHOULD make more concrete (fabric, finish, "
        "lighting direction, lens, framing) and push register toward confident "
        "editorial rather than safe everyday defaults — when the source / presets "
        "already lean bold, follow that lean, don't sand it down. You may merge "
        "synonymous fragments. Do not change what the image depicts at a high level."
    ),
    "gacha": (
        "INTENSITY=GACHA. Identity (age/ethnicity/body/face) stays the same person. "
        "Wardrobe, makeup, hair, pose, expression, scene, lighting, time-of-day, "
        "framing — all open. Commit to ONE coherent new look; do not enumerate "
        "alternatives or hedge with 'or'. At least two of {wardrobe, pose, "
        "lighting, framing} must be a deliberate non-default editorial choice, "
        "not the safest reading of the brief."
    ),
    "aggressive": (
        "INTENSITY=AGGRESSIVE. Only thing fixed is identity. Reinvent wardrobe, "
        "makeup, hair, pose, scene, style, lighting boldly into a new concept "
        "shoot of the same person. Banned defaults: plain neutral standing or "
        "sitting, closed-mouth neutral expression, flat overhead daylight, blank "
        "seamless backdrop, eye-level full-body framing. Every aspect must read "
        "as a deliberate editorial decision. Still ONE coherent image, not a "
        "moodboard. Adult subject."
    ),
    "unhinged": (
        "INTENSITY=UNHINGED (max editorial commitment). Identity stays. Every "
        "other aspect — wardrobe, coverage, posture, expression, framing, scene, "
        "lighting, color grade — MUST be pushed to the strongest editorial "
        "reading the brief and active presets can support; safe / neutral "
        "defaults are FORBIDDEN. Active presets always win on direction (see "
        "PRESET PRIORITY block); within whatever direction the presets set, you "
        "go to the bold edge of it. Rules:\n"
        "- Coverage: when no wardrobe preset has fixed it, lean toward the "
        "boldest reading the source allows — deep necklines, exposed back / "
        "midriff / thigh / shoulder, sheer / cling / wet / lace fabric, high "
        "slits — phrased as adult fashion-editorial work, never as crude or "
        "anatomy-diagram prose.\n"
        "- Posture & expression: theatrical, deliberate, confident-and-aware; "
        "no demure, no closed body, no neutral resting face. Body language "
        "must read as a decision, not a default.\n"
        "- Lighting & framing: dramatic — directional / hard / coloured / "
        "low-key / rim / backlit / window-shaft / practical light source. "
        "Unusual angle (low / overhead / dutch / close-cropped) or close "
        "framing strongly preferred over flat eye-level full-body.\n"
        "- Scene: committed and atmospheric, with concrete textures and props; "
        "no blank seamless, no undescribed rooms, no 'a studio'.\n"
        "- At minimum FIVE of {wardrobe, coverage, pose, expression, framing, "
        "lens choice, lighting, color grade, scene mood} must be a deliberate "
        "non-default decision.\n"
        "Stay editorial / magazine-cover register (no anatomy descriptors, no "
        "pornographic prose, no underage cues — see project rules); Adult "
        "subject. Still ONE coherent image, not a moodboard."
    ),
}

_LOCK_NOTES = {
    "identity": "IDENTITY: never alter age, ethnicity, body type, face features.",
    "wardrobe": "WARDROBE LOCKED: keep exact clothing items, colors, accessories.",
    "makeup":   "MAKEUP LOCKED: keep exact makeup look and intensity.",
    "hair":     "HAIR LOCKED: keep exact hair color, length, style.",
    "pose":     "POSE LOCKED: keep exact pose, body framing, expression.",
    "scene":    "SCENE LOCKED: keep exact location, environment, time of day.",
    "style":    "STYLE LOCKED: keep exact lighting, camera, lens, color grade.",
}


# 润色"底子" — 影响润色 AI 选择服饰倾向的方向。
# 设计原则：只描述 silhouette（身形）/ coverage（暴露度）/ posture（姿态）/ expression（神情），
# 不指定具体单品（穿什么留给用户自己写）。
WARDROBE_PRESETS: dict[str, str] = {
    "无": "",  # 不注入方向，按原系统提示词跑

    "不改动": (
        "WARDROBE PRESERVE (override intensity if it conflicts): keep the existing "
        "wardrobe / clothing / accessories references in person_prompt unchanged — "
        "do not invent new garments, do not swap colors, fabrics, cuts, or coverage, "
        "and do not delete listed wardrobe items. You may still translate, fix "
        "grammar, and tighten the language around them. Refinement of non-wardrobe "
        "aspects of person_prompt (face, hair, makeup, expression, pose) is still "
        "allowed unless those are locked elsewhere."
    ),

    "保守": (
        "STYLING BASE (apply to person_prompt; do NOT name specific garment items — "
        "leave the actual clothing for the user to fill in):\n"
        "- Silhouette: composed and contained, no emphasis on body curves; "
        "structured rather than body-conscious.\n"
        "- Coverage: high — necklines closed, sleeves long, hems past the knee.\n"
        "- Posture: contained — hands close to body, weight settled, no theatrical poses.\n"
        "- Expression: calm, focused, professional, slightly reserved."
    ),

    "日常": (
        "STYLING BASE (apply to person_prompt; do NOT name specific garment items — "
        "leave the actual clothing for the user to fill in):\n"
        "- Silhouette: comfortable natural fit, neither tight nor baggy.\n"
        "- Coverage: ordinary — regular daily-wear baseline, nothing notable in either direction.\n"
        "- Posture: relaxed and unposed — slight slouch fine, mid-action fine.\n"
        "- Expression: at ease, candid, real-person energy."
    ),

    "时尚编辑": (
        "STYLING BASE (apply to person_prompt; do NOT name specific garment items — "
        "leave the actual clothing for the user to fill in):\n"
        "- Silhouette: dramatic and architectural — structured shoulders, narrow waist, "
        "asymmetric proportions, intentional shape.\n"
        "- Coverage: bold but stylized — can be very covered or quite exposed; "
        "choice is editorial, not casual.\n"
        "- Posture: posed and high-fashion — distant, slightly aloof, theatrical lines.\n"
        "- Expression: cool, magazine-cover, unbothered."
    ),

    "大胆露肤": (
        "STYLING BASE (apply to person_prompt; do NOT name specific garment items — "
        "leave the actual clothing for the user to fill in):\n"
        "- Silhouette: full curvy figure with defined contours — generous bust and hips, "
        "narrow waist, ripe yet architecturally shaped, not formless.\n"
        "- Coverage: low and revealing — deep necklines, open shoulders, exposed back, "
        "exposed décolletage, exposed midriff or thigh wherever the figure can be shown.\n"
        "- Posture: confident and open — chest forward, shoulders back, deliberately "
        "displaying the figure rather than concealing it.\n"
        "- Expression: assured, alluring, unembarrassed; comfortable being looked at, "
        "a woman who is aware of what she is showing and chooses to show it.\n"
        "- Adult subject."
    ),

    "泳装": (
        "STYLING BASE (apply to person_prompt; do NOT name specific garment items — "
        "leave the actual clothing for the user to fill in):\n"
        "- Silhouette: swim-ready figure, exposed skin baseline.\n"
        "- Coverage: swimwear range — torso and limbs largely exposed; do not pick a specific cut.\n"
        "- Posture: outdoor / leisure — leaning, walking on sand, half-turning, relaxed.\n"
        "- Expression: sun-lit ease, confident in skin, daytime energy."
    ),

    "贴身运动": (
        "STYLING BASE (apply to person_prompt; do NOT name specific garment items — "
        "leave the actual clothing for the user to fill in):\n"
        "- Silhouette: athletic — muscle definition visible, body-conscious lines from performance fabric.\n"
        "- Coverage: athletic norm — torso covered or sports-bra baseline, limbs typically bare.\n"
        "- Posture: mid-action or post-action — stretching, water-bottle-in-hand, walking off a court.\n"
        "- Expression: focused, alert, kinetic."
    ),
}
WARDROBE_PRESET_KEYS = tuple(WARDROBE_PRESETS.keys())
DEFAULT_WARDROBE_PRESET = "无"


# 场景底子：影响 scene_prompt。和 wardrobe 同样的设计 — 只给方向，不锁单品。
SCENE_PRESETS: dict[str, str] = {
    "无": "",

    "不改动": (
        "SCENE PRESERVE (override intensity if it conflicts): keep the existing "
        "scene_prompt references — location, props, background elements, time of "
        "day — byte-for-byte in meaning. Do not relocate the shoot, do not invent "
        "new props, do not change indoor↔outdoor or day↔night. You may translate "
        "and tighten the language but the place stays the same place."
    ),

    "室内日常": (
        "SCENE BASE (apply to scene_prompt; do NOT pick a single specific venue — "
        "give direction only):\n"
        "- Type: ordinary indoor everyday space (apartment, café, studio room, "
        "small shop, bedroom, kitchen — pick what fits the brief).\n"
        "- Light source: soft window light dominating, optional weak indoor lamp.\n"
        "- Background props: a few real-life objects in soft focus, not styled.\n"
        "- Mood: lived-in, candid, low-key, not staged."
    ),

    "街头都市": (
        "SCENE BASE (apply to scene_prompt; do NOT pick a single specific street — "
        "give direction only):\n"
        "- Type: urban sidewalk, alley, crosswalk, shopfront, or transit station.\n"
        "- Light source: mixed daylight + signage, or night neon + streetlamps.\n"
        "- Background: out-of-focus pedestrians, vehicles, signage with unreadable "
        "text, wet or dry pavement depending on mood.\n"
        "- Mood: candid documentary, sense of place, real city air."
    ),

    "自然户外": (
        "SCENE BASE (apply to scene_prompt; do NOT pick a single specific landmark — "
        "give direction only):\n"
        "- Type: outdoor natural setting — park, garden, beach, forest, riverside, "
        "mountainside, open field.\n"
        "- Light source: natural sun (golden hour, soft overcast, or harsh midday — "
        "pick what fits the brief).\n"
        "- Background: foliage, sky, water, terrain — organic and atmospheric.\n"
        "- Mood: open, airy, present in the environment."
    ),

    "影棚极简": (
        "SCENE BASE (apply to scene_prompt; keep the environment minimal):\n"
        "- Type: studio shoot — seamless paper or fabric backdrop, single solid tone.\n"
        "- Light source: controlled strobe or softbox setup (specified in style_prompt).\n"
        "- Background: clean, no props or only one deliberate styling prop.\n"
        "- Mood: editorial, controlled, focus entirely on subject."
    ),

    "夜店霓虹": (
        "SCENE BASE (apply to scene_prompt; do NOT name a specific venue — give "
        "direction only):\n"
        "- Type: nightclub, neon-lit bar, late-night arcade, karaoke room, "
        "neon-soaked alley.\n"
        "- Light source: saturated colored neons (magenta / cyan / electric blue / "
        "red), with reflective surfaces catching the color.\n"
        "- Background: blurred crowd, glowing signage with unreadable text, glass "
        "and wet reflections.\n"
        "- Mood: high-saturation nightlife, charged atmosphere."
    ),

    "复古居所": (
        "SCENE BASE (apply to scene_prompt; do NOT pick a specific decade brand — "
        "give direction only):\n"
        "- Type: vintage / retro home interior (mid-century, 90s, Y2K — pick what "
        "fits the brief).\n"
        "- Light source: warm tungsten lamps + soft window, slight color cast.\n"
        "- Background: period-appropriate furniture, CRT TV / cassette player / "
        "rotary phone / patterned wallpaper textures — period objects in soft focus.\n"
        "- Mood: nostalgic, lived-in, era-coded but not costume."
    ),
}
SCENE_PRESET_KEYS = tuple(SCENE_PRESETS.keys())
DEFAULT_SCENE_PRESET = "无"


# 拍摄风格底子：影响 style_prompt（相机/胶片/光线特性/调色取向）。不锁具体机身。
SHOOTING_STYLE_PRESETS: dict[str, str] = {
    "无": "",

    "不改动": (
        "STYLE PRESERVE (override intensity if it conflicts): keep the existing "
        "style_prompt references — camera era, film stock, lens choice, lighting "
        "direction, color grading — unchanged. Do not swap a film look for a "
        "digital look or vice versa, do not change ISO regime, do not re-grade. "
        "You may still tighten the language and add concrete photographic anchors "
        "that are consistent with what is already there."
    ),

    "现代数码自然光": (
        "STYLE BASE (apply to style_prompt; do not pick a single body — give "
        "direction only):\n"
        "- Camera era: modern full-frame mirrorless (Canon R-series / Sony A7 / "
        "Nikon Z / Leica Q range).\n"
        "- Lens: 35mm or 50mm or 85mm fast prime, shallow depth of field.\n"
        "- Exposure: ISO 100–400 daylight, 1/200–1/500s.\n"
        "- Lighting: natural window or ambient daylight, no flash.\n"
        "- Grading: clean neutral with mild warm rolloff, no heavy LUT."
    ),

    "35mm 胶片": (
        "STYLE BASE (apply to style_prompt; do not pick a single body — give "
        "direction only):\n"
        "- Camera era: 35mm film SLR or rangefinder (Canon AE-1 / Pentax K1000 / "
        "Contax T2 / Leica M).\n"
        "- Lens: 35mm f/1.4 or 50mm f/1.8 prime.\n"
        "- Film: Kodak Portra 400 / Portra 800 / Gold 200 / Cinestill 800T / "
        "Fuji 400H — pick what fits the mood.\n"
        "- Lighting: available light, mild grain, gentle highlight rolloff.\n"
        "- Grading: warm cream highlights, cool blue shadows, slight pastel fade."
    ),

    "CCD 千禧直闪": (
        "STYLE BASE (apply to style_prompt):\n"
        "- Camera era: early-2000s Sony Cyber-shot CCD / Canon PowerShot G / "
        "Casio EX-Z compact digital.\n"
        "- Lens: built-in compact zoom, slightly soft.\n"
        "- Lighting: on-camera direct flash dominating, harsh frontal illumination, "
        "cool ambient rolloff behind subject.\n"
        "- Sensor character: heavy luminance noise in shadows, low dynamic range, "
        "slight chromatic fringing on high-contrast edges, mild blown highlights "
        "on cheekbones and forehead.\n"
        "- Grading: snapshot Y2K look, no post-grade polish."
    ),

    "黑白纪实": (
        "STYLE BASE (apply to style_prompt):\n"
        "- Camera era: 35mm film B&W (Ilford HP5+ 400 / Kodak Tri-X 400) or "
        "modern digital converted to B&W.\n"
        "- Lens: 35mm or 50mm fast prime.\n"
        "- Lighting: available light, strong tonal contrast, deep blacks, "
        "controlled highlights.\n"
        "- Grading: contrasty monochrome, visible film grain, no color cast.\n"
        "- Vibe: documentary, observational, candid moment."
    ),

    "影棚商业大片": (
        "STYLE BASE (apply to style_prompt; do not pick a single body — give "
        "direction only):\n"
        "- Camera era: medium-format digital (Fujifilm GFX / Hasselblad H / "
        "Phase One) or modern full-frame for studio.\n"
        "- Lens: 85mm or 110mm portrait prime.\n"
        "- Lighting: studio strobe with softbox key + fill + rim + hair light, "
        "tightly controlled.\n"
        "- Exposure: ISO 100, f/8–f/11, 1/200s sync.\n"
        "- Grading: clean editorial, neutral skin tones, controlled contrast, "
        "magazine-cover finish (without cliché 'cinematic')."
    ),

    "夜晚直闪": (
        "STYLE BASE (apply to style_prompt):\n"
        "- Camera era: any (film or digital), but treated as snapshot.\n"
        "- Lens: 28mm or 35mm.\n"
        "- Lighting: on-camera direct flash as primary, cold ambient streetlight or "
        "neon in background, slow shutter mix possible (1/30s) for ambient bleed.\n"
        "- Exposure: ISO 800–1600 for ambient pickup.\n"
        "- Grading: high-contrast, crushed shadows, slight motion smear on edges, "
        "color shift between flash-lit subject and ambient background."
    ),
}
SHOOTING_STYLE_PRESET_KEYS = tuple(SHOOTING_STYLE_PRESETS.keys())
DEFAULT_SHOOTING_STYLE_PRESET = "无"


# 景别 / 构图底子：影响 style_prompt 的取景方向，不指定具体焦距。
FRAMING_PRESETS: dict[str, str] = {
    "无": "",

    "不改动": (
        "FRAMING PRESERVE (override intensity if it conflicts): keep any existing "
        "framing / shot-distance / composition language in the prompt unchanged. "
        "Do not switch close-up to wide or vice versa."
    ),

    "大头特写": (
        "FRAMING: extreme close-up — face fills the frame, eyes and lips dominate "
        "composition, environment is irrelevant or fully out of focus. Shallow "
        "depth of field on the iris plane."
    ),

    "特写": (
        "FRAMING: close-up portrait — head and shoulders only, subject fills the "
        "vertical frame, background heavily blurred and secondary. Crop just below "
        "the collarbone."
    ),

    "半身": (
        "FRAMING: half-body — waist-up framing, subject occupies the main vertical "
        "space, environment readable but clearly secondary, modest depth of field."
    ),

    "全身": (
        "FRAMING: full-body — entire figure visible head-to-toe with slight headroom "
        "and slight foot room, environment context clearly visible around the subject."
    ),

    "远景": (
        "FRAMING: wide environmental shot — subject is small in the frame, scene "
        "and atmosphere dominate, strong sense of place, subject as one element "
        "of the larger composition."
    ),

    "低角度": (
        "FRAMING: low-angle composition — camera below subject's chin or waist line, "
        "looking up, subject appears taller and more commanding, sky / ceiling "
        "visible in background."
    ),

    "高角度": (
        "FRAMING: high-angle composition — camera above subject's eye line, looking "
        "down at the subject, ground / floor visible, sense of intimacy or scale."
    ),
}
FRAMING_PRESET_KEYS = tuple(FRAMING_PRESETS.keys())
DEFAULT_FRAMING_PRESET = "无"


# 姿态 / 动作底子：影响 person_prompt 的肢体方向。
POSE_PRESETS: dict[str, str] = {
    "无": "",

    "不改动": (
        "POSE PRESERVE (override intensity if it conflicts): keep the existing "
        "pose, gesture, expression, and gaze direction language in person_prompt "
        "unchanged. Do not switch standing to sitting, do not invent new gestures, "
        "do not change where the subject is looking."
    ),

    "自然站立": (
        "POSE: relaxed standing — balanced weight on both legs or one, arms hanging "
        "naturally or one hand lightly resting (pocket / bag strap / hair / surface). "
        "Neutral attentive expression, gaze can be toward camera or natural off-axis."
    ),

    "走动中": (
        "POSE: candid mid-stride — one foot leading, slight motion in clothing and "
        "hair, weight on the back leg, gaze natural and not posed. Slight motion "
        "feeling allowed (motion blur on a finger / hair tip is fine)."
    ),

    "坐姿": (
        "POSE: seated naturally — on a chair, step, sofa, bed edge, floor cushion, "
        "or low wall as fits the scene. Posture relaxed but composed, hands engaged "
        "(holding a cup / book / phone / fabric / another hand), not stiffly clasped."
    ),

    "斜倚 / 躺卧": (
        "POSE: reclining or leaning — propped on an elbow against a surface, leaning "
        "against a wall, or lying on bed / grass / sofa with the body extended. "
        "Head supported by arm or surface, restful unposed expression."
    ),

    "互动道具": (
        "POSE: actively interacting with a scene-appropriate object — cup, book, "
        "phone, instrument, fabric, accessory, food, drink. The interaction should "
        "look candid and mid-action, not staged display."
    ),

    "镜头互动": (
        "POSE: directly engaging the camera — steady eye contact, slight closed-mouth "
        "smile or composed neutral expression. Subject is aware of being photographed "
        "and comfortable with it; shoulders open toward lens."
    ),

    "侧身回眸": (
        "POSE: side-body with a turn back toward camera — torso angled three-quarters "
        "or fully away, head turned over shoulder to meet the lens, gaze direct. "
        "Hair and clothing show the rotation."
    ),

    # ── 大胆姿态 ────────────────────────────────────────────────────────────
    # 和 WARDROBE_PRESETS["大胆露肤"] 同一个 register：成人编辑摄影语言，
    # 自带 "Adult subject." 收尾。和 "X 岁少女" 类年龄措辞放一起仍会被 gpt-image-2
    # 拒，必须搭"成人/早20+/编辑"语境。

    "大胆开腿": (
        "POSE: deliberate wide-open-leg editorial geometry — knees angle "
        "outward so the gap between the thighs is clearly open (the angle "
        "between the thighs reads well past 90°), feet planted apart wider "
        "than the hip line. Choose ONE body posture that fits the existing "
        "scene / wardrobe context and commit to it — typically one of:\n"
        "  · seated on a low chair, stool, bed edge, step, floor cushion, "
        "or ottoman, weight settled on hips or heels;\n"
        "  · squatting low, feet flat, hips dropped close to heels (street / "
        "studio / gravure variant);\n"
        "  · supine on bed / rug / studio floor, shoulders down, knees up "
        "and apart.\n"
        "If the source/scene reads as 'standing / walking / mid-stride', "
        "switch to squatting so the open-leg geometry actually reads — do "
        "NOT keep the original standing posture and just spread the feet. "
        "One hand rests on the inner thigh, the other free or framing the "
        "face. Camera framing MUST include both knees and the open space "
        "between them — do not crop above the knees and do not hide the "
        "geometry behind hands, fabric, or props. Editorial confidence, "
        "direct steady eye contact, closed-mouth confident expression. "
        "Adult subject."
    ),

    "M字开腿": (
        "POSE: deliberate M-shape open-leg geometry — both knees bent at "
        "roughly 90°, knees pulled WIDE apart so that, seen from the camera, "
        "the silhouette of the legs forms a clear M (NOT crossed, NOT "
        "closed, NOT casually splayed). Choose ONE body posture that fits "
        "the existing scene / wardrobe context and commit to it — typically "
        "one of:\n"
        "  · squatting M-leg: feet flat on the floor, heels under or "
        "slightly behind the hips, knees pulled up roughly to shoulder "
        "height, hips dropped low (the gravure / stage variant — strongly "
        "preferred when the source/scene is standing, walking, outdoor, or "
        "otherwise NOT a sitting context);\n"
        "  · seated M-leg on a low surface (floor cushion, bed, low step, "
        "wide ottoman): feet flat AHEAD OF the hips, heels close to the "
        "buttocks, thighs visibly parted;\n"
        "  · supine M-leg on bed / rug / studio floor: shoulders down, "
        "knees up bent at ~90°, feet flat or lifted, knees fallen open.\n"
        "Do NOT default to 'sit on the ground' just because the M is easier "
        "to draw seated — match the scene. Hands rest on the knees, on the "
        "inner knees, supporting the torso behind the hips, or framing the "
        "hair. The M geometry MUST be clearly readable in the frame — both "
        "knees and the open shape between them visible; do not crop above "
        "the knees, do not bury the geometry under a long skirt or props. "
        "Direct camera gaze, unambiguous confident expression — NOT a tame "
        "cross-legged sit. Adult subject."
    ),

    "挺胸后仰": (
        "POSE: deliberate chest-out back-arch — hips push forward, lower "
        "back arches, ribs and chest lift toward the camera, shoulders pull "
        "back hard, neck extends, head tilts slightly back exposing the "
        "throat line. One hand can rest at the back of the head, lift hair "
        "off the neck, or trail down the body; do not break the arch with "
        "crossed arms. The arch MUST be clearly readable in the frame — "
        "include the torso, do not crop tight to the face. Decisive "
        "editorial line, unembarrassed expression. Adult subject."
    ),

    "弯腰俯身": (
        "POSE: deliberate forward bend — torso tipped forward from the hips "
        "toward the camera at a clear angle (around 30°–60° from vertical), "
        "back held flat or with slight natural curve, shoulders carried "
        "forward, chest line angled toward the lens. Hands rest on the "
        "knees, on a surface in front, or frame the face from below. The "
        "lean MUST be clearly visible — frame the torso, do not crop tight "
        "to the face. Direct eye contact, knowing closed-mouth expression. "
        "Adult subject."
    ),

    "跪姿": (
        "POSE: kneeling editorial — kneel on a soft surface (bed, rug, sand, "
        "studio floor) with knees together or slightly apart, torso upright "
        "or leaning back to rest on the heels, one hand on the thigh and "
        "the other trailing along the body or supporting the head. Composed, "
        "unhurried, fully intentional. Adult subject."
    ),

    "跨坐": (
        "POSE: deliberate editorial straddle — sit astride a backward-facing "
        "chair, low stool, ottoman, padded rail, or motorcycle saddle. "
        "Thighs splayed wide around the seat, knees angled outward, weight "
        "settled at the hips. Forearms drape over the chair back, hands "
        "rest on the seat behind the hips, or hold the saddle. The straddle "
        "geometry (legs apart around the seat) MUST be clearly visible in "
        "the frame — do not foreshorten or shoot from an angle that hides "
        "it. Direct gaze, magazine-cover composure. Adult subject."
    ),

    "趴卧": (
        "POSE: prone reclining editorial — lie face-down on bed / rug / "
        "grass / studio backdrop, torso propped on forearms or elbows, "
        "lower legs lifted and either crossed at the ankles or trailing "
        "back, hips slightly raised, head turned toward camera with relaxed "
        "gaze. Editorial languor. Adult subject."
    ),

    "撩起裙摆": (
        "POSE: deliberate hem-lift — fingers grip the hem of a skirt, dress, "
        "slip, or shirt and visibly pull it upward (to one side or toward "
        "the front), exposing additional thigh or waist. The lifted hem MUST "
        "land high enough that the lift is unambiguous in the frame — do "
        "not draw it back to the original hemline, do not crop above where "
        "the new hem sits. Other hand free or supporting the body. Direct "
        "gaze, composed playful expression. Styled fashion gesture, never "
        "crude reveal. Adult subject."
    ),

    "胸前夹持": (
        "POSE: chest-press editorial hold — a slim styling prop (lipstick "
        "tube, slim phone, paintbrush handle, flower stem, lollipop, "
        "business card, pen) wedged at the open neckline of a low V-cut "
        "garment. CRITICAL: the V opening must be deep and open enough that "
        "the décolletage area is clearly exposed in the frame and the prop "
        "is plainly visible — the upper half of the prop must protrude "
        "above the garment edge, the lower half held in place by the "
        "natural body-against-garment pressure at the inner neckline. The "
        "prop must NOT be tucked under fabric or hidden behind clothing; "
        "if wardrobe coverage would hide it, override coverage so the prop "
        "reads clearly on camera. Arms relaxed at sides or trailing along "
        "the hips, no hand support on the prop. Eye contact with camera, "
        "slight closed-mouth smile, playful editorial fashion register. "
        "Adult subject."
    ),

    "手探内侧": (
        "POSE: deliberate inner-thigh hand placement — one hand rests on "
        "the inner thigh BETWEEN the knees, fingers relaxed and styled, "
        "clearly positioned in the open space between the legs (NOT on top "
        "of the thigh, NOT pressed against the side of a closed-knee leg). "
        "The other hand supports the head, frames the face, or trails along "
        "the body. The hand placement MUST be unambiguous in the frame. "
        "Direct composed eye contact. Adult subject."
    ),

    "含蓄遮挡": (
        "POSE: deliberate caught-mid-cover — one arm crossed loosely over "
        "the chest OR a hand at the collarbone / lap in a half-shielding "
        "gesture, as if surprised by the camera but NOT actually concealing "
        "anything material. The gesture must read as visibly performative "
        "(the body line is still visible past the gesture). Lips slightly "
        "parted in mock-coy / amused expression, gaze flicked toward the "
        "lens. Editorial caught-moment register. Adult subject."
    ),

    "撩拨": (
        "POSE: deliberate mid-action wardrobe interaction — fingers hooked "
        "at a hem, strap, lace edge, zipper pull, or neckline edge and "
        "visibly pulling the fabric (lifting it / drawing it aside / "
        "loosening it). The fabric tension MUST be readable in the frame — "
        "show the cloth being moved, not the moment after release. Direct "
        "eye contact, knowing closed-mouth expression. Composed styled "
        "flirtation, never crude. Adult subject."
    ),

    "舔唇": (
        "POSE: deliberate tongue-or-lip expression — tip of tongue resting "
        "at the corner of the mouth, OR tongue tracing the lower lip, OR "
        "lower lip drawn slightly between the teeth. Eyes half-lidded, gaze "
        "toward the camera. The expression MUST be unambiguous in the "
        "frame; do not soften to a closed mouth. Editorial sultry, never "
        "cartoonish. Adult subject."
    ),

    "湿身写真": (
        "POSE: deliberate wet-look editorial — fabric visibly soaked and "
        "clinging to the body silhouette so the body line reads clearly "
        "through the material, water beads tracking on exposed skin, hair "
        "damp and trailing. Posture relaxed but charged: standing in shower "
        "light, leaning against a wet wall, mid-splash at a beach edge, or "
        "rinsing off poolside. Confident composed expression. The wet, "
        "body-conforming fabric MUST be clearly visible in the frame — do "
        "NOT replace it with dry composed wardrobe. Adult subject."
    ),

    "被定格": (
        "POSE: deliberate candid mid-gesture — body frozen mid-action: "
        "turning toward the camera, tossing hair, brushing fabric aside, "
        "undoing a button or strap, adjusting hem, kicking off a shoe. The "
        "body MUST read as in motion, not as a static pose. Expression "
        "caught between surprise and composure, lips slightly parted. Sense "
        "of being seen unplanned. Adult subject."
    ),
}
POSE_PRESET_KEYS = tuple(POSE_PRESETS.keys())
DEFAULT_POSE_PRESET = "无"


# ── 随机灵感 anchor 池 ───────────────────────────────────────────────────────
# 模型对抽象 seed 整数基本无感，但对具体例子高度敏感。每次润色调用按 seed 抽几个
# 具体 anchor 塞进 system prompt —— 相同 prompt 跑两次得到不同 anchor → 不同输出。
_ANCHOR_POOLS: dict[str, tuple[str, ...]] = {
    "era": (
        "90s paparazzi / tabloid flash",
        "70s gravure soft-focus magazine spread",
        "80s mall portrait studio (vignette + harsh hair light)",
        "Y2K early digital camcorder still",
        "Showa-era found-photo",
        "2000s Tumblr indie zine",
        "2010s independent fashion magazine",
        "contemporary fashion-week street capture",
        "early-90s film photography, pre-digital",
        "60s collected-portrait Polaroid",
        "late-2000s flip-phone snapshot",
        "70s disco / studio 54 era",
    ),
    "venue": (
        "Tokyo Harajuku alley with vending machines",
        "Seoul Gangnam high-rise glass interior",
        "LA poolside late afternoon, deck chairs",
        "Berlin techno-club basement bathroom mirrors",
        "Paris pre-war atelier with tall windows",
        "Shanghai neon alley after rain",
        "Bangkok night-market sidewalk, food steam",
        "NYC East Village dive-bar booth",
        "Kyoto ryokan tatami room with paper screens",
        "Hong Kong rooftop with skyline backdrop",
        "Osaka karaoke booth, plush velvet seats",
        "Singapore HDB stairwell, fluorescent overhead",
        "Taipei night market under awnings",
        "Beijing hutong courtyard with red lanterns",
    ),
    "lighting": (
        "harsh single bare-bulb hard light, deep cast shadows",
        "low-key noir, single key from camera-left, near-black backdrop",
        "magic-hour rim from window, warm core + cool spill",
        "mixed tungsten + fluorescent contamination, slight green cast",
        "single direct on-camera flash, blown highlights",
        "club gel wash, cyan + magenta cross-lighting",
        "soft window-side daylight, no fill",
        "neon sign as practical key, weak ambient",
        "ringlight beauty key, almost no shadows",
        "candlelight + practical lamp, very warm",
        "stage spotlight from above, hard pool of light",
        "harsh midday sun, contrasty cast shadows",
    ),
    "palette": (
        "neon pink + cyan, saturated",
        "sepia bronze, muted brown highlights",
        "muted earth — olive, ochre, terracotta",
        "cool blue-grey shadows, neutral highlights",
        "blown-out near-white high-key",
        "Y2K bright saturated rainbow",
        "monochrome with cool highlights",
        "warm golden cores + crimson shadows",
        "teal-and-orange editorial",
        "desaturated film-like, slight green",
        "pastel candy pinks + lilac",
        "deep wine red + ivory cream",
    ),
    "lens": (
        "24mm wide, slight edge distortion",
        "35mm street, environmental",
        "50mm classic normal",
        "85mm portrait, creamy bokeh",
        "135mm tele compression",
        "60mm macro close-detail",
        "16mm fisheye, exaggerated near-far",
        "28mm reportage",
    ),
}


def _pick_random_anchors(seed: int | None) -> dict[str, str]:
    """Seed-driven random anchor pick. Same seed → same anchors (reproducible)."""
    rng = random.Random(seed) if seed is not None else random.Random()
    return {axis: rng.choice(pool) for axis, pool in _ANCHOR_POOLS.items()}


def _anchor_block(intensity: str, variant_kind: str | None, anchors: dict[str, str]) -> str:
    """Format the random anchors as a system-prompt block, scaled by intensity / variant."""
    if variant_kind == "wild" or intensity == "unhinged":
        axes = ("era", "venue", "lighting", "palette", "lens")
    elif intensity == "aggressive":
        axes = ("venue", "lighting", "palette", "lens")
    elif variant_kind == "soft":
        axes = ("palette", "lens", "lighting")
    elif intensity == "gacha":
        axes = ("palette", "lighting")
    else:
        return ""

    lines = [
        "INSPIRATION ANCHORS (THIS CALL ONLY — different seeds yield different "
        "anchors, which is the mechanism that makes repeat invocations produce "
        "different shoots):"
    ]
    for axis in axes:
        if axis in anchors:
            lines.append(f"  • {axis}: {anchors[axis]}")
    lines.append(
        "Use the anchor (or a clear neighbour of it) as the concrete direction "
        "for each listed aspect. Anchors that CONFLICT with an active PRESET "
        "or LOCK are SKIPPED — presets / locks always win on the aspects they "
        "own. For aspects the presets/locks do NOT touch, commit to the "
        "anchor rather than falling back to a generic default ('studio "
        "backdrop', 'natural daylight', '50mm portrait')."
    )
    return "\n".join(lines)


def _build_polish_system(
    mode: str,
    intensity: str,
    locks: dict[str, bool] | None,
    target_language: str,
    wardrobe_preset: str | None = None,
    extra_polish_rules: str | None = None,
    scene_preset: str | None = None,
    shooting_style_preset: str | None = None,
    framing_preset: str | None = None,
    pose_preset: str | None = None,
    variant_kind: str | None = None,
    creative_seed: int | None = None,
) -> str:
    """根据模式/强度/锁/底子/用户补充拼出 system prompt。"""
    sections: list[str] = [_BASE_SYSTEM_PROMPT]

    lang_note = (
        "Keep each field in its original language; only fix grammar and density."
        if target_language in {"保持原文", "same", "same language"}
        else f"Output every field in {target_language}."
    )
    sections.append(f"LANGUAGE: {lang_note}")

    if mode == "translate":
        sections.append(
            "MODE=TRANSLATE-ONLY. Translate faithfully and keep density similar to "
            "the source. Do not invent new visual details. Ignore intensity, treat "
            "as CONSERVATIVE."
        )
    elif mode == "polish":
        sections.append(
            "MODE=POLISH. The input is already in the target language. Tighten it, "
            "strip filler praise, push vague adjectives into concrete photography "
            "vocabulary, and push negatives into avoid_prompt."
        )
    else:
        sections.append(
            "MODE=TRANSLATE+POLISH. Translate, then apply the photography rewrite. "
            "Stay grounded in real-camera vocabulary."
        )

    intensity_text = _INTENSITY_NOTES.get(intensity, _INTENSITY_NOTES["conservative"])
    sections.append(intensity_text)

    # 全局优先级 + 三步应用程序：用户主动选的 preset 比 intensity 的「不要改」
    # 硬规则优先；并且必须 DETECT → STRIP → APPLY 三步走 —— 否则模型会把 preset
    # 当成"附加修饰"贴在原文旁边，原姿态/原服装仍然主导最终输出。
    any_preset_active = any(
        bool(p) and p not in ("无",) for p in (
            wardrobe_preset, scene_preset, shooting_style_preset,
            framing_preset, pose_preset,
        )
    )
    if any_preset_active:
        sections.append(
            "PRESET PRIORITY & APPLICATION PROCEDURE (READ BEFORE WRITING ANY "
            "OUTPUT FIELD):\n"
            "One or more PRESET BASE / PRESET DIRECTIVE blocks appear below "
            "(wardrobe / scene / style / framing / pose). Each is the user's "
            "ACTIVE creative choice for that aspect and is HIGHER PRIORITY "
            "than the intensity rule above — they OVERRIDE 'do not invent new "
            "outfits / new poses / new locations' for the aspect they "
            "address. Intensity still governs every aspect NOT touched by any "
            "preset. LOCKS still beat presets (a locked field is never "
            "rewritten regardless of which preset is selected).\n"
            "\n"
            "FOR EACH ACTIVE PRESET, you MUST execute these three steps in "
            "order before writing the affected output field:\n"
            "  STEP 1 — DETECT: scan ALL editable input fields "
            "(general_prompt, main_prompt, person_prompt, style_prompt, "
            "scene_prompt) for ANY language about the SAME aspect the preset "
            "addresses. Aspect → input-language map:\n"
            "    • WARDROBE preset → any garment / coverage / fabric / "
            "neckline / hemline / silhouette language (e.g. 'high-neck "
            "sweater', 'long jeans', 'covered shoulders', 'modest dress').\n"
            "    • SCENE preset → any location / venue / time-of-day / "
            "weather / background language (e.g. 'on a beach at sunset', "
            "'in a coffee shop').\n"
            "    • STYLE preset → any camera body / lens / film stock / ISO "
            "/ lighting setup / color grade language.\n"
            "    • FRAMING preset → any shot-distance / close-up / half-body "
            "/ full-body / angle language.\n"
            "    • POSE preset → any pose / gesture / body-position / "
            "expression / gaze / hand-placement language (e.g. 'sitting "
            "cross-legged', 'standing with arms folded', 'looking at her "
            "phone', 'gentle smile', 'hands on hips' — ALL of these are "
            "pose language).\n"
            "  STEP 2 — STRIP: in your OUTPUT for editable fields, REMOVE "
            "the detected language entirely. Do NOT echo it back. Do NOT "
            "keep both the original description AND the preset description "
            "side-by-side — that yields an incoherent prompt where two "
            "poses / two outfits / two locations fight each other. "
            "Examples:\n"
            "    • Input person_prompt: 'sitting cross-legged on a sofa "
            "looking at her phone, gentle smile'. POSE preset: M-shape "
            "open-leg. → Output person_prompt DELETES 'sitting cross-legged "
            "... gentle smile' entirely and writes the M-shape pose in its "
            "place. Do not append M-shape after the cross-legged phrase.\n"
            "    • Input person_prompt: 'wearing a turtleneck sweater and "
            "long jeans'. WARDROBE preset: 大胆露肤 (low and revealing). → "
            "Output person_prompt DELETES the turtleneck+jeans description "
            "and writes a wardrobe that matches the preset's low-coverage "
            "silhouette and posture instead.\n"
            "    • Input scene_prompt: 'on a sunset beach with waves'. "
            "SCENE preset: 室内日常 (indoor everyday). → Output scene_prompt "
            "DELETES the beach/sunset/waves and writes an indoor everyday "
            "scene instead.\n"
            "  STEP 3 — APPLY: write the preset's geometry / coverage / "
            "camera / framing / location language into the appropriate "
            "target field, translated into concrete real-shoot vocabulary. "
            "Target-field map:\n"
            "    • WARDROBE → person_prompt\n"
            "    • SCENE → scene_prompt\n"
            "    • STYLE → style_prompt\n"
            "    • FRAMING → style_prompt\n"
            "    • POSE → person_prompt\n"
            "\n"
            "Frozen fields (locked, or outside the editable set the user "
            "specified) are EXEMPT from STEPS 1 and 2 — never strip or "
            "rewrite frozen-field language even if it conflicts with a "
            "preset. In that case the preset only applies to the editable "
            "fields, and the resulting prompt may be partially incoherent — "
            "that is the user's deliberate choice via the lock."
        )

    # 变体指令：注入 creative seed + 多样性约束，让多次调用结果分散
    if variant_kind in {"soft", "wild"}:
        seed_str = (
            f"creative_seed={creative_seed}. Use this as a hidden tie-breaker — "
            "different seeds MUST yield different concrete details, vocabulary, and "
            "framing choices. Do not echo the seed in output."
        ) if creative_seed is not None else (
            "Treat this as a creative variation pass — different invocations should "
            "yield different concrete details."
        )
        if variant_kind == "soft":
            sections.append(
                "MODE=VARIANT-SOFT (detail variation). " + seed_str + " The "
                "SUBJECT IDENTITY (who they are), the broad scene category, "
                "and the overall genre / register all STAY. Randomly swap "
                "out the CONCRETE DETAILS: wardrobe colors and materials, "
                "accessories, camera body / lens / film stock, lighting "
                "direction, time of day, weather, props, small scene "
                "elements. Do NOT introduce a new person, a different "
                "venue category, or a different shoot genre. The result "
                "must read as a different shoot of the same subject in a "
                "comparable setting."
            )
        else:  # wild
            sections.append(
                "MODE=VARIANT-WILD (creative pivot pass — NOT a regular "
                "polish). " + seed_str + " This is a deliberate creative "
                "DEPARTURE from the source brief. Treat the input as a "
                "starting point you are FREE to leave behind.\n"
                "Required behavior:\n"
                "  1) Pick ONE bold pivot anchor — a deliberate shift on "
                "an axis the source did NOT specify or did NOT commit to. "
                "Examples (pick a different one each time the seed "
                "changes): a different decade (90s tabloid / 70s gravure / "
                "2010s indie-mag), a different region (Tokyo street / "
                "Seoul editorial / Berlin techno-club / LA poolside), a "
                "different time of day, a different venue category, a "
                "different wardrobe genre, a different lighting key.\n"
                "  2) Aggressively rewrite wardrobe, scene type, lighting, "
                "mood, camera era, and pose vocabulary AROUND that anchor — "
                "the output must read as a DIFFERENT SHOOT CONCEPT, not "
                "the same shoot with cosmetic tweaks.\n"
                "  3) BANNED: keeping the original location category, the "
                "original wardrobe genre, the original lighting key, or "
                "the original pose vocabulary — except where a PRESET "
                "directive or LOCK explicitly fixes that aspect.\n"
                "  4) The ONLY survivors from the source are: subject "
                "identity, any locked fields, and any active preset "
                "directives below. Everything else is yours to reinvent.\n"
                "Stay coherent and shootable — ONE concrete shoot, not a "
                "moodboard. Editorial register, never crude."
            )

    active_locks = {"identity": True}
    if locks:
        active_locks.update(locks)
    lock_lines = [f"- {_LOCK_NOTES[name]}" for name, enabled in active_locks.items() if enabled and name in _LOCK_NOTES]
    if lock_lines:
        sections.append("LOCKS (override intensity when they conflict):\n" + "\n".join(lock_lines))

    # 润色底子（多维度方向预设）：服装/场景/风格/构图/姿态。
    # 每个预设的 "不改动" 选项自带 PRESERVE 指令，权重高于 intensity。
    preset_directives: list[tuple[str, dict[str, str]]] = [
        (wardrobe_preset or "", WARDROBE_PRESETS),
        (scene_preset or "",    SCENE_PRESETS),
        (shooting_style_preset or "", SHOOTING_STYLE_PRESETS),
        (framing_preset or "",  FRAMING_PRESETS),
        (pose_preset or "",     POSE_PRESETS),
    ]
    for key, table in preset_directives:
        directive = table.get(key, "") if key else ""
        if directive:
            sections.append(directive)

    # 随机灵感 anchor（gacha 及以上 / 变体模式下注入）：让相同 prompt 不同 seed
    # 产生不同输出。anchor 自带 "presets / locks 冲突时跳过" 的说明，权重低于
    # 前面的 preset directives。
    if creative_seed is not None:
        block = _anchor_block(intensity, variant_kind, _pick_random_anchors(creative_seed))
        if block:
            sections.append(block)

    # 用户自定义润色补充：放在最后，权重最高（覆盖前面的预设倾向）
    extra = (extra_polish_rules or "").strip()
    if extra:
        sections.append(
            "USER ADDITIONAL DIRECTIVES (these instructions come from the user and "
            "OVERRIDE any styling base or intensity preset above when they conflict):\n"
            + extra
        )

    return "\n\n".join(sections)


def refine_prompt_fields_with_gpt5(
    fields: dict[str, str],
    mode: str = "translate_polish",
    model: str = "gpt-5.5",
    retries: int = 5,
    timeout: int = 300,
    target_language: str = "English",
    intensity: str = "conservative",
    target_fields: list[str] | None = None,
    locks: dict[str, bool] | None = None,
    locked_fields: list[str] | None = None,
    provider: str | None = None,
    base_url: str | None = None,
    wardrobe_preset: str | None = None,
    extra_polish_rules: str | None = None,
    scene_preset: str | None = None,
    shooting_style_preset: str | None = None,
    framing_preset: str | None = None,
    pose_preset: str | None = None,
    variant_kind: str | None = None,
    creative_seed: int | None = None,
    polish_link_mode: str = "linked",
) -> dict[str, str]:
    """对一组 prompt 字段做协调式翻译/润色。返回 JSON 解析后的字段字典。

    参数：
      variant_kind: None / "soft"（保主体）/ "wild"（脑洞）—— 仅影响 system prompt
      creative_seed: 变体随机种子；None 时若 variant_kind 已开启会自动生成
      polish_link_mode: "linked"（默认，一次性把所有可改字段塞进同一 JSON 调用，
                       字段之间互相参考）/ "independent"（每个可改字段单独发一次
                       请求，字段之间互不干扰；其它字段仍作为 frozen 上下文传入）
    """
    # 锁定字段先单独抽出来 —— dedupe 不能动它们，否则 lock 起不到隔离作用
    locked_set = {k for k in (locked_fields or []) if k in fields}
    source = dedupe_prompt_fields(
        {k: repair_mojibake(v) for k, v in fields.items()},
        target_language=target_language,
        exempt=locked_set,
    )
    compact = {k: v for k, v in source.items() if v.strip()}
    if not compact:
        return {}

    # translate-only 强制保守
    if mode == "translate":
        intensity = "conservative"

    requested = [k for k in (target_fields or []) if k in compact]
    requested_locks = {k for k in (locked_fields or []) if k in compact}
    editable = requested or list(compact.keys())
    editable = [k for k in editable if k not in requested_locks]
    frozen = [k for k in compact if k not in editable]

    # 变体 / 高强度档：seed 缺省时自动生成 —— 让 system prompt 里的 creative_seed
    # 以及随后注入的 INSPIRATION ANCHORS 真随机。低强度档（保守/开放）不种随机
    # 避免用户期待"忠实翻译"时被惊到。
    creative_intensity = intensity in {"gacha", "aggressive", "unhinged"}
    if (variant_kind in {"soft", "wild"} or creative_intensity) and creative_seed is None:
        creative_seed = random.randint(1, 999_999_999)

    def _build_sys() -> str:
        return _build_polish_system(
            mode, intensity, locks, target_language, wardrobe_preset, extra_polish_rules,
            scene_preset=scene_preset,
            shooting_style_preset=shooting_style_preset,
            framing_preset=framing_preset,
            pose_preset=pose_preset,
            variant_kind=variant_kind,
            creative_seed=creative_seed,
        )

    # —— 独立润色：每个可改字段单独发一次 API，其它字段都作 frozen 上下文 ——
    if polish_link_mode == "independent" and editable:
        output: dict[str, str] = {}
        for key in fields:
            if key in frozen:
                output[key] = source.get(key, "")
        # 独立模式下每个 editable 用独立 seed（变体或高强度档）—— 进一步分散
        for idx, target_key in enumerate(editable):
            this_seed = creative_seed
            if creative_seed is not None and (variant_kind in {"soft", "wild"} or creative_intensity):
                this_seed = (creative_seed + idx * 9973) % 1_000_000_000

            sys_one = _build_polish_system(
                mode, intensity, locks, target_language, wardrobe_preset, extra_polish_rules,
                scene_preset=scene_preset,
                shooting_style_preset=shooting_style_preset,
                framing_preset=framing_preset,
                pose_preset=pose_preset,
                variant_kind=variant_kind,
                creative_seed=this_seed,
            )
            other_frozen = [k for k in compact if k != target_key]
            user_message = (
                "INDEPENDENT MODE. Rewrite ONLY this single field; all other fields "
                "are frozen context (do not echo them in your output). "
                f"Editable field: {json.dumps([target_key], ensure_ascii=False)}\n"
                f"Frozen context fields: {json.dumps(other_frozen, ensure_ascii=False)}\n\n"
                "Return a JSON object with exactly one key (the editable field).\n\n"
                "INPUT FIELDS:\n"
                f"{json.dumps(compact, ensure_ascii=False, indent=2)}"
            )
            response_text = call_text_model(
                user_message,
                model=model, retries=retries, timeout=timeout,
                provider=provider, base_url=base_url,
                system_prompt=sys_one,
            )
            parsed_one = _parse_polish_json(response_text)
            if not isinstance(parsed_one, dict):
                raise ValueError("gpt-5.5 did not return a JSON object.")
            raw = parsed_one.get(target_key, source.get(target_key, ""))
            output[target_key] = repair_mojibake(str(raw).strip())
        # 保证返回顺序按原 fields 顺序
        return {k: output.get(k, source.get(k, "")) for k in fields if k in output}

    # —— 关联润色（默认）：一次性把所有 editable+frozen 送给模型 ——
    system_prompt = _build_sys()

    user_message = (
        "Editable fields (rewrite per the rules): "
        f"{json.dumps(editable, ensure_ascii=False)}\n"
        "Frozen fields (return byte-for-byte unchanged; use only for context): "
        f"{json.dumps(frozen, ensure_ascii=False)}\n\n"
        "INPUT FIELDS:\n"
        f"{json.dumps(compact, ensure_ascii=False, indent=2)}"
    )

    response_text = call_text_model(
        user_message,
        model=model, retries=retries, timeout=timeout,
        provider=provider, base_url=base_url,
        system_prompt=system_prompt,
    )

    parsed = _parse_polish_json(response_text)
    if not isinstance(parsed, dict):
        raise ValueError("gpt-5.5 did not return a JSON object.")

    output = {}
    for key in fields:
        if key in frozen:
            output[key] = source.get(key, "")
            continue
        raw = parsed.get(key, source.get(key, ""))
        output[key] = repair_mojibake(str(raw).strip())
    return output


def _parse_polish_json(text: str) -> dict:
    candidate = text.strip()
    if candidate.startswith("```"):
        candidate = re.sub(r"^```(?:json)?\s*", "", candidate)
        candidate = re.sub(r"\s*```$", "", candidate)
    try:
        return json.loads(candidate)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", candidate, re.S)
        if not match:
            raise ValueError(f"gpt-5.5 did not return JSON: {text[:300]}")
        return json.loads(match.group(0))


# ─────────────────────────────────────────────────────────────────────────────
# 15. argparse 类型校验
# ─────────────────────────────────────────────────────────────────────────────

def _bounded_int(low: int, high: int, label: str) -> Callable[[str], int]:
    def parser(value: str) -> int:
        try:
            n = int(value)
        except ValueError as exc:
            raise argparse.ArgumentTypeError(f"{label} must be an integer") from exc
        if not low <= n <= high:
            raise argparse.ArgumentTypeError(f"{label} must be between {low} and {high}")
        return n
    return parser


image_count        = _bounded_int(1, 10, "image count")
compression_value  = _bounded_int(0, 100, "compression")
partial_images_value = _bounded_int(0, 3, "partial images")
concurrency_value  = _bounded_int(0, 10, "concurrency")


def positive_int(value: str) -> int:
    try:
        n = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be an integer") from exc
    if n < 1:
        raise argparse.ArgumentTypeError("must be greater than 0")
    return n


# ─────────────────────────────────────────────────────────────────────────────
# 16. CLI
# ─────────────────────────────────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate gpt-image-2 quality images via Responses API (multi-provider).",
        allow_abbrev=False,
    )
    parser.add_argument("prompt", nargs="*",
                        help="Prompt text. Spaces OK; quotes optional unless text begins with '-'.")
    parser.add_argument("-n", type=image_count, default=1, help="Number of images (1-10).")
    parser.add_argument("-q", "--quality", default="auto",
                        choices=list(ImageLimits.QUALITIES))
    parser.add_argument("-s", "--size", default="1024x1024", type=image_size,
                        help="Image size like 1024x1024, 2048x2048, or auto.")
    parser.add_argument("-f", "--format", default="png",
                        choices=("png", "webp"), dest="fmt")
    parser.add_argument("-r", "--ref", action="append", default=[],
                        help="Reference image path, URL, or data URL. Repeatable.")
    parser.add_argument("-o", "--out-dir", default="output")
    parser.add_argument("-m", "--model", default="gpt-5.5")
    parser.add_argument("--prompt-file", action="append", default=[],
                        help="Extra prompt text from file. Repeatable.")
    parser.add_argument("--retries", type=positive_int, default=5)
    parser.add_argument("--timeout", type=positive_int, default=180)
    parser.add_argument("--background", default="auto",
                        choices=list(ImageLimits.BG_CHOICES))
    parser.add_argument("--compression", type=compression_value, default=None,
                        help="Output compression for webp, 0-100.")
    parser.add_argument("--moderation", default="auto",
                        choices=list(ImageLimits.MOD_CHOICES))
    parser.add_argument("--action", default="generate", choices=["generate"])
    parser.add_argument("--partial-images", type=partial_images_value, default=0)
    parser.add_argument("--max-concurrency", type=concurrency_value, default=0,
                        help="0 = auto (min of image count and token count).")
    parser.add_argument("--stream", action=argparse.BooleanOptionalAction, default=True,
                        help="Use SSE streaming (default on).")
    parser.add_argument("--ref-max-edge", type=int, default=RefImage.DEFAULT_MAX_EDGE,
                        help=f"Long edge cap for ref images (default {RefImage.DEFAULT_MAX_EDGE}; 0 disables).")
    parser.add_argument("--provider", default=DEFAULT_PROVIDER,
                        choices=sorted(PROVIDERS.keys()))
    parser.add_argument("--base-url", default=None,
                        help="Override provider base URL, e.g. https://api.example.com/v1")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print parsed options and exit without calling the API.")
    return parser


def _parse_args(parser: argparse.ArgumentParser) -> argparse.Namespace:
    if hasattr(parser, "parse_intermixed_args"):
        return parser.parse_intermixed_args()
    return parser.parse_args()


def _resolve_prompt(args: argparse.Namespace, parser: argparse.ArgumentParser) -> str:
    text = " ".join(args.prompt or []).strip()
    for prompt_file in args.prompt_file:
        path = Path(prompt_file).expanduser()
        if not path.is_file():
            parser.error(f"prompt file not found: {prompt_file}")
        chunk = read_text_file(path).strip()
        text = f"{text}\n{chunk}".strip() if text else chunk
    if not text and not sys.stdin.isatty():
        text = sys.stdin.read().strip()
    if not text:
        parser.error("prompt is required.")
    return text


def main() -> None:
    configure_stdio()
    parser = build_parser()
    args = _parse_args(parser)
    prompt = _resolve_prompt(args, parser)
    refs = [image_to_data_url(ref, max_edge=args.ref_max_edge) for ref in args.ref]

    if args.dry_run:
        preview = {
            "prompt": prompt, "n": args.n, "quality": args.quality, "size": args.size,
            "format": args.fmt, "ref_count": len(refs), "out_dir": args.out_dir,
            "model": args.model, "retries": args.retries, "timeout": args.timeout,
            "background": args.background, "compression": args.compression,
            "moderation": args.moderation, "action": args.action,
            "partial_images": args.partial_images, "max_concurrency": args.max_concurrency,
            "stream": args.stream, "provider": args.provider, "base_url": args.base_url,
            "ref_max_edge": args.ref_max_edge,
            "endpoint": resolve_endpoint(args.provider, args.base_url),
        }
        print(json.dumps(preview, ensure_ascii=False, indent=2))
        return

    generate(
        prompt,
        n=args.n, quality=args.quality, size=args.size, output_format=args.fmt,
        ref_images=refs or None, out_dir=args.out_dir, model=args.model,
        retries=args.retries, timeout=args.timeout,
        background=args.background, output_compression=args.compression,
        moderation=args.moderation, action=args.action,
        partial_images=args.partial_images, max_concurrency=args.max_concurrency,
        provider=args.provider, base_url=args.base_url, stream=args.stream,
    )


if __name__ == "__main__":
    main()
