#!/usr/bin/env python3
"""API_GPT5_5_to_IMAGE2_GUI.py — Tkinter GUI for the image generation backend."""
from __future__ import annotations

import contextlib
import hashlib
import io
import json
import os
import queue
import subprocess
import sys
import threading
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, simpledialog, ttk

import requests

from API_GPT5_5_to_IMAGE2 import (
    DEFAULT_FRAMING_PRESET,
    DEFAULT_POSE_PRESET,
    DEFAULT_PROVIDER,
    DEFAULT_SCENE_PRESET,
    DEFAULT_SHOOTING_STYLE_PRESET,
    DEFAULT_WARDROBE_PRESET,
    FRAMING_PRESET_KEYS,
    POLISH_MODEL_PRESETS,
    POSE_PRESET_KEYS,
    PROVIDERS,
    REF_IMAGE_DEFAULT_MAX_EDGE,
    SCENE_PRESET_KEYS,
    SHOOTING_STYLE_PRESET_KEYS,
    WARDROBE_PRESET_KEYS,
    configure_stdio,
    dedupe_prompt_fields,
    generate,
    image_size,
    image_to_data_url,
    read_text_file,
    refine_prompt_fields_with_gpt5,
    repair_mojibake,
)


APP_DIR = Path(__file__).resolve().parent
CLI_SCRIPT = APP_DIR / "API_GPT5_5_to_IMAGE2.py"
PROFILES_PATH = APP_DIR / "gui_profiles.json"
LEGACY_PROFILES_PATH = APP_DIR / "freemodel_gui_params.json"


class _Tooltip:
    """A no-frills hover tooltip for ttk widgets (Tk has no native one)."""

    def __init__(self, widget: tk.Widget, text: str, *, delay_ms: int = 400):
        self.widget = widget
        self.text = text
        self.delay = delay_ms
        self._after_id: str | None = None
        self._tip: tk.Toplevel | None = None
        widget.bind("<Enter>", self._schedule, add="+")
        widget.bind("<Leave>", self._hide, add="+")
        widget.bind("<ButtonPress>", self._hide, add="+")

    def _schedule(self, _event=None):
        self._cancel()
        self._after_id = self.widget.after(self.delay, self._show)

    def _cancel(self):
        if self._after_id:
            self.widget.after_cancel(self._after_id)
            self._after_id = None

    def _show(self):
        if self._tip is not None:
            return
        x = self.widget.winfo_rootx() + 12
        y = self.widget.winfo_rooty() + self.widget.winfo_height() + 4
        tip = tk.Toplevel(self.widget)
        tip.wm_overrideredirect(True)
        tip.wm_geometry(f"+{x}+{y}")
        label = tk.Label(
            tip, text=self.text, justify="left",
            background="#ffffe0", relief="solid", borderwidth=1,
            padx=6, pady=4, wraplength=380,
        )
        label.pack()
        self._tip = tip

    def _hide(self, _event=None):
        self._cancel()
        if self._tip is not None:
            self._tip.destroy()
            self._tip = None


def _migrate_legacy_profiles() -> None:
    """If only the old freemodel_gui_params.json exists, rename it once."""
    if PROFILES_PATH.is_file() or not LEGACY_PROFILES_PATH.is_file():
        return
    try:
        LEGACY_PROFILES_PATH.rename(PROFILES_PATH)
    except OSError:
        # Filesystem refused (file in use, perms). Fall back to a copy.
        try:
            PROFILES_PATH.write_bytes(LEGACY_PROFILES_PATH.read_bytes())
        except OSError:
            pass


SIZE_PRESETS = (
    ("Auto", "auto"),
    ("Portrait竖图 1024x1536", "1024x1536"),
    ("Portrait竖图 1440x2560", "1440x2560"),
    ("Portrait竖图 2160x3840", "2160x3840"),
    ("Portrait竖图 safe 2144x3824", "2144x3824"),
    ("Landscape横图 1536x1024", "1536x1024"),
    ("Landscape横图 2560x1440", "2560x1440"),
    ("Landscape横图 3840x2160", "3840x2160"),
    ("Landscape横图 safe 3824x2144", "3824x2144"),
    ("Square方图 1024x1024", "1024x1024"),
    ("Square方图 2048x2048", "2048x2048"),
    ("Square方图 2880x2880", "2880x2880"),
    ("Custom", "custom"),
)
SIZE_LABELS = [label for label, _ in SIZE_PRESETS]
SIZE_BY_LABEL = dict(SIZE_PRESETS)

PROVIDER_OPTIONS = [(cfg.label, key) for key, cfg in PROVIDERS.items()]
PROVIDER_LABELS = [label for label, _ in PROVIDER_OPTIONS]
PROVIDER_BY_LABEL = dict(PROVIDER_OPTIONS)
PROVIDER_LABEL_BY_KEY = {key: label for label, key in PROVIDER_OPTIONS}

# 图像生成必须走 Responses API（gpt-image-2 tool 调用）。Chat-completions 厂商
# （DeepSeek 等）只能用来润色文本，不能用来出图，所以图像下拉里隐藏掉。
IMAGE_PROVIDER_LABELS = [
    cfg.label for key, cfg in PROVIDERS.items() if cfg.text_api == "responses"
]

POLISH_FOLLOW_LABEL = "跟随图像"
POLISH_PROVIDER_LABELS = [POLISH_FOLLOW_LABEL] + PROVIDER_LABELS

REF_MAX_EDGE_PRESETS = (
    ("1024 (轻量/风格参考)", 1024),
    ("1536 (推荐)", 1536),
    ("2048 (细节)", 2048),
    ("原始尺寸 (不缩)", 0),
)
REF_MAX_EDGE_LABELS = [label for label, _ in REF_MAX_EDGE_PRESETS]
REF_MAX_EDGE_BY_LABEL = dict(REF_MAX_EDGE_PRESETS)
REF_MAX_EDGE_LABEL_BY_VALUE = {value: label for label, value in REF_MAX_EDGE_PRESETS}


class _QueueWriter(io.TextIOBase):
    """File-like stdout/stderr redirect that funnels writes into a queue."""

    def __init__(self, sink: "queue.Queue[tuple[str, object]]") -> None:
        self._sink = sink

    def write(self, text: str) -> int:
        if text:
            self._sink.put(("log", text))
        return len(text)

    def flush(self) -> None:  # pragma: no cover - required by io.TextIOBase
        pass


class ImageWorkbench(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title("GPT-5.5 → Image 2 工作台")
        self.geometry("1180x880")
        self.minsize(1040, 760)

        self._events: "queue.Queue[tuple[str, object]]" = queue.Queue()
        self._ref_paths: list[str] = []
        self._profile_names: list[str] = ["Default"]
        self._busy = False
        self._refining = False
        self._cancel_event: threading.Event | None = None
        self._http_session: requests.Session | None = None
        self._polish_history: list[dict[str, str]] = []
        self._polish_history_limit = 20

        self._init_vars()
        self._build_layout()
        self._wire_traces()
        self._load_profile(quiet=True)
        self._refresh_size_controls()
        self._refresh_format_controls()
        self._refresh_provider_controls()
        self._refresh_command_preview()
        self.after(100, self._pump_events)

    # ── tk Variables ────────────────────────────────────────────────────────

    def _init_vars(self) -> None:
        self.profile_name_var = tk.StringVar(value="Default")
        self.profile_note_var = tk.StringVar()
        self.prompt_file_var = tk.StringVar()
        self.n_var = tk.IntVar(value=1)
        self.quality_var = tk.StringVar(value="auto")
        self.size_preset_var = tk.StringVar(value="Auto")
        self.custom_size_var = tk.StringVar(value="1024x1536")
        self.orientation_var = tk.StringVar(value="真正自动")
        self.format_var = tk.StringVar(value="png")
        self.compression_var = tk.IntVar(value=90)
        self.moderation_var = tk.StringVar(value="auto")
        self.action_var = tk.StringVar(value="generate")
        self.partial_images_var = tk.IntVar(value=0)
        self.stream_var = tk.BooleanVar(value=True)
        self.ref_max_edge_var = tk.StringVar(
            value=REF_MAX_EDGE_LABEL_BY_VALUE.get(REF_IMAGE_DEFAULT_MAX_EDGE, REF_MAX_EDGE_LABELS[1])
        )
        self.out_dir_var = tk.StringVar(value=str(APP_DIR / "output"))
        self.model_var = tk.StringVar(value="gpt-5.5")
        self.polish_model_var = tk.StringVar(value="gpt-5.5")
        self.target_language_var = tk.StringVar(value="English")
        self.retries_var = tk.IntVar(value=5)
        self.timeout_var = tk.IntVar(value=180)
        self.max_concurrency_var = tk.IntVar(value=0)
        self.provider_var = tk.StringVar(
            value=PROVIDER_LABEL_BY_KEY.get(DEFAULT_PROVIDER, PROVIDER_LABELS[0])
        )
        self.base_url_var = tk.StringVar(value="")
        # 润色专用 provider —— 默认跟随图像 provider；选其它项时下方 base_url 才生效。
        self.polish_provider_var = tk.StringVar(value=POLISH_FOLLOW_LABEL)
        self.polish_base_url_var = tk.StringVar(value="")
        self.status_var = tk.StringVar(value="Ready")

        self.intensity_var = tk.StringVar(value="保守")
        self.scope_var = tk.StringVar(value="全部")
        # 关联润色（默认）：所有可改字段一次性塞进同一 JSON 调用，字段之间互相参考
        # 独立润色：每个可改字段单独发一次 API，其它字段作 frozen 上下文，互不污染
        self.polish_link_mode_var = tk.StringVar(value="关联")
        self.wardrobe_preset_var = tk.StringVar(value=DEFAULT_WARDROBE_PRESET)
        self.scene_preset_var = tk.StringVar(value=DEFAULT_SCENE_PRESET)
        self.shooting_style_preset_var = tk.StringVar(value=DEFAULT_SHOOTING_STYLE_PRESET)
        self.framing_preset_var = tk.StringVar(value=DEFAULT_FRAMING_PRESET)
        self.pose_preset_var = tk.StringVar(value=DEFAULT_POSE_PRESET)
        # 用户在 GUI 里"+保存"过的润色模型名 —— 跟着 profile 持久化
        self._polish_model_custom_presets: list[str] = []
        # 防止 load_profile 阶段触发 polish provider 自动覆盖 model / base_url
        self._suppress_polish_provider_sync = False
        # 润色补充：用户写在 GUI 文本框的额外润色规则，挂在所有内置 prompt 之后
        self.lock_general_prompt_var = tk.BooleanVar(value=False)
        self.lock_main_prompt_var = tk.BooleanVar(value=False)
        self.lock_person_prompt_var = tk.BooleanVar(value=False)
        self.lock_style_prompt_var = tk.BooleanVar(value=False)
        self.lock_scene_prompt_var = tk.BooleanVar(value=False)
        self.lock_avoid_prompt_var = tk.BooleanVar(value=False)

    # ── layout ──────────────────────────────────────────────────────────────

    def _build_layout(self) -> None:
        self.columnconfigure(0, weight=1)
        self.rowconfigure(0, weight=1)

        shell = ttk.Frame(self, padding=10)
        shell.grid(row=0, column=0, sticky="nsew")
        shell.columnconfigure(0, weight=1)
        shell.rowconfigure(0, weight=1)
        shell.rowconfigure(1, weight=0)

        canvas = tk.Canvas(shell, highlightthickness=0)
        scroll = ttk.Scrollbar(shell, orient="vertical", command=canvas.yview)
        canvas.configure(yscrollcommand=scroll.set)
        canvas.grid(row=0, column=0, sticky="nsew")
        scroll.grid(row=0, column=1, sticky="ns")

        self.content = ttk.Frame(canvas)
        window = canvas.create_window((0, 0), window=self.content, anchor="nw")
        self.content.bind(
            "<Configure>",
            lambda _event: canvas.configure(scrollregion=canvas.bbox("all")),
        )
        canvas.bind(
            "<Configure>",
            lambda event: canvas.itemconfigure(window, width=event.width),
        )

        self.content.columnconfigure(0, weight=3)
        self.content.columnconfigure(1, weight=2)
        self.content.rowconfigure(0, weight=1)
        self.content.rowconfigure(1, weight=1)

        self._build_prompt_pane(self.content).grid(
            row=0, column=0, sticky="nsew", padx=(0, 8), pady=(0, 8)
        )

        right = ttk.Frame(self.content)
        right.grid(row=0, column=1, sticky="nsew", pady=(0, 8))
        right.columnconfigure(0, weight=1)
        self._build_settings_pane(right).grid(row=0, column=0, sticky="ew", pady=(0, 8))

        self._build_refs_pane(self.content).grid(
            row=1, column=0, columnspan=2, sticky="nsew", pady=(0, 8)
        )

        bottom = ttk.Frame(shell)
        bottom.grid(row=1, column=0, columnspan=2, sticky="nsew", pady=(8, 0))
        bottom.columnconfigure(0, weight=1)
        bottom.rowconfigure(1, weight=1)
        self._build_actions_pane(bottom).grid(row=0, column=0, sticky="ew", pady=(0, 8))
        self._build_log_pane(bottom).grid(row=1, column=0, sticky="nsew")

    def _build_prompt_pane(self, parent: ttk.Frame) -> ttk.LabelFrame:
        frame = ttk.LabelFrame(parent, text="提示词")
        frame.columnconfigure(1, weight=1)

        self.general_prompt = self._add_text_row(
            frame, 0, "通用提示词", 6,
            tooltip=(
                "全局/硬规则字段。建议放：_反AI生成感总纲、整体风格基调、不希望出现"
                "的 AI tells（cinematic / perfect / 8K 等）。\n"
                "拼接顺序：放在最前面 → 模型先看到这些规则再读后续字段。"
            ),
        )
        self.main_prompt = self._add_text_row(
            frame, 1, "主体", 4,
            tooltip=(
                "一句话定调：谁、在哪、在做什么。例：『一名 mid-twenties Korean editorial "
                "subject 站在 Seoul 后巷霓虹灯下』。\n"
                "其它字段都围绕这一句展开 —— 主体一改，整张图意境会跟着变。"
            ),
        )
        self.person_prompt = self._add_text_row(
            frame, 2, "人物参数细节", 4,
            tooltip=(
                "人物档案：年龄段（写 adult / mid-twenties，避免少女/X 岁）、体型、妆发、"
                "瞳色、肤色、表情、服装款式与材质、配饰、佩戴细节。\n"
                "注意 gpt-image-2 审核线：禁止命名公众人物 + 禁止「年轻 + 露骨身体规格」"
                "组合（用 adult fashion / editorial subject 改写）。"
            ),
        )
        self.style_prompt = self._add_text_row(
            frame, 3, "拍摄风格", 4,
            tooltip=(
                "摄影技术参数：相机机身、镜头焦段与光圈、胶片型号、ISO、快门、光位"
                "（key/fill/rim）、调色倾向、景深、动态范围。\n"
                "润色阶段会把抽象形容词（『高级感』『质感』）替换为具体器材语言。"
            ),
        )
        self.scene_prompt = self._add_text_row(
            frame, 4, "场景", 4,
            tooltip=(
                "环境：地点、时间、季节、天气、光线来源（自然光/路灯/霓虹/手机屏）、"
                "氛围、道具、围观者、地面/墙面材质。\n"
                "场景描述影响整张图的情绪与构图节奏。"
            ),
        )
        self.avoid_prompt = self._add_text_row(
            frame, 5, "避免内容", 3,
            tooltip=(
                "Negative prompt：不希望出现的元素 / 风格 / AI tells。\n"
                "润色阶段会自动把『perfect skin / hyperrealistic / cinematic / 8K』"
                "这类词搬进这里。"
            ),
        )

        ttk.Label(frame, text="提示词文件").grid(row=6, column=0, sticky="w", padx=8, pady=6)
        file_row = ttk.Frame(frame)
        file_row.grid(row=6, column=1, sticky="ew", padx=(0, 8), pady=6)
        file_row.columnconfigure(0, weight=1)
        ttk.Entry(file_row, textvariable=self.prompt_file_var).grid(row=0, column=0, sticky="ew", padx=(0, 6))
        ttk.Button(file_row, text="选择", command=self._pick_prompt_files).grid(row=0, column=1, padx=(0, 6))
        ttk.Button(file_row, text="清空", command=lambda: self.prompt_file_var.set("")).grid(row=0, column=2)

        polish_row = ttk.Frame(frame)
        polish_row.grid(row=7, column=1, sticky="ew", padx=(0, 8), pady=(0, 4))
        for col in range(6):
            polish_row.columnconfigure(col, weight=1)
        btn_translate = ttk.Button(
            polish_row, text="翻译",
            command=lambda: self._start_polish("translate"),
        )
        btn_translate.grid(row=0, column=0, sticky="ew", padx=(0, 4))
        _Tooltip(btn_translate, "只翻译不改写 —— 把字段忠实转成目标语言，不增删视觉细节。")

        btn_polish = ttk.Button(
            polish_row, text="润色",
            command=lambda: self._start_polish("polish"),
        )
        btn_polish.grid(row=0, column=1, sticky="ew", padx=4)
        _Tooltip(btn_polish, "假设输入已是目标语言，按摄影词汇库 + 反 AI tells 改写收紧。")

        btn_both = ttk.Button(
            polish_row, text="翻译+润色",
            command=lambda: self._start_polish("translate_polish"),
        )
        btn_both.grid(row=0, column=2, sticky="ew", padx=4)
        _Tooltip(btn_both, "先翻译再润色 —— 最常用的一档。")

        btn_variant_soft = ttk.Button(
            polish_row, text="变体（保主体）",
            command=lambda: self._start_polish("translate_polish", variant_kind="soft"),
        )
        btn_variant_soft.grid(row=0, column=3, sticky="ew", padx=4)
        _Tooltip(
            btn_variant_soft,
            "细节变体 —— 保留主体 / 场景类型 / 体裁不变，随机替换具体细节：\n"
            "  服装颜色 / 材质 / 配饰 / 相机 / 镜头 / 光位 / 时间 / 道具。\n"
            "强度自动 +1 档（最低开放）。适合方向定了想换个具体相机/配色试试。",
        )

        btn_variant_wild = ttk.Button(
            polish_row, text="脑洞（重写）",
            command=lambda: self._start_polish("translate_polish", variant_kind="wild"),
        )
        btn_variant_wild.grid(row=0, column=4, sticky="ew", padx=4)
        _Tooltip(
            btn_variant_wild,
            "创意 pivot —— 强制模型挑一个 anchor（不同年代 / 地区 / 场地 / 光线 key），\n"
            "围绕它重写服装 / 场景 / 光线 / 相机时代 / 姿态。\n"
            "强度自动 +2 档（最低激进）；仅保留：主体身份 + 锁定字段 + 预设方向。\n"
            "适合完全没想法时让模型给个全新概念。",
        )

        self.undo_button = ttk.Button(
            polish_row, text="↶ 回退", command=self._undo_polish, state="disabled",
        )
        self.undo_button.grid(row=0, column=5, sticky="ew", padx=(4, 0))
        _Tooltip(self.undo_button, "回退到上一次润色 / 变体前的提示词（最多 5 步）。")

        polish_row2 = ttk.Frame(frame)
        polish_row2.grid(row=8, column=1, sticky="ew", padx=(0, 8), pady=(0, 4))
        polish_row2.columnconfigure(1, weight=1)
        polish_row2.columnconfigure(3, weight=1)
        polish_row2.columnconfigure(5, weight=1)
        lbl_intensity = ttk.Label(polish_row2, text="润色强度")
        lbl_intensity.grid(row=0, column=0, sticky="w")
        _Tooltip(
            lbl_intensity,
            "只影响润色 / 变体（翻译档强制保守）。每档大致这么干：\n"
            "\n"
            "保守 — 只翻译 + 去重 + 把模糊词换成具体摄影词。\n"
            "      姿态 / 场景 / 服装一字不动，最安全的档。\n"
            "\n"
            "开放 — 不锁定字段允许细化，主动推编辑摄影口吻。\n"
            "      例：'夕阳海边的女生' → 'golden hour 海边, 35mm 胶片, 暖膝盖光'，\n"
            "      场地和主体动作不变，只把抽象词写实。\n"
            "\n"
            "抽卡 — 像扭一次蛋：身份保留，其余全开，重写出一套连贯新造型。\n"
            "      服装 / 姿态 / 光线 / 构图 至少 2 个轴做非默认抉择。\n"
            "      适合想看新方向又懒得自己写时。\n"
            "\n"
            "激进 — 除身份外全部重写；禁用安全默认\n"
            "      （中性站姿 / 平顶光 / 留白底 / 平视全身）。\n"
            "      出来像同一人的另一组概念片，不再是原 prompt 的微调。\n"
            "\n"
            "暴走 — 顶配档。coverage / pose / 表情 / framing / lens / 光位 /\n"
            "      调色 / 场景 mood 至少 5 个轴必须非默认；安全选项全部禁用。\n"
            "      封面感大片档，配 wardrobe / pose preset 最够味。",
        )
        ttk.Combobox(polish_row2, textvariable=self.intensity_var,
                     values=("保守", "开放", "抽卡", "激进", "暴走"),
                     state="readonly", width=8).grid(row=0, column=1, sticky="ew", padx=(4, 8))
        ttk.Label(polish_row2, text="范围").grid(row=0, column=2, sticky="w")
        ttk.Combobox(polish_row2, textvariable=self.scope_var,
                     values=("全部", "仅人物", "仅场景", "仅风格", "人物+场景", "人物+风格"),
                     state="readonly", width=12).grid(row=0, column=3, sticky="ew", padx=(4, 8))
        lbl_link = ttk.Label(polish_row2, text="润色模式")
        lbl_link.grid(row=0, column=4, sticky="w")
        _Tooltip(
            lbl_link,
            "关联（默认）：所有可改字段一次性塞给模型，字段之间互相参考、保持协调。\n"
            "独立：每个字段单独发一次 API，其余字段作上下文不可改 —— 防止"
            "强势字段（例如场景）盖过弱势字段（例如人物细节）；代价是 API 请求数 × 字段数。",
        )
        ttk.Combobox(polish_row2, textvariable=self.polish_link_mode_var,
                     values=("关联", "独立"),
                     state="readonly", width=8).grid(row=0, column=5, sticky="ew", padx=(4, 0))

        polish_row3 = ttk.Frame(frame)
        polish_row3.grid(row=9, column=1, sticky="ew", padx=(0, 8), pady=(0, 8))
        lbl_lock = ttk.Label(polish_row3, text="锁定")
        lbl_lock.grid(row=0, column=0, sticky="w", padx=(0, 6))
        _Tooltip(
            lbl_lock,
            "勾选后：该字段在润色/变体时\n"
            "  ① 不被改写（保留原文一字不动）\n"
            "  ② 不会被 dedupe 抽走片段并入其他字段\n"
            "  ③ 仍作为 frozen 上下文给模型参考（保证整段语义协调）",
        )
        lock_specs = (
            ("通用", self.lock_general_prompt_var, "锁定 通用提示词（_反AI生成感总纲 等硬规则）"),
            ("主体", self.lock_main_prompt_var, "锁定 主体（核心场景+主体一句话定调）"),
            ("人物参数细节", self.lock_person_prompt_var, "锁定 人物参数细节（年龄段/体型/妆发/服装结构）"),
            ("拍摄风格", self.lock_style_prompt_var, "锁定 拍摄风格（相机/镜头/胶片/光位/调色）"),
            ("场景", self.lock_scene_prompt_var, "锁定 场景（地点/时间/光线/氛围/道具）"),
            ("避免内容", self.lock_avoid_prompt_var, "锁定 避免内容（negative prompt / 反 AI tells）"),
        )
        for index, (text, var, tip) in enumerate(lock_specs, start=1):
            chk = ttk.Checkbutton(polish_row3, text=text, variable=var)
            chk.grid(row=0, column=index, sticky="w", padx=(0, 6))
            _Tooltip(chk, tip)

        # 润色预设：5 个方向下拉（服装/场景/风格/构图/姿态）。
        # 每个下拉都自带「不改动」选项 — 选中后把对应方面冻结在原文。
        polish_presets = ttk.LabelFrame(frame, text="润色预设")
        polish_presets.grid(
            row=10, column=0, columnspan=2, sticky="ew", padx=8, pady=(4, 6),
        )
        for col in range(6):
            polish_presets.columnconfigure(col, weight=1 if col % 2 else 0)

        preset_rows = (
            (0, 0, "服装", self.wardrobe_preset_var, WARDROBE_PRESET_KEYS),
            (0, 2, "场景", self.scene_preset_var, SCENE_PRESET_KEYS),
            (0, 4, "风格", self.shooting_style_preset_var, SHOOTING_STYLE_PRESET_KEYS),
            (1, 0, "构图", self.framing_preset_var, FRAMING_PRESET_KEYS),
            (1, 2, "姿态", self.pose_preset_var, POSE_PRESET_KEYS),
        )
        for row, col, label, var, keys in preset_rows:
            ttk.Label(polish_presets, text=label).grid(
                row=row, column=col, sticky="w", padx=(8 if col == 0 else 4, 4),
                pady=4,
            )
            ttk.Combobox(
                polish_presets, textvariable=var,
                values=tuple(keys), state="readonly", width=14,
            ).grid(row=row, column=col + 1, sticky="ew", padx=(0, 4), pady=4)

        # 润色服务商：跟随图像 (默认) 或单独配 deepseek / openai 等。
        # 选 "跟随图像" 时下方 base_url 输入被忽略。
        ttk.Label(polish_presets, text="润色服务商").grid(
            row=2, column=0, sticky="w", padx=(8, 4), pady=4,
        )
        ttk.Combobox(
            polish_presets, textvariable=self.polish_provider_var,
            values=POLISH_PROVIDER_LABELS, state="readonly", width=20,
        ).grid(row=2, column=1, columnspan=3, sticky="ew", padx=(0, 4), pady=4)
        ttk.Label(polish_presets, text="Base URL").grid(
            row=2, column=4, sticky="w", padx=(4, 4), pady=4,
        )
        ttk.Entry(
            polish_presets, textvariable=self.polish_base_url_var,
        ).grid(row=2, column=5, sticky="ew", padx=(0, 4), pady=4)

        # 润色模型 —— Combobox 可下拉选预设，也可手输任意模型名。
        # 切换润色服务商时会自动填成该 provider 的默认模型。
        ttk.Label(polish_presets, text="润色模型").grid(
            row=3, column=0, sticky="w", padx=(8, 4), pady=4,
        )
        self.polish_model_combo = ttk.Combobox(
            polish_presets, textvariable=self.polish_model_var,
            values=self._polish_model_dropdown_values(), width=20,
        )
        self.polish_model_combo.grid(
            row=3, column=1, columnspan=3, sticky="ew", padx=(0, 4), pady=4,
        )
        model_btns = ttk.Frame(polish_presets)
        model_btns.grid(row=3, column=4, columnspan=2, sticky="ew", padx=(4, 4), pady=4)
        model_btns.columnconfigure(0, weight=1)
        model_btns.columnconfigure(1, weight=1)
        ttk.Button(
            model_btns, text="+ 保存", command=self._save_polish_model_preset,
        ).grid(row=0, column=0, sticky="ew", padx=(0, 4))
        ttk.Button(
            model_btns, text="× 删除", command=self._delete_polish_model_preset,
        ).grid(row=0, column=1, sticky="ew")

        # 润色补充自由文本（最后才追加，会覆盖前面的强度/底子）
        ttk.Label(frame, text="润色补充").grid(row=11, column=0, sticky="nw", padx=8, pady=(6, 6))
        self.extra_polish_rules = tk.Text(frame, height=3, wrap="word", undo=True)
        self.extra_polish_rules.grid(
            row=11, column=1, sticky="nsew", padx=(0, 8), pady=(6, 6),
        )
        return frame

    def _add_text_row(
        self, parent: ttk.LabelFrame, row: int, label: str, height: int,
        *, tooltip: str | None = None,
    ) -> tk.Text:
        lbl = ttk.Label(parent, text=label)
        lbl.grid(row=row, column=0, sticky="nw", padx=8, pady=6)
        if tooltip:
            _Tooltip(lbl, tooltip)
        widget = tk.Text(parent, height=height, wrap="word", undo=True)
        widget.grid(row=row, column=1, sticky="nsew", padx=(0, 8), pady=6)
        widget.bind("<KeyRelease>", lambda _event: self._refresh_command_preview())
        return widget

    def _build_settings_pane(self, parent: ttk.Frame) -> ttk.LabelFrame:
        frame = ttk.LabelFrame(parent, text="参数")
        for col in range(4):
            frame.columnconfigure(col, weight=1)

        ttk.Label(frame, text="数量").grid(row=0, column=0, sticky="w", padx=8, pady=6)
        ttk.Spinbox(frame, from_=1, to=10, textvariable=self.n_var, width=8).grid(
            row=0, column=1, sticky="ew", padx=(0, 8), pady=6
        )
        ttk.Label(frame, text="质量").grid(row=0, column=2, sticky="w", padx=8, pady=6)
        ttk.Combobox(frame, textvariable=self.quality_var,
                     values=("auto", "low", "medium", "high"),
                     state="readonly", width=10).grid(row=0, column=3, sticky="ew", padx=(0, 8), pady=6)

        ttk.Label(frame, text="尺寸").grid(row=1, column=0, sticky="w", padx=8, pady=6)
        ttk.Combobox(frame, textvariable=self.size_preset_var,
                     values=SIZE_LABELS, state="readonly"
                     ).grid(row=1, column=1, columnspan=2, sticky="ew", padx=(0, 8), pady=6)
        self.custom_size_entry = ttk.Entry(frame, textvariable=self.custom_size_var, width=14)
        self.custom_size_entry.grid(row=1, column=3, sticky="ew", padx=(0, 8), pady=6)

        ttk.Label(frame, text="方向").grid(row=2, column=0, sticky="w", padx=8, pady=6)
        self.orientation_combo = ttk.Combobox(
            frame, textvariable=self.orientation_var,
            values=("真正自动", "竖图", "横图", "方图"), state="readonly",
        )
        self.orientation_combo.grid(row=2, column=1, columnspan=3, sticky="ew", padx=(0, 8), pady=6)

        ttk.Label(frame, text="格式").grid(row=3, column=0, sticky="w", padx=8, pady=6)
        ttk.Combobox(frame, textvariable=self.format_var,
                     values=("png", "webp"), state="readonly", width=10
                     ).grid(row=3, column=1, sticky="ew", padx=(0, 8), pady=6)
        ttk.Label(frame, text="压缩").grid(row=3, column=2, sticky="w", padx=8, pady=6)
        self.compression_spinbox = ttk.Spinbox(
            frame, from_=0, to=100, textvariable=self.compression_var, width=8
        )
        self.compression_spinbox.grid(row=3, column=3, sticky="ew", padx=(0, 8), pady=6)

        ttk.Label(frame, text="审核").grid(row=4, column=0, sticky="w", padx=8, pady=6)
        ttk.Combobox(frame, textvariable=self.moderation_var,
                     values=("auto", "low"), state="readonly", width=10
                     ).grid(row=4, column=1, sticky="ew", padx=(0, 8), pady=6)
        ttk.Label(frame, text="动作").grid(row=4, column=2, sticky="w", padx=8, pady=6)
        ttk.Combobox(frame, textvariable=self.action_var,
                     values=("generate",), state="disabled", width=10
                     ).grid(row=4, column=3, sticky="ew", padx=(0, 8), pady=6)

        ttk.Label(frame, text="模型").grid(row=5, column=0, sticky="w", padx=8, pady=6)
        ttk.Entry(frame, textvariable=self.model_var).grid(
            row=5, column=1, columnspan=3, sticky="ew", padx=(0, 8), pady=6
        )

        # 润色模型已移到下方"润色预设"区，方便和润色服务商联动调整。
        ttk.Label(frame, text="目标语言").grid(row=6, column=0, sticky="w", padx=8, pady=6)
        ttk.Combobox(frame, textvariable=self.target_language_var,
                     values=("English", "简体中文", "繁體中文", "日本語", "한국어", "保持原文")
                     ).grid(row=6, column=1, columnspan=3, sticky="ew", padx=(0, 8), pady=6)

        ttk.Label(frame, text="预览帧").grid(row=7, column=0, sticky="w", padx=8, pady=6)
        ttk.Spinbox(frame, from_=0, to=3, textvariable=self.partial_images_var, width=8
                    ).grid(row=7, column=1, sticky="ew", padx=(0, 8), pady=6)
        ttk.Checkbutton(frame, text="流式 (SSE)", variable=self.stream_var
                        ).grid(row=7, column=2, columnspan=2, sticky="w", padx=8, pady=6)

        ttk.Label(frame, text="输出目录").grid(row=8, column=0, sticky="w", padx=8, pady=6)
        out_row = ttk.Frame(frame)
        out_row.grid(row=8, column=1, columnspan=3, sticky="ew", padx=(0, 8), pady=6)
        out_row.columnconfigure(0, weight=1)
        ttk.Entry(out_row, textvariable=self.out_dir_var).grid(row=0, column=0, sticky="ew", padx=(0, 6))
        ttk.Button(out_row, text="选择", command=self._pick_out_dir).grid(row=0, column=1)

        ttk.Label(frame, text="重试").grid(row=9, column=0, sticky="w", padx=8, pady=6)
        ttk.Spinbox(frame, from_=1, to=20, textvariable=self.retries_var, width=8
                    ).grid(row=9, column=1, sticky="ew", padx=(0, 8), pady=6)
        ttk.Label(frame, text="超时秒").grid(row=9, column=2, sticky="w", padx=8, pady=6)
        ttk.Spinbox(frame, from_=30, to=3600, increment=30,
                    textvariable=self.timeout_var, width=8
                    ).grid(row=9, column=3, sticky="ew", padx=(0, 8), pady=6)

        ttk.Label(frame, text="并发(0自动)").grid(row=10, column=0, sticky="w", padx=8, pady=6)
        ttk.Spinbox(frame, from_=0, to=10, textvariable=self.max_concurrency_var, width=8
                    ).grid(row=10, column=1, sticky="ew", padx=(0, 8), pady=6)

        ttk.Label(frame, text="中转站").grid(row=11, column=0, sticky="w", padx=8, pady=6)
        self.provider_combo = ttk.Combobox(
            frame, textvariable=self.provider_var,
            values=IMAGE_PROVIDER_LABELS, state="readonly",
        )
        self.provider_combo.grid(row=11, column=1, columnspan=3, sticky="ew", padx=(0, 8), pady=6)

        ttk.Label(frame, text="Base URL").grid(row=12, column=0, sticky="w", padx=8, pady=6)
        self.base_url_entry = ttk.Entry(frame, textvariable=self.base_url_var)
        self.base_url_entry.grid(row=12, column=1, columnspan=3, sticky="ew", padx=(0, 8), pady=6)

        ttk.Label(frame, text="参数名").grid(row=13, column=0, sticky="w", padx=8, pady=6)
        self.profile_combo = ttk.Combobox(
            frame, textvariable=self.profile_name_var, values=self._profile_names,
        )
        self.profile_combo.grid(row=13, column=1, columnspan=3, sticky="ew", padx=(0, 8), pady=6)
        self.profile_combo.bind("<<ComboboxSelected>>",
                                lambda _event: self._load_profile(quiet=False))

        ttk.Label(frame, text="备注").grid(row=14, column=0, sticky="w", padx=8, pady=6)
        ttk.Entry(frame, textvariable=self.profile_note_var).grid(
            row=14, column=1, columnspan=3, sticky="ew", padx=(0, 8), pady=6
        )
        return frame

    def _build_refs_pane(self, parent: ttk.Frame) -> ttk.LabelFrame:
        frame = ttk.LabelFrame(parent, text="参考图")
        frame.columnconfigure(0, weight=1)
        frame.rowconfigure(0, weight=1)

        list_row = ttk.Frame(frame)
        list_row.grid(row=0, column=0, sticky="nsew", padx=8, pady=8)
        list_row.columnconfigure(0, weight=1)
        list_row.rowconfigure(0, weight=1)
        self.refs_listbox = tk.Listbox(list_row, height=12)
        self.refs_listbox.grid(row=0, column=0, sticky="nsew")
        scroll = ttk.Scrollbar(list_row, orient="vertical", command=self.refs_listbox.yview)
        scroll.grid(row=0, column=1, sticky="ns")
        self.refs_listbox.configure(yscrollcommand=scroll.set)

        buttons = ttk.Frame(frame)
        buttons.grid(row=1, column=0, sticky="ew", padx=8, pady=(0, 8))
        for col in range(4):
            buttons.columnconfigure(col, weight=1)
        ttk.Button(buttons, text="添加文件", command=self._pick_ref_files
                   ).grid(row=0, column=0, sticky="ew", padx=(0, 4))
        ttk.Button(buttons, text="添加URL", command=self._add_ref_url
                   ).grid(row=0, column=1, sticky="ew", padx=4)
        ttk.Button(buttons, text="删除", command=self._remove_selected_ref
                   ).grid(row=0, column=2, sticky="ew", padx=4)
        ttk.Button(buttons, text="清空", command=self._clear_refs
                   ).grid(row=0, column=3, sticky="ew", padx=(4, 0))

        edge_row = ttk.Frame(frame)
        edge_row.grid(row=2, column=0, sticky="ew", padx=8, pady=(0, 8))
        edge_row.columnconfigure(1, weight=1)
        ttk.Label(edge_row, text="上传前长边").grid(row=0, column=0, sticky="w")
        ttk.Combobox(edge_row, textvariable=self.ref_max_edge_var,
                     values=REF_MAX_EDGE_LABELS, state="readonly"
                     ).grid(row=0, column=1, sticky="ew", padx=(6, 0))
        return frame

    def _build_actions_pane(self, parent: ttk.Frame) -> ttk.LabelFrame:
        frame = ttk.LabelFrame(parent, text="命令")
        frame.columnconfigure(0, weight=1)

        self.command_text = tk.Text(frame, height=3, wrap="word")
        self.command_text.grid(row=0, column=0, sticky="ew", padx=8, pady=8)
        self.command_text.configure(state="disabled")

        buttons = ttk.Frame(frame)
        buttons.grid(row=1, column=0, sticky="ew", padx=8, pady=(0, 8))
        for col in range(9):
            buttons.columnconfigure(col, weight=1)

        self.start_button = ttk.Button(buttons, text="开始生成", command=self._start_generation)
        self.start_button.grid(row=0, column=0, sticky="ew", padx=(0, 4))
        self.stop_button = ttk.Button(buttons, text="终止",
                                      command=self._stop_generation, state="disabled")
        self.stop_button.grid(row=0, column=1, sticky="ew", padx=4)
        self.dry_run_button = ttk.Button(buttons, text="Dry-run", command=self._dry_run)
        self.dry_run_button.grid(row=0, column=2, sticky="ew", padx=4)
        ttk.Button(buttons, text="保存参数", command=self._save_profile
                   ).grid(row=0, column=3, sticky="ew", padx=4)
        ttk.Button(buttons, text="另存为", command=self._save_profile_as
                   ).grid(row=0, column=4, sticky="ew", padx=4)
        ttk.Button(buttons, text="加载参数", command=lambda: self._load_profile(quiet=False)
                   ).grid(row=0, column=5, sticky="ew", padx=4)
        ttk.Button(buttons, text="删除参数", command=self._delete_profile
                   ).grid(row=0, column=6, sticky="ew", padx=4)
        ttk.Button(buttons, text="复制命令", command=self._copy_command
                   ).grid(row=0, column=7, sticky="ew", padx=4)
        ttk.Button(buttons, text="打开输出目录", command=self._open_out_dir
                   ).grid(row=0, column=8, sticky="ew", padx=(4, 0))

        ttk.Label(frame, textvariable=self.status_var).grid(
            row=2, column=0, sticky="w", padx=8, pady=(0, 8)
        )
        return frame

    def _build_log_pane(self, parent: ttk.Frame) -> ttk.LabelFrame:
        frame = ttk.LabelFrame(parent, text="日志")
        frame.columnconfigure(0, weight=1)
        frame.rowconfigure(0, weight=1)

        self.log_text = tk.Text(frame, height=10, wrap="word", state="disabled")
        self.log_text.grid(row=0, column=0, sticky="nsew", padx=(8, 0), pady=8)
        scroll = ttk.Scrollbar(frame, orient="vertical", command=self.log_text.yview)
        scroll.grid(row=0, column=1, sticky="ns", padx=(0, 8), pady=8)
        self.log_text.configure(yscrollcommand=scroll.set)
        return frame

    # ── reactive wiring ─────────────────────────────────────────────────────

    def _wire_traces(self) -> None:
        watched = (
            self.prompt_file_var, self.n_var, self.quality_var, self.size_preset_var,
            self.custom_size_var, self.orientation_var, self.format_var,
            self.compression_var, self.moderation_var, self.action_var,
            self.partial_images_var, self.stream_var, self.out_dir_var,
            self.model_var, self.polish_model_var, self.target_language_var,
            self.retries_var, self.timeout_var, self.max_concurrency_var,
        )
        for var in watched:
            var.trace_add("write", lambda *_args: self._refresh_command_preview())

        self.size_preset_var.trace_add("write", lambda *_args: self._refresh_size_controls())
        self.format_var.trace_add("write", lambda *_args: self._refresh_format_controls())
        self.size_preset_var.trace_add("write", lambda *_args: self._refresh_orientation_controls())
        self.provider_var.trace_add("write", lambda *_args: self._refresh_provider_controls())
        self.provider_var.trace_add("write", lambda *_args: self._refresh_command_preview())
        self.provider_var.trace_add(
            "write", lambda *_args: self._refresh_polish_provider_controls()
        )
        self.polish_provider_var.trace_add(
            "write", lambda *_args: self._refresh_polish_provider_controls()
        )
        self.base_url_var.trace_add("write", lambda *_args: self._refresh_command_preview())
        self.ref_max_edge_var.trace_add("write", lambda *_args: self._refresh_command_preview())

    def _current_provider_key(self) -> str:
        return PROVIDER_BY_LABEL.get(self.provider_var.get(), DEFAULT_PROVIDER)

    def _current_polish_provider_key(self) -> str:
        """润色 provider key —— 跟随图像 时回退到主 provider。"""
        label = self.polish_provider_var.get()
        if label == POLISH_FOLLOW_LABEL:
            return self._current_provider_key()
        return PROVIDER_BY_LABEL.get(label, self._current_provider_key())

    def _current_polish_base_url(self) -> str | None:
        """润色 base_url —— 跟随图像 时复用主 base_url。"""
        label = self.polish_provider_var.get()
        if label == POLISH_FOLLOW_LABEL:
            return self.base_url_var.get().strip() or None
        return self.polish_base_url_var.get().strip() or None

    def _current_ref_max_edge(self) -> int:
        return REF_MAX_EDGE_BY_LABEL.get(self.ref_max_edge_var.get(), REF_IMAGE_DEFAULT_MAX_EDGE)

    def _refresh_provider_controls(self) -> None:
        key = self._current_provider_key()
        cfg = PROVIDERS.get(key)
        preset_url = cfg.base_url if cfg else ""
        if key == "custom":
            self.base_url_entry.configure(state="normal")
        else:
            current = self.base_url_var.get().strip()
            known = {p.base_url for p in PROVIDERS.values()}
            if not current or current in known:
                self.base_url_var.set(preset_url)
            self.base_url_entry.configure(state="normal")

    # ── polish provider auto-fill + model presets ───────────────────────────

    def _polish_model_dropdown_values(self) -> tuple[str, ...]:
        """合并：当前 polish provider 的 text_models + 全局 + 用户自存。"""
        cfg = PROVIDERS.get(self._current_polish_provider_key())
        primary = list(cfg.text_models) if cfg else []
        seen: list[str] = []
        for source in (primary, list(POLISH_MODEL_PRESETS), self._polish_model_custom_presets):
            for model in source:
                if model and model not in seen:
                    seen.append(model)
        return tuple(seen)

    def _refresh_polish_model_combo_values(self) -> None:
        if hasattr(self, "polish_model_combo"):
            self.polish_model_combo.configure(values=self._polish_model_dropdown_values())

    def _refresh_polish_provider_controls(self) -> None:
        """切换 polish provider 时：Base URL + 润色模型 同步预设。"""
        if self._suppress_polish_provider_sync:
            return
        label = self.polish_provider_var.get()
        if label == POLISH_FOLLOW_LABEL:
            # 跟随图像 → 清空独立 base_url（让 _current_polish_base_url 回落到主 base_url）
            self.polish_base_url_var.set("")
            cfg = PROVIDERS.get(self._current_provider_key())
        else:
            cfg = PROVIDERS.get(PROVIDER_BY_LABEL.get(label, ""))
            if cfg is not None and not cfg.is_custom:
                self.polish_base_url_var.set(cfg.base_url)
        if cfg is not None and cfg.default_text_model:
            self.polish_model_var.set(cfg.default_text_model)
        self._refresh_polish_model_combo_values()

    def _save_polish_model_preset(self) -> None:
        value = self.polish_model_var.get().strip()
        if not value:
            messagebox.showinfo("润色模型", "请先输入或选一个模型名再保存。")
            return
        if value in self._polish_model_custom_presets:
            self.status_var.set(f"已存在于自定义预设：{value}")
            return
        self._polish_model_custom_presets.append(value)
        self._refresh_polish_model_combo_values()
        persisted = self._persist_polish_custom_presets()
        if persisted:
            self.status_var.set(f"已保存自定义润色模型：{value}（已写盘）")
        else:
            self.status_var.set(f"已保存自定义润色模型：{value}（仅内存）")

    def _delete_polish_model_preset(self) -> None:
        value = self.polish_model_var.get().strip()
        if not value:
            return
        if value not in self._polish_model_custom_presets:
            messagebox.showinfo(
                "润色模型",
                f"「{value}」是内置预设，不能从下拉里删；只能删自己加的。",
            )
            return
        self._polish_model_custom_presets.remove(value)
        self._refresh_polish_model_combo_values()
        persisted = self._persist_polish_custom_presets()
        if persisted:
            self.status_var.set(f"已从自定义预设里移除：{value}（已写盘）")
        else:
            self.status_var.set(f"已从自定义预设里移除：{value}（仅内存）")

    def _persist_polish_custom_presets(self) -> bool:
        """把 self._polish_model_custom_presets 立即写盘到当前 active profile。
        只更新这一个字段 —— 不动用户其它未保存的编辑，也绕开 _save_profile
        的 dedupe / active-switching 副作用。

        返回 True 表示成功落盘。失败仅记日志不抛 —— 内存里的列表仍可用。
        """
        try:
            profile_name = self.profile_name_var.get().strip() or "Default"
            store = self._read_profile_store()
            profiles = store.setdefault("profiles", {})
            if not isinstance(profiles, dict):
                profiles = {}
                store["profiles"] = profiles
            entry = profiles.get(profile_name)
            if isinstance(entry, dict):
                settings = entry.get("settings")
                if not isinstance(settings, dict):
                    settings = {}
                    entry["settings"] = settings
                settings["polish_model_custom_presets"] = list(
                    self._polish_model_custom_presets
                )
            else:
                # active profile 还没建过：用当前完整 settings 建一份。
                # 这样下次启动能 load 到自定义模型，也确保 active 指向真存在的 profile。
                profiles[profile_name] = {
                    "note": self.profile_note_var.get().strip(),
                    "settings": self._profile_payload(),
                }
            store["active"] = profile_name
            self._write_profile_store(store)
            return True
        except Exception as exc:
            self._log(f"[warn] 自定义润色模型未能写盘：{exc}\n")
            return False

    # ── ref/file pickers ────────────────────────────────────────────────────

    def _pick_prompt_files(self) -> None:
        filenames = filedialog.askopenfilenames(
            title="选择提示词文件（可多选）",
            filetypes=(("Text files", "*.txt *.md"), ("All files", "*.*")),
        )
        if filenames:
            self.prompt_file_var.set(";".join(filenames))

    def _pick_out_dir(self) -> None:
        directory = filedialog.askdirectory(title="选择输出目录")
        if directory:
            self.out_dir_var.set(directory)

    def _pick_ref_files(self) -> None:
        filenames = filedialog.askopenfilenames(
            title="选择参考图",
            filetypes=(("Image files", "*.png *.jpg *.jpeg *.webp *.bmp"),
                       ("All files", "*.*")),
        )
        for filename in filenames:
            self._append_ref(filename)

    def _add_ref_url(self) -> None:
        value = simpledialog.askstring("添加URL", "参考图 URL 或 data URL")
        if value:
            self._append_ref(value.strip())

    def _append_ref(self, value: str) -> None:
        if value and value not in self._ref_paths:
            self._ref_paths.append(value)
            self.refs_listbox.insert("end", value)
            self._refresh_command_preview()

    def _remove_selected_ref(self) -> None:
        for index in reversed(list(self.refs_listbox.curselection())):
            self.refs_listbox.delete(index)
            del self._ref_paths[index]
        self._refresh_command_preview()

    def _clear_refs(self) -> None:
        self._ref_paths.clear()
        self.refs_listbox.delete(0, "end")
        self._refresh_command_preview()

    # ── small text helpers ──────────────────────────────────────────────────

    def _set_text(self, widget: tk.Text, value: str) -> None:
        widget.delete("1.0", "end")
        widget.insert("1.0", value or "")

    def _get_text(self, widget: tk.Text) -> str:
        return widget.get("1.0", "end").strip()

    # ── profile persistence ────────────────────────────────────────────────

    def _profile_payload(self) -> dict[str, object]:
        return {
            "general_prompt": self._get_text(self.general_prompt),
            "main_prompt": self._get_text(self.main_prompt),
            "person_prompt": self._get_text(self.person_prompt),
            "style_prompt": self._get_text(self.style_prompt),
            "scene_prompt": self._get_text(self.scene_prompt),
            "avoid_prompt": self._get_text(self.avoid_prompt),
            "prompt_file": self.prompt_file_var.get().strip(),
            "prompt_files": self._prompt_file_list(),
            "refs": list(self._ref_paths),
            "n": int(self.n_var.get()),
            "quality": self.quality_var.get(),
            "size_preset": self.size_preset_var.get(),
            "custom_size": self.custom_size_var.get(),
            "orientation": self.orientation_var.get(),
            "format": self.format_var.get(),
            "compression": int(self.compression_var.get()),
            "moderation": self.moderation_var.get(),
            "action": self.action_var.get(),
            "partial_images": int(self.partial_images_var.get()),
            "stream": bool(self.stream_var.get()),
            "ref_max_edge": self._current_ref_max_edge(),
            "out_dir": self.out_dir_var.get().strip(),
            "model": self.model_var.get().strip(),
            "polish_model": self.polish_model_var.get().strip(),
            "polish_model_custom_presets": list(self._polish_model_custom_presets),
            "target_language": self.target_language_var.get(),
            "retries": int(self.retries_var.get()),
            "timeout": int(self.timeout_var.get()),
            "max_concurrency": int(self.max_concurrency_var.get()),
            "provider": self._current_provider_key(),
            "base_url": self.base_url_var.get().strip(),
            "polish_provider": self.polish_provider_var.get(),
            "polish_base_url": self.polish_base_url_var.get().strip(),
            "intensity": self.intensity_var.get(),
            "scope": self.scope_var.get(),
            "polish_link_mode": self.polish_link_mode_var.get(),
            "wardrobe_preset": self.wardrobe_preset_var.get(),
            "scene_preset": self.scene_preset_var.get(),
            "shooting_style_preset": self.shooting_style_preset_var.get(),
            "framing_preset": self.framing_preset_var.get(),
            "pose_preset": self.pose_preset_var.get(),
            "extra_polish_rules": self._get_text(self.extra_polish_rules),
            "lock_general_prompt": bool(self.lock_general_prompt_var.get()),
            "lock_main_prompt": bool(self.lock_main_prompt_var.get()),
            "lock_person_prompt": bool(self.lock_person_prompt_var.get()),
            "lock_style_prompt": bool(self.lock_style_prompt_var.get()),
            "lock_scene_prompt": bool(self.lock_scene_prompt_var.get()),
            "lock_avoid_prompt": bool(self.lock_avoid_prompt_var.get()),
        }

    def _read_profile_store(self) -> dict[str, object]:
        if not PROFILES_PATH.is_file():
            return {"active": "Default", "profiles": {}}
        data = json.loads(PROFILES_PATH.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            raise ValueError("配置文件格式不正确。")
        if "profiles" in data and isinstance(data["profiles"], dict):
            return data
        # Compat: very old single-profile shape.
        return {
            "active": "Default",
            "profiles": {"Default": {"note": str(data.get("note", "")), "settings": data}},
        }

    def _write_profile_store(self, store: dict[str, object]) -> None:
        PROFILES_PATH.write_text(
            json.dumps(store, ensure_ascii=False, indent=2), encoding="utf-8",
        )

    def _signature(self, item: object) -> str:
        settings = item.get("settings", item) if isinstance(item, dict) else {}
        encoded = json.dumps(settings, ensure_ascii=False, sort_keys=True, default=str)
        return hashlib.sha256(encoded.encode("utf-8")).hexdigest()

    def _existing_duplicate(
        self, profiles: dict[str, object], settings: dict[str, object], skip: str
    ) -> str | None:
        target = self._signature({"settings": settings})
        for name, item in profiles.items():
            if name != skip and self._signature(item) == target:
                return name
        return None

    def _drop_duplicates(self, profiles: dict[str, object]) -> None:
        seen: dict[str, str] = {}
        for name in list(profiles.keys()):
            sig = self._signature(profiles[name])
            if sig in seen:
                del profiles[name]
            else:
                seen[sig] = name

    def _refresh_profile_dropdown(self, store: dict[str, object] | None = None) -> None:
        if store is None:
            try:
                store = self._read_profile_store()
            except Exception:
                store = {"profiles": {}}
        profiles = store.get("profiles", {})
        names = sorted(profiles.keys()) if isinstance(profiles, dict) else []
        self._profile_names = names or ["Default"]
        if hasattr(self, "profile_combo"):
            self.profile_combo.configure(values=self._profile_names)

    def _save_profile(self) -> None:
        try:
            profile_name = self.profile_name_var.get().strip() or "Default"
            store = self._read_profile_store()
            profiles = store.setdefault("profiles", {})
            if not isinstance(profiles, dict):
                raise ValueError("配置文件 profiles 不是 dict。")
            self._drop_duplicates(profiles)
            settings = self._profile_payload()
            dup = self._existing_duplicate(profiles, settings, profile_name)
            if dup:
                self.profile_name_var.set(dup)
                store["active"] = dup
                self._write_profile_store(store)
                self._refresh_profile_dropdown(store)
                self.status_var.set(f"相同参数已存在: {dup}")
                return
            profiles[profile_name] = {
                "note": self.profile_note_var.get().strip(),
                "settings": settings,
            }
            store["active"] = profile_name
            self._write_profile_store(store)
            self._refresh_profile_dropdown(store)
            self.profile_name_var.set(profile_name)
        except Exception as exc:
            messagebox.showerror("保存失败", str(exc))
            return
        self.status_var.set(f"参数已保存: {self.profile_name_var.get().strip() or 'Default'}")

    def _save_profile_as(self) -> None:
        current = self.profile_name_var.get().strip()
        new_name = simpledialog.askstring("另存为参数", "输入新的参数名", initialvalue=current)
        if not new_name:
            return
        self.profile_name_var.set(new_name.strip())
        self._save_profile()

    def _load_profile(self, quiet: bool = False) -> None:
        if not PROFILES_PATH.is_file():
            if not quiet:
                messagebox.showinfo("加载参数", f"未找到 {PROFILES_PATH.name}")
            return
        try:
            store = self._read_profile_store()
            self._refresh_profile_dropdown(store)
            profiles = store.get("profiles", {})
            if not isinstance(profiles, dict):
                raise ValueError("配置文件 profiles 不是 dict。")
            self._drop_duplicates(profiles)
            self._write_profile_store(store)
            self._refresh_profile_dropdown(store)
            target = self.profile_name_var.get().strip()
            if quiet:
                target = str(store.get("active") or target or "Default")
            target = target or "Default"
            if target not in profiles and profiles:
                target = next(iter(profiles))
            item = profiles.get(target)
            if not isinstance(item, dict):
                raise ValueError(f"找不到参数: {target}")
            settings = item.get("settings", item)
            if not isinstance(settings, dict):
                raise ValueError(f"参数 {target} 格式不正确。")
            self.profile_name_var.set(target)
            self.profile_note_var.set(str(item.get("note", "")))
            self._apply_profile(settings)
        except Exception as exc:
            if not quiet:
                messagebox.showerror("加载失败", str(exc))
            return
        if not quiet:
            self.status_var.set(f"参数已加载: {self.profile_name_var.get().strip() or 'Default'}")

    def _delete_profile(self) -> None:
        target = self.profile_name_var.get().strip()
        if not target:
            return
        try:
            store = self._read_profile_store()
            profiles = store.get("profiles", {})
            if not isinstance(profiles, dict) or target not in profiles:
                messagebox.showinfo("删除参数", f"找不到参数: {target}")
                return
            del profiles[target]
            store["active"] = next(iter(profiles), "Default")
            self._write_profile_store(store)
            self._refresh_profile_dropdown(store)
            self.profile_name_var.set(str(store["active"]))
            self.profile_note_var.set("")
        except Exception as exc:
            messagebox.showerror("删除失败", str(exc))
            return
        self.status_var.set(f"参数已删除: {target}")

    def _apply_profile(self, data: dict[str, object]) -> None:
        # 整个加载过程暂停 polish provider 自动覆盖 —— 不然主 provider_var
        # 一变，挂在它上面的 trace 会把刚还原的 polish_model 又改成默认值。
        self._suppress_polish_provider_sync = True
        try:
            self._apply_profile_inner(data)
        finally:
            self._suppress_polish_provider_sync = False
        self._refresh_polish_model_combo_values()

    def _apply_profile_inner(self, data: dict[str, object]) -> None:
        self._set_text(self.general_prompt, str(data.get("general_prompt", "")))
        self._set_text(self.main_prompt, str(data.get("main_prompt", "")))
        self._set_text(self.person_prompt, str(data.get("person_prompt", "")))
        self._set_text(self.style_prompt, str(data.get("style_prompt", "")))
        self._set_text(self.scene_prompt, str(data.get("scene_prompt", "")))
        self._set_text(self.avoid_prompt, str(data.get("avoid_prompt", "")))

        prompt_files = data.get("prompt_files", [])
        if isinstance(prompt_files, list) and prompt_files:
            self.prompt_file_var.set(";".join(str(p).strip() for p in prompt_files if str(p).strip()))
        else:
            self.prompt_file_var.set(str(data.get("prompt_file", "")))

        self.n_var.set(int(data.get("n", self.n_var.get())))
        quality = str(data.get("quality", self.quality_var.get())).strip().lower()
        if quality not in {"auto", "low", "medium", "high"}:
            quality = "auto"
        self.quality_var.set(quality)
        self.size_preset_var.set(str(data.get("size_preset", self.size_preset_var.get())))
        self.custom_size_var.set(str(data.get("custom_size", self.custom_size_var.get())))
        self.orientation_var.set(str(data.get("orientation", self.orientation_var.get())))
        fmt = str(data.get("format", self.format_var.get()))
        if fmt == "jpeg":
            fmt = "png"
        self.format_var.set(fmt)
        self.compression_var.set(int(data.get("compression", self.compression_var.get())))
        self.moderation_var.set(str(data.get("moderation", self.moderation_var.get())))
        self.action_var.set("generate")
        self.partial_images_var.set(int(data.get("partial_images", self.partial_images_var.get())))
        self.stream_var.set(bool(data.get("stream", True)))

        raw_edge = data.get("ref_max_edge", REF_IMAGE_DEFAULT_MAX_EDGE)
        try:
            edge_value = int(raw_edge)
        except (TypeError, ValueError):
            edge_value = REF_IMAGE_DEFAULT_MAX_EDGE
        self.ref_max_edge_var.set(
            REF_MAX_EDGE_LABEL_BY_VALUE.get(edge_value, REF_MAX_EDGE_LABELS[1])
        )

        self.out_dir_var.set(str(data.get("out_dir", self.out_dir_var.get())))
        self.model_var.set(str(data.get("model", self.model_var.get())))
        polish_model = str(data.get("polish_model", self.polish_model_var.get()))
        if polish_model == "gpt-5":
            polish_model = "gpt-5.5"
        # 用户自存的润色模型列表 —— 必须在 polish_model_var 之前 set 好，
        # 这样下面 _refresh_polish_model_combo_values 能把它们都列出来
        raw_custom = data.get("polish_model_custom_presets", [])
        if isinstance(raw_custom, list):
            self._polish_model_custom_presets = [
                str(item).strip() for item in raw_custom if str(item).strip()
            ]
        else:
            self._polish_model_custom_presets = []
        polish_provider_label = str(data.get("polish_provider", POLISH_FOLLOW_LABEL))
        if polish_provider_label not in POLISH_PROVIDER_LABELS:
            polish_provider_label = POLISH_FOLLOW_LABEL
        self.polish_provider_var.set(polish_provider_label)
        self.polish_base_url_var.set(str(data.get("polish_base_url", "")).strip())
        self.polish_model_var.set(polish_model)
        self.target_language_var.set(str(data.get("target_language", self.target_language_var.get())))
        self.retries_var.set(int(data.get("retries", max(5, self.retries_var.get()))))
        self.timeout_var.set(int(data.get("timeout", self.timeout_var.get())))
        self.max_concurrency_var.set(int(data.get("max_concurrency", self.max_concurrency_var.get())))

        provider_key = str(data.get("provider", self._current_provider_key())).strip().lower()
        if provider_key not in PROVIDERS:
            provider_key = DEFAULT_PROVIDER
        # 图像 provider 不能用 chat-completions 厂商（不支持出图）。
        if PROVIDERS[provider_key].text_api != "responses":
            provider_key = DEFAULT_PROVIDER
        self.provider_var.set(PROVIDER_LABEL_BY_KEY.get(provider_key, PROVIDER_LABELS[0]))
        self.base_url_var.set(str(data.get("base_url", "")).strip())

        self.intensity_var.set(str(data.get("intensity", "保守")))
        self.scope_var.set(str(data.get("scope", "全部")))
        link_raw = str(data.get("polish_link_mode", "关联"))
        self.polish_link_mode_var.set(link_raw if link_raw in ("关联", "独立") else "关联")
        preset = str(data.get("wardrobe_preset", DEFAULT_WARDROBE_PRESET))
        if preset not in WARDROBE_PRESET_KEYS:
            preset = DEFAULT_WARDROBE_PRESET
        self.wardrobe_preset_var.set(preset)
        for var, key, default, valid_keys in (
            (self.scene_preset_var, "scene_preset", DEFAULT_SCENE_PRESET, SCENE_PRESET_KEYS),
            (self.shooting_style_preset_var, "shooting_style_preset",
             DEFAULT_SHOOTING_STYLE_PRESET, SHOOTING_STYLE_PRESET_KEYS),
            (self.framing_preset_var, "framing_preset", DEFAULT_FRAMING_PRESET, FRAMING_PRESET_KEYS),
            (self.pose_preset_var, "pose_preset", DEFAULT_POSE_PRESET, POSE_PRESET_KEYS),
        ):
            value = str(data.get(key, default))
            if value not in valid_keys:
                value = default
            var.set(value)
        self._set_text(self.extra_polish_rules, str(data.get("extra_polish_rules", "")))
        self.lock_general_prompt_var.set(bool(data.get("lock_general_prompt", False)))
        self.lock_main_prompt_var.set(bool(data.get("lock_main_prompt", False)))
        self.lock_person_prompt_var.set(bool(data.get("lock_person_prompt", False)))
        self.lock_style_prompt_var.set(bool(data.get("lock_style_prompt", False)))
        self.lock_scene_prompt_var.set(bool(data.get("lock_scene_prompt", False)))
        self.lock_avoid_prompt_var.set(bool(data.get("lock_avoid_prompt", False)))

        refs = data.get("refs", [])
        self._ref_paths = [str(item) for item in refs if str(item).strip()] if isinstance(refs, list) else []
        self.refs_listbox.delete(0, "end")
        for ref in self._ref_paths:
            self.refs_listbox.insert("end", ref)

        self._refresh_size_controls()
        self._refresh_orientation_controls()
        self._refresh_format_controls()
        self._refresh_provider_controls()
        self._refresh_command_preview()

    # ── prompt composition ─────────────────────────────────────────────────

    def _composed_prompt(self, include_file: bool = True) -> str:
        fields = self._collect_prompt_fields(include_file=include_file)
        fields = dedupe_prompt_fields(fields, target_language=self.target_language_var.get())
        parts: list[str] = []
        if fields["general_prompt"]:
            parts.append(fields["general_prompt"])
        labels = self._section_labels()
        for key in ("main_prompt", "person_prompt", "style_prompt", "scene_prompt", "avoid_prompt"):
            value = fields.get(key, "")
            if value:
                parts.append(f"{labels[key]}: {value}")
        orientation = self._orientation_hint()
        if orientation:
            parts.append(orientation)
        return "\n".join(parts).strip()

    def _collect_prompt_fields(self, include_file: bool = True) -> dict[str, str]:
        fields = {
            "general_prompt": self._get_text(self.general_prompt),
            "main_prompt": self._get_text(self.main_prompt),
            "person_prompt": self._get_text(self.person_prompt),
            "style_prompt": self._get_text(self.style_prompt),
            "scene_prompt": self._get_text(self.scene_prompt),
            "avoid_prompt": self._get_text(self.avoid_prompt),
        }
        if include_file:
            file_chunks: list[str] = []
            for entry in self._prompt_file_list():
                path = Path(entry).expanduser()
                if not path.is_file():
                    raise ValueError(f"提示词文件不存在: {entry}")
                text = read_text_file(path).strip()
                if text:
                    file_chunks.append(text)
            combined = "\n".join(file_chunks).strip()
            if combined:
                fields["general_prompt"] = (
                    f"{fields['general_prompt']}\n{combined}".strip()
                    if fields["general_prompt"] else combined
                )
        return fields

    def _prompt_file_list(self) -> list[str]:
        raw = self.prompt_file_var.get().strip()
        if not raw:
            return []
        return [part.strip() for part in raw.split(";") if part.strip()]

    def _section_labels(self) -> dict[str, str]:
        language = self.target_language_var.get()
        labels_by_lang = {
            "English": ("Subject", "Person details", "Shooting style", "Scene", "Avoid"),
            "繁體中文": ("主體", "人物參數細節", "拍攝風格", "場景", "避免內容"),
            "日本語": ("主体", "人物詳細", "撮影スタイル", "シーン", "避ける内容"),
            "한국어": ("주제", "인물 세부", "촬영 스타일", "장면", "피할 내용"),
        }
        labels = labels_by_lang.get(language)
        if not labels:
            labels = ("主体", "人物参数细节", "拍摄风格", "场景", "避免内容")
        keys = ("main_prompt", "person_prompt", "style_prompt", "scene_prompt", "avoid_prompt")
        return dict(zip(keys, labels))

    def _orientation_hint(self) -> str:
        if self._resolved_size() != "auto":
            return ""
        orientation = self.orientation_var.get()
        language = self.target_language_var.get()
        en_map = {
            "横图": "Composition requirement: landscape horizontal image, wider than tall.",
            "竖图": "Composition requirement: portrait vertical image, taller than wide.",
            "方图": "Composition requirement: square image, 1:1 aspect ratio.",
        }
        cn_map = {
            "横图": "构图要求：横向画面，宽大于高。",
            "竖图": "构图要求：竖向画面，高大于宽。",
            "方图": "构图要求：方形画面，1:1 比例。",
        }
        tw_map = {
            "横图": "構圖要求：橫向畫面，寬大於高。",
            "竖图": "構圖要求：直向畫面，高大於寬。",
            "方图": "構圖要求：方形畫面，1:1 比例。",
        }
        jp_map = {
            "横图": "構図要件：横長の画像、幅が高さより大きい。",
            "竖图": "構図要件：縦長の画像、高さが幅より大きい。",
            "方图": "構図要件：正方形の画像、1:1 比率。",
        }
        kr_map = {
            "横图": "구도 요구사항: 가로형 이미지, 너비가 높이보다 큼.",
            "竖图": "구도 요구사항: 세로형 이미지, 높이가 너비보다 큼.",
            "方图": "구도 요구사항: 정사각형 이미지, 1:1 비율.",
        }
        table = {
            "English": en_map, "简体中文": cn_map, "繁體中文": tw_map,
            "日本語": jp_map, "한국어": kr_map,
        }.get(language, cn_map)
        return table.get(orientation, "")

    def _resolved_size(self) -> str:
        selected = SIZE_BY_LABEL.get(self.size_preset_var.get(), "auto")
        if selected == "custom":
            selected = self.custom_size_var.get().strip()
        return image_size(selected)

    def _gather_options(self, include_file_prompt: bool = True) -> dict[str, object]:
        prompt = self._composed_prompt(include_file=include_file_prompt)
        if not prompt:
            raise ValueError("提示词不能为空。")

        n = int(self.n_var.get())
        if not 1 <= n <= 10:
            raise ValueError("数量必须在 1 到 10 之间。")
        retries = int(self.retries_var.get())
        timeout = int(self.timeout_var.get())
        compression = int(self.compression_var.get())
        partial = int(self.partial_images_var.get())
        concurrency = int(self.max_concurrency_var.get())
        if retries < 1:
            raise ValueError("重试次数必须大于 0。")
        if timeout < 1:
            raise ValueError("超时时间必须大于 0。")
        if not 0 <= compression <= 100:
            raise ValueError("压缩值必须在 0 到 100 之间。")
        if not 0 <= partial <= 3:
            raise ValueError("预览帧必须在 0 到 3 之间。")
        if not 0 <= concurrency <= 10:
            raise ValueError("并发必须在 0 到 10 之间，0 表示自动。")

        return {
            "prompt": prompt,
            "n": n,
            "quality": self.quality_var.get(),
            "size": self._resolved_size(),
            "output_format": self.format_var.get(),
            "ref_images": [
                image_to_data_url(path, max_edge=self._current_ref_max_edge())
                for path in self._ref_paths
            ],
            "out_dir": self.out_dir_var.get().strip() or str(APP_DIR / "output"),
            "model": self.model_var.get().strip() or "gpt-5.5",
            "retries": retries,
            "timeout": timeout,
            "background": "auto",
            "output_compression": compression if self.format_var.get() == "webp" else None,
            "moderation": self.moderation_var.get(),
            "action": self.action_var.get(),
            "partial_images": partial,
            "max_concurrency": concurrency,
            "provider": self._current_provider_key(),
            "base_url": self.base_url_var.get().strip() or None,
            "stream": bool(self.stream_var.get()),
        }

    def _build_cli_command(self) -> str:
        args: list[str] = ["python", str(CLI_SCRIPT)]
        prompt = self._composed_prompt(include_file=False)
        if prompt:
            args.append(prompt)
        for prompt_file in self._prompt_file_list():
            args.extend(["--prompt-file", prompt_file])
        args.extend(["-n", str(self.n_var.get())])
        args.extend(["-q", self.quality_var.get()])
        args.extend(["-s", self._resolved_size()])
        args.extend(["-f", self.format_var.get()])
        for ref in self._ref_paths:
            args.extend(["-r", ref])
        args.extend(["-o", self.out_dir_var.get().strip() or str(APP_DIR / "output")])
        args.extend(["-m", self.model_var.get().strip() or "gpt-5.5"])
        args.extend(["--retries", str(self.retries_var.get())])
        args.extend(["--timeout", str(self.timeout_var.get())])
        args.extend(["--moderation", self.moderation_var.get()])
        args.extend(["--action", self.action_var.get()])
        args.extend(["--partial-images", str(self.partial_images_var.get())])
        args.extend(["--max-concurrency", str(self.max_concurrency_var.get())])
        if self.format_var.get() == "webp":
            args.extend(["--compression", str(self.compression_var.get())])
        provider_key = self._current_provider_key()
        if provider_key != DEFAULT_PROVIDER:
            args.extend(["--provider", provider_key])
        base_url = self.base_url_var.get().strip()
        if base_url:
            args.extend(["--base-url", base_url])
        if not bool(self.stream_var.get()):
            args.append("--no-stream")
        edge = self._current_ref_max_edge()
        if edge != REF_IMAGE_DEFAULT_MAX_EDGE:
            args.extend(["--ref-max-edge", str(edge)])
        return subprocess.list2cmdline(args)

    # ── small UI refresh helpers ───────────────────────────────────────────

    def _refresh_size_controls(self) -> None:
        state = "normal" if SIZE_BY_LABEL.get(self.size_preset_var.get()) == "custom" else "disabled"
        self.custom_size_entry.configure(state=state)
        self._refresh_orientation_controls()

    def _refresh_format_controls(self) -> None:
        state = "normal" if self.format_var.get() == "webp" else "disabled"
        self.compression_spinbox.configure(state=state)

    def _refresh_orientation_controls(self) -> None:
        state = "readonly" if self._resolved_size() == "auto" else "disabled"
        self.orientation_combo.configure(state=state)

    def _refresh_command_preview(self) -> None:
        try:
            command = self._build_cli_command()
        except Exception as exc:
            command = f"参数待修正: {exc}"
        self.command_text.configure(state="normal")
        self.command_text.delete("1.0", "end")
        self.command_text.insert("1.0", command)
        self.command_text.configure(state="disabled")

    def _copy_command(self) -> None:
        command = self.command_text.get("1.0", "end").strip()
        self.clipboard_clear()
        self.clipboard_append(command)
        self.status_var.set("Command copied")

    # ── GPT-5.5 polish flow ─────────────────────────────────────────────────

    def _snapshot_prompt_fields(self) -> dict[str, str]:
        """6 个 prompt 字段 + 文件选择的当前快照，用于回退。"""
        return {
            "general_prompt": self._get_text(self.general_prompt),
            "main_prompt":    self._get_text(self.main_prompt),
            "person_prompt":  self._get_text(self.person_prompt),
            "style_prompt":   self._get_text(self.style_prompt),
            "scene_prompt":   self._get_text(self.scene_prompt),
            "avoid_prompt":   self._get_text(self.avoid_prompt),
            "prompt_file":    self.prompt_file_var.get(),
        }

    def _apply_prompt_snapshot(self, snap: dict[str, str]) -> None:
        self._set_text(self.general_prompt, snap.get("general_prompt", ""))
        self._set_text(self.main_prompt,    snap.get("main_prompt", ""))
        self._set_text(self.person_prompt,  snap.get("person_prompt", ""))
        self._set_text(self.style_prompt,   snap.get("style_prompt", ""))
        self._set_text(self.scene_prompt,   snap.get("scene_prompt", ""))
        self._set_text(self.avoid_prompt,   snap.get("avoid_prompt", ""))
        self.prompt_file_var.set(snap.get("prompt_file", ""))
        self._refresh_command_preview()

    def _undo_polish(self) -> None:
        if self._refining or not self._polish_history:
            return
        snap = self._polish_history.pop()
        self._apply_prompt_snapshot(snap)
        self._refresh_undo_button()
        remaining = len(self._polish_history)
        self.status_var.set(f"已回退（还可退 {remaining} 步）")
        self._log(f"↶ 已恢复到润色前的状态（还可退 {remaining} 步）\n")

    def _refresh_undo_button(self) -> None:
        if not hasattr(self, "undo_button"):
            return
        depth = len(self._polish_history)
        text = f"↶ 回退 ({depth})" if depth else "↶ 回退"
        state = "disabled" if (depth == 0 or self._refining) else "normal"
        self.undo_button.configure(text=text, state=state)

    def _start_polish(self, mode: str, variant_kind: str | None = None) -> None:
        if self._refining:
            return
        try:
            fields = self._collect_prompt_fields(include_file=True)
            if not any(value.strip() for value in fields.values()):
                raise ValueError("提示词不能为空。")
        except Exception as exc:
            messagebox.showerror("提示词错误", str(exc))
            return

        intensity = self._resolve_polish_intensity(mode, variant_kind)
        target_fields = self._resolve_polish_scope(intensity)
        locked_fields = self._resolve_polish_locks()
        link_mode = "independent" if self.polish_link_mode_var.get() == "独立" else "linked"

        # 快照当前 prompt 字段到 undo 栈；失败时由 _pump_events 弹掉
        self._polish_history.append(self._snapshot_prompt_fields())
        if len(self._polish_history) > self._polish_history_limit:
            self._polish_history = self._polish_history[-self._polish_history_limit:]

        self._refining = True
        self._refresh_undo_button()
        intensity_label = self._intensity_label(intensity)
        scope_label = self.scope_var.get() if intensity != "conservative" else "全部"
        variant_label = {"soft": " / 变体(保主体)", "wild": " / 脑洞(重写)"}.get(variant_kind or "", "")
        link_label = "独立" if link_mode == "independent" else "关联"
        self.status_var.set(
            f"GPT-5.5 {mode} ({intensity_label} / {scope_label} / {link_label}{variant_label})..."
        )
        self._log(
            f"\n--- GPT-5.5 prompt processing ({mode}, intensity={intensity}, "
            f"scope={scope_label}, link={link_mode}, variant={variant_kind}, "
            f"locked_fields={locked_fields}) ---\n"
        )
        threading.Thread(
            target=self._do_polish,
            args=(fields, mode, intensity, target_fields, locked_fields, variant_kind, link_mode),
            daemon=True,
        ).start()

    def _resolve_polish_intensity(self, mode: str, variant_kind: str | None = None) -> str:
        if mode == "translate":
            return "conservative"
        order = ("conservative", "open", "gacha", "aggressive", "unhinged")
        base_map = {"保守": "conservative", "开放": "open", "抽卡": "gacha", "激进": "aggressive", "暴走": "unhinged"}
        base = base_map.get(self.intensity_var.get(), "conservative")
        sel = order.index(base)
        if variant_kind == "wild":
            # 脑洞：滑块基础 +2 档，且最低保证 aggressive（避免"保守 + 脑洞"还是温和）
            target = max(sel + 2, order.index("aggressive"))
            return order[min(target, len(order) - 1)]
        if variant_kind == "soft":
            # 软变体：滑块基础 +1 档，最低 open
            target = max(sel + 1, order.index("open"))
            return order[min(target, len(order) - 1)]
        return base

    def _intensity_label(self, intensity: str) -> str:
        return {"conservative": "保守", "open": "开放", "gacha": "抽卡", "aggressive": "激进", "unhinged": "暴走"}.get(
            intensity, intensity
        )

    def _resolve_polish_scope(self, intensity: str) -> list[str] | None:
        if intensity == "conservative":
            return None
        scope_map = {
            "全部": None,
            "仅人物": ["main_prompt", "person_prompt"],
            "仅场景": ["scene_prompt"],
            "仅风格": ["style_prompt"],
            "人物+场景": ["main_prompt", "person_prompt", "scene_prompt"],
            "人物+风格": ["main_prompt", "person_prompt", "style_prompt"],
        }
        return scope_map.get(self.scope_var.get())

    def _resolve_polish_locks(self) -> list[str]:
        toggles = (
            ("general_prompt", self.lock_general_prompt_var),
            ("main_prompt", self.lock_main_prompt_var),
            ("person_prompt", self.lock_person_prompt_var),
            ("style_prompt", self.lock_style_prompt_var),
            ("scene_prompt", self.lock_scene_prompt_var),
            ("avoid_prompt", self.lock_avoid_prompt_var),
        )
        return [key for key, var in toggles if bool(var.get())]

    def _do_polish(
        self,
        fields: dict[str, str],
        mode: str,
        intensity: str,
        target_fields: list[str] | None,
        locked_fields: list[str],
        variant_kind: str | None = None,
        polish_link_mode: str = "linked",
    ) -> None:
        writer = _QueueWriter(self._events)
        try:
            with contextlib.redirect_stdout(writer), contextlib.redirect_stderr(writer):
                result = refine_prompt_fields_with_gpt5(
                    fields,
                    mode=mode,
                    model=self.polish_model_var.get().strip() or "gpt-5.5",
                    retries=max(1, int(self.retries_var.get())),
                    timeout=max(30, int(self.timeout_var.get())),
                    target_language=self.target_language_var.get(),
                    intensity=intensity,
                    target_fields=target_fields,
                    locked_fields=locked_fields,
                    provider=self._current_polish_provider_key(),
                    base_url=self._current_polish_base_url(),
                    wardrobe_preset=self.wardrobe_preset_var.get() or None,
                    extra_polish_rules=self._get_text(self.extra_polish_rules) or None,
                    scene_preset=self.scene_preset_var.get() or None,
                    shooting_style_preset=self.shooting_style_preset_var.get() or None,
                    framing_preset=self.framing_preset_var.get() or None,
                    pose_preset=self.pose_preset_var.get() or None,
                    variant_kind=variant_kind,
                    polish_link_mode=polish_link_mode,
                )
            self._events.put(("refine_done", result))
        except Exception as exc:
            self._events.put(("refine_error", str(exc)))

    def _apply_polished_fields(self, fields: dict[str, str]) -> None:
        cleaned = {k: repair_mojibake(str(v)) for k, v in fields.items()}
        self._set_text(self.general_prompt, cleaned.get("general_prompt", ""))
        self._set_text(self.main_prompt, cleaned.get("main_prompt", ""))
        self._set_text(self.person_prompt, cleaned.get("person_prompt", ""))
        self._set_text(self.style_prompt, cleaned.get("style_prompt", ""))
        self._set_text(self.scene_prompt, cleaned.get("scene_prompt", ""))
        self._set_text(self.avoid_prompt, cleaned.get("avoid_prompt", ""))
        self.prompt_file_var.set("")
        self._refresh_command_preview()

    def _apply_polished_prompt(self, text: str) -> None:
        self._set_text(self.general_prompt, repair_mojibake(text))
        self._set_text(self.main_prompt, "")
        self._set_text(self.person_prompt, "")
        self._set_text(self.style_prompt, "")
        self._set_text(self.scene_prompt, "")
        self._set_text(self.avoid_prompt, "")
        self.prompt_file_var.set("")
        self._refresh_command_preview()

    # ── dry-run / generation ───────────────────────────────────────────────

    def _dry_run(self) -> None:
        try:
            options = self._gather_options(include_file_prompt=True)
        except Exception as exc:
            messagebox.showerror("参数错误", str(exc))
            return
        preview = dict(options)
        preview["ref_count"] = len(preview.pop("ref_images"))
        self._log(json.dumps(preview, ensure_ascii=False, indent=2) + "\n")
        self.status_var.set("Dry-run OK")

    def _start_generation(self) -> None:
        if self._busy:
            return
        try:
            options = self._gather_options(include_file_prompt=True)
        except Exception as exc:
            messagebox.showerror("参数错误", str(exc))
            return

        self._cancel_event = threading.Event()
        self._http_session = requests.Session()
        options["cancel_event"] = self._cancel_event
        options["session"] = self._http_session

        self._busy = True
        self.start_button.configure(state="disabled")
        self.dry_run_button.configure(state="disabled")
        self.stop_button.configure(state="normal")
        self.status_var.set("Generating...")
        self._log("\n--- Start generation ---\n")

        threading.Thread(target=self._do_generate, args=(options,), daemon=True).start()

    def _stop_generation(self) -> None:
        if not self._busy:
            return
        if self._cancel_event is not None:
            self._cancel_event.set()
        if self._http_session is not None:
            try:
                self._http_session.close()
            except Exception:
                pass
        self.stop_button.configure(state="disabled")
        self.status_var.set("正在终止...")
        self._log(
            "\n--- 终止请求已发送 ---\n"
            "已经落盘的图会全部保留。\n"
            "正在切断在飞的 HTTP 请求并等待 worker 收尾...\n"
        )

    def _do_generate(self, options: dict[str, object]) -> None:
        writer = _QueueWriter(self._events)
        try:
            with contextlib.redirect_stdout(writer), contextlib.redirect_stderr(writer):
                saved = generate(**options)
            kind = "cancelled" if (self._cancel_event and self._cancel_event.is_set()) else "done"
            self._events.put((kind, saved))
        except Exception as exc:
            self._events.put(("error", str(exc)))

    # ── event pump ──────────────────────────────────────────────────────────

    def _pump_events(self) -> None:
        try:
            while True:
                kind, payload = self._events.get_nowait()
                if kind == "log":
                    self._log(str(payload))
                elif kind == "done":
                    saved = payload if isinstance(payload, list) else []
                    self._log(f"完成：共保存 {len(saved)} 张\n")
                    for path in saved:
                        self._log(f"  · {path}\n")
                    self.status_var.set(f"完成：{len(saved)} 张")
                    self._mark_idle()
                elif kind == "cancelled":
                    saved = payload if isinstance(payload, list) else []
                    self._log(f"已终止：保留 {len(saved)} 张已落盘图像\n")
                    for path in saved:
                        self._log(f"  · {path}\n")
                    self.status_var.set(f"已终止：保留 {len(saved)} 张")
                    self._mark_idle()
                elif kind == "error":
                    self._log(f"错误：{payload}\n")
                    self.status_var.set("失败")
                    self._mark_idle()
                elif kind == "refine_done":
                    if isinstance(payload, dict):
                        self._apply_polished_fields(payload)
                    else:
                        self._apply_polished_prompt(str(payload))
                    self._log("GPT-5.5 prompt processing done.\n")
                    self.status_var.set("Prompt updated")
                    self._refining = False
                    self._refresh_undo_button()
                elif kind == "refine_error":
                    self._log(f"GPT-5.5 ERROR: {payload}\n")
                    self.status_var.set("GPT-5.5 prompt processing failed")
                    self._refining = False
                    # 调用失败 → 没有"润色后"状态，丢弃刚 push 的那一帧
                    if self._polish_history:
                        self._polish_history.pop()
                    self._refresh_undo_button()
        except queue.Empty:
            pass
        self.after(100, self._pump_events)

    def _mark_idle(self) -> None:
        self._busy = False
        self.start_button.configure(state="normal")
        self.dry_run_button.configure(state="normal")
        self.stop_button.configure(state="disabled")
        self._cancel_event = None
        if self._http_session is not None:
            try:
                self._http_session.close()
            except Exception:
                pass
            self._http_session = None

    def _log(self, text: str) -> None:
        self.log_text.configure(state="normal")
        self.log_text.insert("end", text)
        self.log_text.see("end")
        self.log_text.configure(state="disabled")

    def _open_out_dir(self) -> None:
        path = Path(self.out_dir_var.get().strip() or str(APP_DIR / "output"))
        path.mkdir(parents=True, exist_ok=True)
        if os.name == "nt":
            os.startfile(path)  # type: ignore[attr-defined]
        else:
            subprocess.Popen(["xdg-open", str(path)])


def main() -> None:
    configure_stdio()
    os.chdir(APP_DIR)
    _migrate_legacy_profiles()
    app = ImageWorkbench()
    app.mainloop()


if __name__ == "__main__":
    main()
