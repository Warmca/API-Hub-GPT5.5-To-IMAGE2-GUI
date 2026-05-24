# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 项目概览 / Overview

个人 Windows 工具，通过 OpenAI 兼容中转的 **Responses API** 调 `gpt-5.5` + `image_generation` 工具，产出 `gpt-image-2` 质量的图。用户主要走 GUI；CLI 是同一模块的另一个入口。**仅 Windows 使用，未做 Linux 适配。** 用户语言偏好中文。

Personal Windows tool. Calls OpenAI-compatible relays (freemodel.dev / OpenAI official / laozhang.ai / aimlapi / DeepSeek / Anthropic / xAI / GLM / Kimi / Qwen / 自定义) using the Responses API with the `image_generation` tool to produce gpt-image-2 output. GUI-first; the CLI is the same module's `main()`. Windows-only by design.

## 入口与日常命令 / Entry points

```bash
# GUI（日常）
python API_GPT5_5_to_IMAGE2_GUI.py

# CLI 单发
python API_GPT5_5_to_IMAGE2.py "your prompt" -q high -s 2160x3840 -n 2

# Dry run — 只打印解析后的参数和最终端点，不调 API
python API_GPT5_5_to_IMAGE2.py "anything" --dry-run

# 切换中转
python API_GPT5_5_to_IMAGE2.py "..." --provider openai
python API_GPT5_5_to_IMAGE2.py "..." --provider custom --base-url https://api.example.com/v1

# 关 SSE 流（调试用；多数中转必须开 SSE 才能拿到 image_generation 输出）
python API_GPT5_5_to_IMAGE2.py "..." --no-stream
```

启动前必须在 `.env` 写 token。模板见 `.env.example`。默认 provider 是 freemodel，读取 `FREEMODEL_TOKEN` / `FREEMODEL_TOKEN_2..10` / `FREEMODEL_TOKENS`（逗号、空格、分号分隔均可）。

无构建步骤、无 lint、无测试套件——纯 Python 3.10+，依赖 `requests` + `Pillow`（可选）+ `python-dotenv`（可选）。

## 架构 / Architecture

两个文件：

- **`API_GPT5_5_to_IMAGE2.py`** — 后端 + CLI 入口，约 2700 行，按 16 个分区组织（① 常量/provider ② I/O ③ .env/token ④ 文本工具 ⑤ SSE 解析 ⑥ 取消重试 ⑦ slug ⑧ 参考图压缩 ⑨ 尺寸校验 ⑩ payload ⑪ HTTP 调用 ⑫ `generate` 主入口 ⑬ 文本模型调用 ⑭ 润色 ⑮ argparse ⑯ CLI main）。
- **`API_GPT5_5_to_IMAGE2_GUI.py`** — Tkinter GUI，约 1900 行，单 class `ImageWorkbench(tk.Tk)`。从后端 import 这些公开名：
  - 核心 API：`generate`, `refine_prompt_fields_with_gpt5`, `image_to_data_url`, `image_size`, `dedupe_prompt_fields`, `read_text_file`, `repair_mojibake`, `configure_stdio`
  - 常量：`PROVIDERS`, `DEFAULT_PROVIDER`, `REF_IMAGE_DEFAULT_MAX_EDGE`, `POLISH_MODEL_PRESETS`
  - 5 个润色方向 preset：`WARDROBE_PRESET_KEYS`+`DEFAULT_WARDROBE_PRESET`, `SCENE_PRESET_KEYS`+`DEFAULT_SCENE_PRESET`, `SHOOTING_STYLE_PRESET_KEYS`+`DEFAULT_SHOOTING_STYLE_PRESET`, `FRAMING_PRESET_KEYS`+`DEFAULT_FRAMING_PRESET`, `POSE_PRESET_KEYS`+`DEFAULT_POSE_PRESET`

**改后端时保持这些公开名稳定，GUI 通常不用动。**

`freemodel_gen.py.bak` / `freemodel_gen_gui.py.bak` 是上一代版本，**只做回滚用，不要改**。`gui_profiles.json` 是用户的参数档案（旧名 `freemodel_gui_params.json`，首次启动自动迁移），不要重命名、不要手动改 schema。

## Provider 系统

`PROVIDERS: dict[str, ProviderConfig]` 当前注册 11 个 preset：

| Key | label | text_api | 用途 |
|---|---|---|---|
| `freemodel` | freemodel.dev | responses | 默认；图像 + 润色 |
| `openai` | OpenAI Official | responses | 图像 + 润色 |
| `laozhang` | laozhang.ai | responses | 图像 + 润色 |
| `aiml` | AI/ML API | responses | 图像 + 润色 |
| `deepseek` | DeepSeek | chat_completions | **仅润色** |
| `anthropic` | Anthropic Claude | anthropic_messages | **仅润色** |
| `xai` | xAI Grok | chat_completions | **仅润色** |
| `zhipu` | 智谱 GLM | chat_completions | **仅润色** |
| `moonshot` | 月之暗面 Kimi | chat_completions | **仅润色** |
| `dashscope` | 阿里 Qwen | chat_completions | **仅润色** |
| `custom` | 自定义 | responses | 任意，需手填 base_url |

`ProviderConfig` 是 `@dataclass(frozen=True)`，关键字段：
- `base_url` — provider 根路径，拼端点时自动加 `/responses` 或 `/chat/completions` 或 `/messages`。
- `env_keys` / `env_list_key` — 该 provider 的 token 环境变量名。
- `text_api` — 文本/润色走哪条 API：`"responses"`（OpenAI 新版）/ `"chat_completions"`（OpenAI 兼容老路径）/ `"anthropic_messages"`（Anthropic 专用，需 `x-api-key` + `anthropic-version` 头）。
- `fallback_to_freemodel` — 没找到本 provider 的 token 时是否兜底用 freemodel 的 env。**chat_completions / anthropic 类厂商必须为 False**（用错 token 比报错更糟）。
- `text_models` — GUI 下拉的可选模型清单；第 0 项是默认值，切换润色 provider 时自动填进"润色模型"框。

GUI 里图像 provider 下拉**只显示 `text_api == "responses"` 的**——chat_completions 类只能润色不能出图。润色 provider 下拉额外有"跟随图像"选项，选中时复用主 provider 和 base URL。

加新 provider：在 `PROVIDERS` 字典追加一项即可，GUI 下拉自动出现。`custom` provider 的 `base_url` 是空字符串，必须由调用方填 `--base-url` 或 GUI Base URL 框。

## 生成流程关键 / Generation flow

1. `generate()` 用 round-robin 把 n 张任务分给 token 列表；多 token 时进 `ThreadPoolExecutor`，并发数 = `min(token 数, max_concurrency, 任务数)`。
2. 每张图：`_post_image_request()` POST 一次，解析 SSE 或 JSON 提取 image b64。可重试状态码见 `TRANSIENT_HTTP = {408, 409, 425, 429, 500, 502, 503, 504, 520, 522, 524}`，退避用 `_backoff_seconds()`（基于尝试次数指数 + `Retry-After` 头 + jitter）。
3. 拿到 b64 立即 `save_image()`：**写 `.part` 临时文件 → `Path.replace` 原子重命名**。进程被强杀也不会留半张坏图。日志格式 `✓ 已落盘 [X/N]: 路径`。
4. **每张独立落盘，不批处理**。失败的某张不影响其它张。

## 取消机制 / Cancellation

GUI 终止按钮：
1. 主线程 `_stop_generation()` set `threading.Event` 并 `session.close()`
2. 关 session 让在飞的 `session.post()` 抛 `ConnectionError`
3. `_post_image_request` 异常处理查 cancel event，抛 `CancelledError`
4. worker 退出，GUI 显示「已终止：保留 N 张已落盘图像」+ 逐条列出路径

**已落盘的图永远保留**。下次生成是新批次（新 `{timestamp}_{slug}_` 前缀），不会冲突。

## 流式开关 / Stream

`stream` 默认 **on**，因为大多数中转走 gpt-5.5 + `image_generation` tool 时只在 SSE 路径返回最终图 b64。关掉拿到的是 chat 风格空响应。OpenAI 官方两种都行，`--no-stream` 主要给调试用。

中转 SSE 被中途掐断 → 日志只见 `event: response.created` 然后没了。绝大多数是**中转代理 SSE timeout 太短**或**根本不支持 `image_generation` 工具**。客户端救不了，`_diagnose_empty_response()` 会给一行中文诊断（"SSE 在生成开始后被中转切断"/"中转把请求当成了普通聊天"等），建议换 provider。

## 参考图压缩 / Reference image preprocessing

`image_to_data_url(value, max_edge=1536, jpeg_quality=90)` 用 Pillow 把本地图缩到长边 ≤max_edge、无 alpha 转 JPEG @90、有 alpha 保 PNG。日志每张打印 `原始尺寸 → 新尺寸 (字节变化, mime)`。
- 没装 Pillow → 优雅回退原图 + 一次性警告
- `max_edge=0` 跳过压缩
- 关键作用是把 b64 体积压下来，避免中转因上传慢导致 SSE 在生成完之前被掐
- GUI 下拉默认 `1536 (推荐)`，可选 1024 / 2048 / 原始

## 润色系统 / Polish — 详细

入口：`refine_prompt_fields_with_gpt5(fields, mode, intensity, locks, ...)`。把 6 个字段（`general_prompt` / `main_prompt` / `person_prompt` / `style_prompt` / `scene_prompt` / `avoid_prompt`）整体送给一个文本模型润色，返回同样 schema 的字典。

### Mode（按钮：翻译 / 润色 / 翻译+润色 / 变体（保主体）/ 脑洞（重写））

- `translate` — 只翻译不改写。强度强制下调到 conservative。
- `polish` — 假设输入已是目标语言，按摄影词汇库 + 反 AI tells 改写。
- `translate_polish`（默认）— 先翻再润色。最常用。
- `variant_kind="soft"`（按钮"变体（保主体）"）— 细节变体。主体身份、场景类型、体裁不变，随机替换具体细节（服装颜色 / 材质 / 配饰 / 相机 / 镜头 / 光位 / 时间 / 道具）。强度自动至少升到"开放"。
- `variant_kind="wild"`（按钮"脑洞（重写）"）— 创意 pivot。强制模型挑一个 anchor（不同年代 / 地区 / 场地 / 光线 key），围绕它重写服装 / 场景 / 光线 / 相机时代 / 姿态。强度自动至少升到"激进"。仅保留：主体身份 + 锁定字段 + 预设方向。

### Intensity（润色强度，5 档）

后端 key → GUI 中文：`conservative`→保守 / `open`→开放 / `gacha`→抽卡 / `aggressive`→激进 / `unhinged`→暴走。

- **保守** — 翻译 + 去重 + 把模糊词换成具体摄影词。姿态 / 场景 / 服装一字不动。
- **开放** — 不锁定字段允许细化，主动推编辑摄影口吻；抽象词写实，场地和主体动作不变。
- **抽卡** — 身份保留，其余全开，重写出**一套**连贯新造型。{服装, 姿态, 光线, 构图} 至少 2 轴做非默认抉择。
- **激进** — 除身份外全部重写，**禁用安全默认**（中性站姿 / 平顶光 / 留白底 / 平视全身）。
- **暴走** — 顶配档。{coverage, pose, 表情, framing, lens, 光位, 调色, 场景 mood} 至少 5 轴必须非默认。安全选项全禁。配 wardrobe / pose preset 最够味。

**新增强度档时必须主动推 register**：每档 system prompt 都得列出被禁的默认值 + 最低非默认决策数；不能只写"允许改什么"，否则模型会保守输出。加新档要同步改 4 处 GUI 联动（intensity 下拉 / variant_kind 升档表 / `_INTENSITY_NOTES` / `_anchor_block` 的 axes 选择）。

### Lock 字段（GUI 提示词区底部 6 个勾选框）

`locked_fields` 列表里的字段：
1. 不被润色改写（输出 = 输入原文一字不动）
2. 不被 dedupe 抽走片段并入其他字段（但仍占用 dedupe 的 `seen` 集合，可压制重复）
3. 仍作 frozen 上下文给模型，保证整段语义协调

`locks` 字典里的语义锁（按维度，不是按字段）：`identity` / `wardrobe` / `makeup` / `hair` / `pose` / `scene` / `style`。`identity` 总是 ON。其它由 `_LOCK_NOTES` 注入 system prompt。

**优先级**：LOCK > PRESET > INTENSITY。锁定字段连 preset 都不会改它。

### 5 个方向 preset（GUI"润色预设"区下拉）

`WARDROBE_PRESETS` / `SCENE_PRESETS` / `SHOOTING_STYLE_PRESETS` / `FRAMING_PRESETS` / `POSE_PRESETS`。每张表都自带：
- `"无"` — 不注入方向，按原 intensity 跑。
- `"不改动"` — PRESERVE 指令，**override intensity**。intensity 即便是"激进"也不会动这个维度的内容。
- 若干具体方向（保守 / 日常 / 时尚编辑 / 大胆露肤 / 泳装 / 贴身运动 / 街头都市 / 自然户外 / 影棚极简 / 夜店霓虹 / CCD 千禧直闪 / 35mm 胶片 / 黑白纪实 / 大头特写 / 全身 / 低角度 / 大胆开腿 / M字开腿 / 挺胸后仰 / 跨坐 / ...）

**preset 必须用 DETECT → STRIP → APPLY 三步**走，否则模型会把 preset 当"附加修饰"贴在原文旁边，原姿态/原服装仍主导输出。这套硬规则写在 `_build_polish_system()` 的 "PRESET PRIORITY & APPLICATION PROCEDURE" 块：
1. **DETECT**：扫所有可编辑字段，找该 preset 维度的语言（姿态 preset → 扫 person_prompt 里所有 pose/gesture/expression/gaze/hand-placement 描述）。
2. **STRIP**：在输出里**删除**这些描述，**不**保留两份并列。
3. **APPLY**：把 preset 的 geometry / coverage / camera / framing 翻成具体真人拍摄语言写进对应字段。

frozen / locked 字段豁免 STEP 1/2 — 锁定字段就是用户故意要保留的，即使和 preset 冲突也不动。

### Anchor 池 + creative_seed（让重复调用真的不同）

模型对抽象 seed 整数基本无感，但对具体例子高度敏感。`_ANCHOR_POOLS` 按 5 个轴（`era` / `venue` / `lighting` / `palette` / `lens`）各列 8-14 个具体短语；`_pick_random_anchors(seed)` 按 seed 抽几个塞进 system prompt 的 "INSPIRATION ANCHORS" 块。同 seed → 同 anchor → 可复现；不同 seed → 不同 anchor → 真的不同输出。

axes 数量按 intensity / variant 调整：
- `unhinged` / `variant=wild` → 5 轴
- `aggressive` → 4 轴
- `variant=soft` → 3 轴（palette/lens/lighting）
- `gacha` → 2 轴（palette/lighting）
- 其它档不注入 anchor

`creative_seed` 缺省时：variant 模式或 gacha+ 强度档**自动**生成；低强度档不种随机以免用户期待"忠实翻译"时被惊到。

### Polish link mode（GUI"润色模式"下拉）

- `linked`（默认，关联）— 所有可改字段一次性塞同一 JSON 调用，字段之间互相参考，保持协调。
- `independent`（独立）— 每个可改字段单独发一次 API，其余字段作 frozen 上下文。防止强势字段（例如场景）盖过弱势字段（例如人物细节）。**代价是 API 请求数 × 可改字段数**。独立模式下每个字段用偏移种子（`seed + idx*9973`）以进一步分散。

### 调润色质量去哪改

`_BASE_SYSTEM_PROMPT` / `_PHOTO_DICTIONARY` / `_ANTI_AI_RULES` / `_INTENSITY_NOTES` / `_LOCK_NOTES` / 5 张 `*_PRESETS` 字典 / `_ANCHOR_POOLS`。所有都在 backend 第 14 区。

## gpt-image-2 输入约束 / Size limits

`image_size()` 校验：宽高都是 16 的倍数；长边 ≤ 3840；总像素 655,360 ~ 8,294,400；长宽比 ≤ 3:1。

- `quality`：`auto` / `low` / `medium` / `high`
- `background`：`auto` / `opaque`（**无 transparent**，gpt-image-2 不支持）
- `output_format`：CLI 暴露 `png` / `webp`；backend 内部也接受 `jpeg`

GUI 尺寸下拉有几个 *safe* 预设（`2144x3824` / `3824x2144`），是把 3840 主流尺寸往下找最近的 16 倍数——某些中转对边长上限更严格，这些更稳。

## GUI 各区说明 / GUI panels

主窗口标题 `GPT-5.5 → Image 2 工作台`，最小 1040×760。整窗包在一个 Canvas+Scrollbar 里，内容多时整体滚动。布局：

```
┌─────────────────────────────┬───────────────┐
│  提示词 (prompt_pane)         │ 参数 (settings)│
│  ┌ 6 个文本框 ───────────────┤  - 数量/质量    │
│  │ 通用 / 主体 / 人物 / 风格 │  - 尺寸/方向    │
│  │ / 场景 / 避免内容          │  - 格式/压缩    │
│  ├ 提示词文件 (多选, ; 分隔)  │  - 审核/动作    │
│  ├ 润色按钮行 ×5 + ↶ 回退     │  - 模型/目标语言│
│  ├ 强度 / 范围 / 模式         │  - 预览帧/SSE   │
│  ├ 6 个锁定 checkbox          │  - 输出目录     │
│  └ 润色预设 (5 维度下拉+      │  - 重试/超时    │
│     润色服务商+模型 +保存/-删) │  - 并发        │
│  └ 润色补充 (自由文本)        │  - 中转站+BaseURL│
│                              │  - 参数名+备注   │
├─────────────────────────────┴───────────────┤
│  参考图 (refs_pane)                            │
│  ├ 列表 + 添加文件/URL/删除/清空                │
│  └ 上传前长边下拉                              │
├──────────────────────────────────────────────┤
│  命令 (actions_pane)                           │
│  ├ 拼好的 CLI 命令预览（自动同步参数）          │
│  ├ 开始生成 / 终止 / Dry-run / 保存参数 /       │
│  │ 另存为 / 加载参数 / 删除参数 / 复制命令 /     │
│  │ 打开输出目录                                │
│  └ 状态行 (Ready / Generating ... / Error 等)  │
├──────────────────────────────────────────────┤
│  日志 (log_pane) — 滚动文本，捕获 stdout/stderr │
└──────────────────────────────────────────────┘
```

### 提示词区（左上）
- 6 个多行 Text，每个 label 有 hover tooltip 讲它的用途和注意事项（润色阶段把哪些词搬到哪里、gpt-image-2 审核硬线等）。
- 「提示词文件」选择按钮支持多选，路径用 `;` 拼接到 entry 框，生成时按顺序读入文件追加到 prompt 后面。
- 润色按钮行（5 个）：**翻译 / 润色 / 翻译+润色 / 变体（保主体）/ 脑洞（重写）**，按钮挂的 `_start_polish(mode, variant_kind)` 把当前 6 个文本框 + lock/intensity/preset 一起送给 `refine_prompt_fields_with_gpt5`。完成后**整体替换**文本框内容，**回退 (↶)** 按钮恢复上一次润色前的内容（栈深 20）。
- 强度下拉、范围下拉（全部 / 仅人物 / 仅场景 / 仅风格 / 人物+场景 / 人物+风格）、润色模式（关联 / 独立）。
- 6 个 lock checkbox（通用 / 主体 / 人物 / 风格 / 场景 / 避免）。
- 润色预设区：5 个方向下拉 + 润色服务商下拉 + Base URL + 润色模型 Combobox（可手输任意模型名，"+保存"按钮把它存为 profile 级自定义预设，"×删除"反向操作）。
- 润色补充：自由文本，挂在所有内置 system prompt 之后，**优先级最高**（覆盖前面的强度/预设倾向）。

### 参数区（右上）
- 数量（1-10）、质量（auto/low/medium/high）。
- 尺寸下拉（含 portrait/landscape/square 多个预设 + `safe` 变体 + Custom）；选 Custom 时手填框可编辑。
- 方向下拉（真正自动 / 竖图 / 横图 / 方图）—— 把"自动"展开成具体长边方向，避免中转把所有"auto"都吃成方图。
- 格式（png/webp）+ 压缩（webp 时生效，0-100）。
- 审核（auto/low）。**不要给 background 加 transparent**——gpt-image-2 不支持，会被拒。
- 模型（图像模型，默认 gpt-5.5）。
- 目标语言下拉（English / 简体中文 / 繁體中文 / 日本語 / 한국어 / 保持原文）—— 是润色用的，不是图像生成。
- 预览帧（0-3）+ 流式 SSE checkbox（默认 on，关掉走 stream=false）。
- 输出目录、重试次数、超时秒、并发（0=自动 = min(token 数, 任务数)）。
- 中转站（图像 provider）+ Base URL。
- 参数名（profile 名下拉）+ 备注。

### 参考图区（中部）
- Listbox 列出本地路径 / URL / data: URL。
- 上传前长边下拉（1024 / 1536 / 2048 / 原始）—— 见上文"参考图压缩"。
- 生成时 GUI 会先调 `image_to_data_url` 缩图、转 base64、塞进 payload 的 `input_image` 段。

### 命令区 + 日志区（底部）
- "命令"框显示当前参数对应的 CLI 命令（**实时拼**，改任何 GUI 控件都会刷）。「复制命令」一键到剪贴板。
- 开始/终止/Dry-run/保存/另存为/加载/删除/复制/打开输出目录 共 9 个按钮。
- 日志区接管 stdout/stderr（通过 `_QueueWriter`），把后端的 `log()` 输出实时显示。

### 参数 profile 持久化
- 存在 `gui_profiles.json`（JSON 字典，key=profile 名，value=参数字典）。
- "另存为" / "保存参数" 写入，「加载参数」按下拉读回，「删除参数」从字典里 pop。
- 备注字段 (`profile_note_var`) 跟着 profile 走。
- 用户的"+保存"自定义润色模型也存在 profile 里（`_polish_model_custom_presets`）。

## Prompt 文件目录 / prompt_examples

```
_反AI生成感总纲_cn.txt          ← 每次都建议挂的硬规则
_使用指南_cn.txt                ← 用法说明
辅助片段/  (18 个)              ← 小型可叠加片段，每个聚焦一点
完整风格/  (11 个)              ← 完整场景模板（CCD 千禧年 / 35mm 胶片 / 影棚 / ...）
（根目录 6 个 *_cn.txt 是早期版本，向后兼容）
```

CLI 通过重复 `--prompt-file path`；GUI"提示词文件"按钮多选（分号分隔）。
建议组合：`_反AI生成感总纲` + 完整风格里挑 1 个 + 辅助片段 2-4 个互补的。**别全部叠**——GPT-5.5 会无所适从。同类只选一个（光位类只选一个、负面类内容应进 avoid_prompt 而非正向 prompt）。

## 内容审核硬线 / Moderation guardrails

下面两种组合 gpt-image-2 **必拒**，无论 `--moderation auto` 还是 `low`：

1. **命名公众人物**（K-pop 艺人、明星本名）—— 真实肖像权过滤
2. **"X 岁少女" + 露骨身体规格**（D-cup、低胸、毫不遮掩等）—— 性化未成年/年轻分类器命中

引导改造方向：
- 命名 → 视觉档案类型（"cat-eye Korean editorial subject"）
- 19 岁少女 → "mid-twenties Korean fashion model" / "adult"
- 露骨身体描述 → 编辑时装语言（"curvy hourglass, deep V neckline as styled fashion, confident editorial posture"）

视觉效果几乎一样，但 API 通过率天差地别。`POSE_PRESETS` 里的"大胆开腿/M字开腿/挺胸后仰/胸前夹持/..."所有大胆姿态都自带 `Adult subject.` 收尾——和"X 岁少女"措辞同时出现仍会被拒，必须搭"成人/早20+/编辑"语境。

## 改这个仓库时的注意 / Modifying notes

- 后端公开 API 名（见上文"架构"清单）尽量稳定，改名要同步改 GUI import 块。
- `cancel_event` + `session` 参数在 `generate()` 签名末尾；新增内部 HTTP 调用要透传。
- 加 provider：编辑 `PROVIDERS` dict 即可（注意 `text_api` / `fallback_to_freemodel`）。
- 加润色方向 preset：在对应的 `WARDROBE_PRESETS` / ... 字典追加一项，GUI 下拉自动出现。
- 加 intensity 档：改 `_INTENSITY_NOTES` + `_anchor_block` 的 axes 选择 + GUI intensity 下拉 values + variant_kind 升档表（`_start_polish` 里）。
- 加 prompt 文件：放 `prompt_examples/辅助片段/` 或 `完整风格/`，GUI 文件选择器自动看到。
- **不要把代码再绑死到 "freemodel.dev"** —— `PROVIDERS["freemodel"]` 只是其中一项，新增常量/变量名要 generic。
- 不要碰 `*.bak` / 不要重命名 `gui_profiles.json`（会丢用户 profile）。
- 不要给 image2 的 `background` 加 `transparent` —— 模型不支持，会被拒。
- 新加 preset 别忘了"不改动"分支 —— 用户预期每个方向下拉都有 PRESERVE 选项。
