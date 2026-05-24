# GPT-5.5 → Image 2 工作台

通过 OpenAI 兼容中转的 **Responses API** 调 `gpt-5.5` + `image_generation` 工具，产出 **gpt-image-2** 质量的图。带 Tkinter GUI，主要给一种工作流用：

> 中文写提示词 → 一键润色成专业摄影英文 → 多 token 并发出图 → 失败可重试、可终止、已落盘永远保留。

**仅 Windows 测试**，未做 Linux/macOS 适配。

---

## 它能做什么

- ✅ 调任意 OpenAI 兼容中转出 gpt-image-2 质量的图（freemodel.dev / OpenAI 官方 / laozhang.ai / aimlapi / 自定义）
- ✅ 多 token 自动并发分摊任务
- ✅ 中文 → 专业摄影英文一键润色（**5 档强度** × 5 个方向 preset × 6 个字段锁）
- ✅ 参考图 + 文字描述做 ref-conditioned 改图（换风格 / 换场景 / 换服装 / 换姿态）
- ✅ 6 个独立润色 provider（DeepSeek / Claude / Grok / GLM / Kimi / Qwen）专门跑文本，便宜又快
- ✅ 反 AI 生成感的硬规则 + 摄影词汇库 + few-shot 样例内嵌在润色 system prompt
- ✅ 内容审核硬线自动规避（年轻 + 露骨 / 命名公众人物等组合）
- ✅ 原子落盘：进程被强杀也不会留半张坏图；中途终止保留所有已完成的图

---

## 系统要求

- **Windows 10/11**（仅在此测试，其它系统未保证）
- **Python 3.10+**
- 至少一个 OpenAI 兼容中转的 API token

---

## 安装

```bash
# 1. 克隆/下载本仓库
git clone <repo-url>
cd <repo-folder>

# 2. 装依赖（必需 requests；强烈推荐装 Pillow）
pip install -r requirements.txt
```

`tkinter` 是 Python 标准库，Windows 官方 Python 安装包自带，不用单独装。

---

## 配置 token

复制 `.env.example` 为 `.env`，填入你的 token：

```bash
copy .env.example .env
```

`.env` 内容（freemodel 示例，其它 provider 见下）：

```ini
# 方式 A：一行一个 token（项目会自动去重）
FREEMODEL_TOKEN=sk-xxxxxxxxxxxx
FREEMODEL_TOKEN_2=sk-yyyyyyyyyyyy
FREEMODEL_TOKEN_3=sk-zzzzzzzzzzzz

# 方式 B：合并一行，逗号/空格/分号都行
# FREEMODEL_TOKENS=sk-a,sk-b,sk-c
```

**多 token 的好处**：`generate(n=4)` 时会自动把 4 张图分摊到所有 token 并发跑，整体耗时 ≈ 1 张图的时间。

### 其它中转的环境变量名

| Provider | 主 token env | 多 token 列表 env | 用途 |
|---|---|---|---|
| freemodel.dev（默认） | `FREEMODEL_TOKEN` / `_2` ... `_10` | `FREEMODEL_TOKENS` | 图像 + 润色 |
| OpenAI 官方 | `OPENAI_API_KEY` / `_2` / `_3` | `OPENAI_API_KEYS` | 图像 + 润色 |
| laozhang.ai | `LAOZHANG_API_KEY` / `LAOZHANG_TOKEN` | `LAOZHANG_TOKENS` | 图像 + 润色 |
| AI/ML API | `AIML_API_KEY` / `AIMLAPI_KEY` | `AIML_API_KEYS` | 图像 + 润色 |
| DeepSeek | `DEEPSEEK_API_KEY` / `DEEPSEEK_TOKEN` | `DEEPSEEK_API_KEYS` | **仅润色** |
| Anthropic Claude | `ANTHROPIC_API_KEY` / `CLAUDE_API_KEY` | `ANTHROPIC_API_KEYS` | **仅润色** |
| xAI Grok | `XAI_API_KEY` / `GROK_API_KEY` | `XAI_API_KEYS` | **仅润色** |
| 智谱 GLM | `ZHIPU_API_KEY` / `GLM_API_KEY` | `ZHIPU_API_KEYS` | **仅润色** |
| 月之暗面 Kimi | `MOONSHOT_API_KEY` / `KIMI_API_KEY` | `MOONSHOT_API_KEYS` | **仅润色** |
| 阿里通义 Qwen | `DASHSCOPE_API_KEY` / `QWEN_API_KEY` | `DASHSCOPE_API_KEYS` | **仅润色** |
| 自定义 | `OPENAI_API_KEY` / `API_KEY` | `API_KEYS` | 任意，需手填 base URL |

**为什么有"仅润色"** —— DeepSeek / Claude 等不支持 `image_generation` tool，出不了图，但跑文本润色又快又便宜。GUI 里"润色服务商"下拉可以单独选它们专门跑润色，主图像 provider 不变。

---

## 启动 GUI（推荐）

```bash
python API_GPT5_5_to_IMAGE2_GUI.py
```

### 第一次出图（30 秒上手）

1. **填提示词**。最少填"主体"一栏，例如：`a mid-twenties Korean woman sitting in a small cafe`
2. **选尺寸**：参数区"尺寸"下拉选 `Portrait竖图 1024x1536`（或留 `Auto`）
3. **选质量**：参数区"质量"选 `high`
4. **点 "开始生成"**
5. 等 10-30 秒，图会落到 `output/` 目录，文件名形如 `20260524_153012_korean-woman_1.png`

日志窗口会实时显示：`→ POST endpoint`（发请求）→ `✓ 拿到图`（拿到 b64）→ `✓ 已落盘 [1/1]`（写盘成功）。

### GUI 主要区域

```
┌──────────────────────────────┬─────────────────┐
│  提示词区                       │ 参数区            │
│  6 个文本框 + 润色按钮 +         │ 数量 / 质量 / 尺寸 │
│  5 个 preset 下拉              │ 格式 / 模型 / ...  │
├──────────────────────────────┴─────────────────┤
│  参考图区                                          │
│  添加文件/URL + 上传前长边                          │
├────────────────────────────────────────────────┤
│  命令区（CLI 预览） + 开始/终止/Dry-run/保存参数      │
├────────────────────────────────────────────────┤
│  日志区（实时 stdout）                              │
└────────────────────────────────────────────────┘
```

### 提示词区的 6 个字段

| 字段 | 写什么 |
|---|---|
| **通用提示词** | 全局硬规则、反 AI tells（建议挂 `prompt_examples/_反AI生成感总纲_cn.txt`）|
| **主体** | 一句话定调：谁、在哪、做什么 |
| **人物参数细节** | 年龄段（写 adult / mid-twenties）、体型、妆发、瞳色、肤色、表情、服装款式材质、配饰 |
| **拍摄风格** | 相机机身、镜头、胶片、ISO、快门、光位、调色 |
| **场景** | 地点、时间、季节、天气、光源、氛围、道具 |
| **避免内容** | Negative prompt：不希望出现的元素、AI tells、手部畸形等 |

每个字段标签都有 hover tooltip 讲细节。

### 润色按钮（核心功能）

提示词区底部有 5 个润色按钮 + 回退 ↶：

| 按钮 | 干什么 |
|---|---|
| **翻译** | 只翻译不改写。把中文准确转成目标语言。强度强制保守。|
| **润色** | 假设输入已是目标语言，按摄影词汇库 + 反 AI tells 改写收紧。|
| **翻译+润色** | 先翻后润色。**最常用一档**。|
| **变体（保主体）** | 保留主体 / 场景类型 / 体裁，随机换具体细节（服装颜色 / 相机 / 镜头 / 光位 / 时间 / 道具）。|
| **脑洞（重写）** | 创意 pivot。强制挑一个 anchor（不同年代 / 地区 / 场地 / 光线 key），重写服装 / 场景 / 光线 / 相机时代 / 姿态。|
| **↶ 回退** | 撤销上一次润色，恢复改前的文本（栈深 20）。|

### 润色强度（5 档）

| 档位 | 行为 |
|---|---|
| **保守** | 只翻译 + 去重 + 把模糊词换成具体摄影词。姿态 / 场景 / 服装一字不动。|
| **开放** | 不锁字段允许细化，抽象词写实，场地和主体动作不变。|
| **抽卡** | 身份保留，其余全开，重写一套连贯新造型。{服装, 姿态, 光线, 构图} 至少 2 轴做非默认抉择。|
| **激进** | 除身份外全部重写，禁用安全默认（中性站姿 / 平顶光 / 留白底 / 平视全身）。|
| **暴走** | 顶配档。{coverage, pose, 表情, framing, lens, 光位, 调色, 场景 mood} 至少 5 轴非默认。|

### 锁定字段

6 个 checkbox（通用 / 主体 / 人物 / 风格 / 场景 / 避免），勾上后：
- ① 不被润色改写
- ② 不被 dedupe 抽走片段
- ③ 仍作 frozen 上下文给模型参考

### 5 个方向 preset

下拉决定润色 AI 往哪个方向推：

- **服装** — 无 / 不改动 / 保守 / 日常 / 时尚编辑 / 大胆露肤 / 泳装 / 贴身运动
- **场景** — 室内日常 / 街头都市 / 自然户外 / 影棚极简 / 夜店霓虹 / 复古居所
- **风格** — 现代数码自然光 / 35mm 胶片 / CCD 千禧直闪 / 黑白纪实 / 影棚商业大片 / 夜晚直闪
- **构图** — 大头特写 / 特写 / 半身 / 全身 / 远景 / 低角度 / 高角度
- **姿态** — 自然站立 / 走动 / 坐姿 / 斜倚 / 镜头互动 / 侧身回眸 + 多个大胆姿态

每个下拉都有"不改动"选项，选中即冻结对应方面。**preset 永远 override intensity**——选了"激进"+ 服装"不改动"，服装就真的不动。

### 参考图

参考图区可以加本地文件、URL、或 base64 data URL。

**用法**：在提示词里直接描述要改成什么。例子：
- 保留主体换风格：`同一名主体，改用 35mm Kodak Portra 400 胶片，warm cream highlights`
- 保留主体换场景：`同一名主体，改到 Tokyo Harajuku 后巷霓虹环境，wet asphalt reflections`
- 保留主体换服装：`同一名主体，改穿 black leather biker jacket, chrome zippers`

⚠️ **代码层面没有真正的"编辑"模式**，每张输出都是 ref-conditioned 重新合成。身份和构图都可能漂移。要稳一点：多张同主体不同角度 ref + 人物字段写细 + 勾"人物参数细节"锁。

详见 `CLAUDE.md` 的"用文字描述修改参考图"一节。

---

## CLI 用法

```bash
# 最小命令
python API_GPT5_5_to_IMAGE2.py "an early-20s Korean woman in a small cafe" -q high -n 1

# 完整命令：高质量、竖图、出 2 张、自定 provider
python API_GPT5_5_to_IMAGE2.py "your prompt" \
    -q high \
    -s 2160x3840 \
    -n 2 \
    --provider freemodel \
    -o output

# 用提示词文件（可重复 --prompt-file 叠多个）
python API_GPT5_5_to_IMAGE2.py "an early-20s woman, candid window seat" \
    --prompt-file prompt_examples/_反AI生成感总纲_cn.txt \
    --prompt-file prompt_examples/完整风格/胶片35mm人像_cn.txt \
    --prompt-file prompt_examples/辅助片段/肤质_毛孔与血色.txt \
    -q high -s 2160x3840 -n 2

# 参考图
python API_GPT5_5_to_IMAGE2.py "same subject, switch to 35mm film" \
    -r ./refs/portrait.jpg \
    -r ./refs/style_ref.jpg \
    -q high -n 1

# Dry run —— 不真的调 API，只打印解析后的参数和最终 endpoint
python API_GPT5_5_to_IMAGE2.py "anything" --dry-run

# 全部参数
python API_GPT5_5_to_IMAGE2.py --help
```

---

## 提示词文件目录 `prompt_examples/`

```
prompt_examples/
├── _反AI生成感总纲_cn.txt          每次都建议挂的硬规则
├── _使用指南_cn.txt                详细用法说明
├── 辅助片段/  (18 个)              小型可叠加片段，每个聚焦一点
│   ├── 肤质_毛孔与血色.txt
│   ├── 发丝_自然飞散.txt
│   ├── 构图_扰动与离心.txt
│   ├── 光位_轮廓发光.txt
│   ├── 负面_AI破绽扩展.txt
│   └── ... 等共 18 个
├── 完整风格/  (11 个)              整块风格模板
│   ├── CCD千禧年数码_cn.txt
│   ├── 胶片35mm人像_cn.txt
│   ├── 影棚商业写真_cn.txt
│   └── ... 等共 11 个
└── 旧版本兼容文件
```

**推荐组合**：`_反AI生成感总纲` + 完整风格里挑 1 个 + 辅助片段 2-4 个互补的。**别全部叠**——GPT-5.5 会无所适从。同类只选一个（光位类只选一个、负面类内容应进 avoid_prompt 而非正向 prompt）。

详见 `prompt_examples/_使用指南_cn.txt`。

---

## 故障排查

### "SSE 在生成开始后被中转切断"
日志里只见 `event: response.created` 就没了。绝大多数是**中转代理 SSE timeout 太短**或**根本不支持 `image_generation` 工具**。客户端救不了，建议：
1. 换 provider（参数区"中转站"下拉）
2. 把参考图"上传前长边"调到 1024（减小上传体积，留更多时间给生成）
3. 减小 `--n` 或并发，单张多次试

### "鉴权失败 (401/403)"
- 检查 `.env` 里 token 是否填对
- 检查环境变量名是否对应当前选中的 provider
- 多 token 的话，把每个都试一遍单独跑

### "上游报错：safety / moderation / content policy"
gpt-image-2 的内容审核硬线被命中。常见两种：
1. **命名公众人物**（K-pop 艺人、明星本名）→ 改写为视觉档案类型（"cat-eye Korean editorial subject"）
2. **"X 岁少女" + 露骨身体规格** → 改成 "mid-twenties Korean fashion model" + 编辑时装语言

视觉效果几乎一样，但 API 通过率天差地别。

### "中转把请求当成了普通聊天，没触发 image_generation 工具"
某些中转不支持 `image_generation` tool 或 `tool_choice="required"` 被丢弃。换 provider。

### Pillow / 参考图相关报错
没装 Pillow 时项目会优雅回退原图 + 一次性警告。装上即可：`pip install Pillow`。

---

## 文件清单

```
.
├── API_GPT5_5_to_IMAGE2.py        后端 + CLI 入口
├── API_GPT5_5_to_IMAGE2_GUI.py    Tkinter GUI
├── requirements.txt               依赖清单
├── .env.example                   token 模板
├── .env                           你的 token（不要 commit！）
├── CLAUDE.md                      架构 / 改代码指南（深入阅读）
├── README.md                      本文件
├── gui_profiles.json              用户保存的参数档案（自动生成）
├── output/                        生成的图（自动创建）
└── prompt_examples/               提示词模板
    ├── 辅助片段/
    └── 完整风格/
```

---

## 深入阅读

- **`CLAUDE.md`** — 完整架构、Provider 系统、生成流程、取消机制、流式开关、润色系统全套（mode / intensity / 5 维 preset / DETECT-STRIP-APPLY / anchor 池 / creative_seed / link mode）、GUI 各区详解、改代码注意事项
- **`prompt_examples/_使用指南_cn.txt`** — 提示词文件的组合原则
- **CLI 帮助** — `python API_GPT5_5_to_IMAGE2.py --help`

---

## 免责声明

- **个人 Windows 工具**，未做跨平台 / 测试 / 包装。
- 所有出图都由远端 API（OpenAI / 中转）实际生成，本仓库不绕过任何官方审核。
- 不提供 / 不附带任何 API token。需要自己注册 provider 拿 token。
- 内容审核硬线（命名公众人物、性化未成年等组合）由 gpt-image-2 自身把关，本工具不处理也不绕过。
