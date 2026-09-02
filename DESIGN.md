# DESIGN v2.1 —— AI 智能体助手（mini-Codex · DeepSeek 版）

> 目标形态：一个"会查资料、会看图、会调工具、多模型可切换"的网页智能体，尽量贴近 Codex 的
> **决策(模型) + 执行(工具) + 循环(LangGraph) + 安全(沙盒/确认)** 运行思路。
> 本文是设计与验收说明书，先评审、后编码。

---

## 0. 版本范围（用户已确认）

- ✅ v2.0 核心：多模型抽象层 + 多文件上传 + **本地文件夹路径导入** + RAG 知识库问答（带出处）
- ✅ v2.1 识图：免费视觉模型"视觉桥"（analyze_image 工具），可选直连视觉大模型
- ❌ v2.2 语音（faster-whisper + edge-tts）：留作下一期彩蛋，本期不做
- ❌ 自动改文件 / 任意跑命令工具（v3 再做，本期保持"只读+沙盒写入"安全姿态）

---

## 1. 架构图

```
┌──────────────────────────── 网页层 app.py（Streamlit）─────────────────────────────┐
│ 侧栏：① 模型设置(主力/识图下拉+Key状态)  ② 知识库管理(上传/导入文件夹/列表/删除)     │
│ 主区：对话窗口 + 图片上传识图 + 🧰工具轨迹 + 📚引用出处                              │
└──────────────────────────────────┬──────────────────────────────────────────────────┘
                                   ▼
┌──────────────────────── 编排层 agent/graph.py（LangGraph，沿用）───────────────────┐
│   agent(模型决策) ↔ tools(本地执行)  循环，直到产出最终答案                          │
└──────────────────────────────────┬──────────────────────────────────────────────────┘
                                   ▼
┌──────────────────────── 模型抽象层 model_providers.py（新增）───────────────────────┐
│   PROVIDERS 注册表：deepseek / zhipu / qwen / openai(可选)                           │
│   统一走 OpenAI 兼容协议：换 base_url+model+api_key 即可切换                          │
│   职责拆分：DEFAULT_MAIN(对话)  vs  VISION(识图桥)                                    │
└───────────────┬──────────────────────────────────┬──────────────────────────────────┘
                ▼                                  ▼
┌──────── 工具层 agent/tools.py（扩展）───────┐  ┌──── RAG 子系统 rag/（新增）────────┐
│ ① read_file ② get_weather                  │  │ ingest(切分/入库) store(Chroma)     │
│ ③ tidy_spreadsheet ④ search_knowledge_base │  │ search(检索) prompts(带出处生成)    │
│ ⑤ analyze_image（视觉桥，v2.1）             │  └─────────────────────────────────────┘
└────────────────────────────────────────────┘
                                   ▼
              沙盒目录 data/：uploads(上传) / kb(知识库原始+chroma)
              本地文件夹导入：只读扫描原目录 → 受支持文档复制进 data/kb/raw/
```

---

## 2. 模型抽象层设计（核心新增）

### 2.1 需求
用户要能换模型，例如"主力用 DeepSeek 省钱，看图用免费识图模型，以后想用 GPT 随时切"。

### 2.2 设计：Provider 注册表 + 职责分离
```python
# model_providers.py（示意）
PROVIDERS = {
  "deepseek": {"base_url": "https://api.deepseek.com",                       "models": ["deepseek-chat", "deepseek-v4-flash-vision-exp"]},
  "zhipu":    {"base_url": "https://open.bigmodel.cn/api/paas/v4",           "models": ["glm-4v-flash"]},     # 免费档
  "qwen":     {"base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1", "models": ["qwen-vl-plus"]},
  "openai":   {"base_url": "https://api.openai.com/v1",                      "models": ["gpt-4o-mini"]},       # 可选付费
}
def create_chat_model(provider, model):  # 统一 ChatOpenAI(base_url, api_key, model)
def create_vision_model():               # 由 VISION_PROVIDER 决定，供 analyze_image 使用
```

### 2.3 .env 配置约定（Key 一律不进 git）
```ini
# 主力对话模型
DEFAULT_PROVIDER=deepseek
DEEPSEEK_API_KEY=sk-...
DEEPSEEK_MODEL=deepseek-chat
# 识图视觉桥（免费档优先）
VISION_PROVIDER=zhipu              # 或 qwen
ZHIPU_API_KEY=...                  # 免费申请 open.bigmodel.cn
ZHIPU_VISION_MODEL=glm-4v-flash    # 免费；备选 glm-4.6v-flash
# 可选：Qwen（阿里百炼，新用户免费额度）
# DASHSCOPE_API_KEY=...
# QWEN_VISION_MODEL=qwen-vl-plus
```

### 2.4 UI 表现
- 侧栏"主力模型 / 识图模型"两个下拉框；
- 未配置 Key 的 Provider 显示"未配置（置灰）"，避免用户选了报错；
- 聊天中一切模型调用走统一工厂，不散落各处。

---

## 3. 文件与本地文件夹导入

| 方式 | 设计 |
|---|---|
| **多文件上传** | Streamlit 多选上传 → 存 `data/uploads/`；白名单：pdf/txt/md/csv/png/jpg/jpeg/gif/webp；单文件 ≤20MB |
| **本地文件夹导入**（用户已选"本地路径"） | 侧栏输入**绝对路径** → 前端先展示"将扫描到的支持文档清单 + 预估" → 用户点确认 → 只读遍历，把支持的文档**复制**进 `data/kb/raw/<源文件夹名>/` → 逐份入库(ingest) → 报告"N 个文件、M 个片段" |

**为什么不直接引用原路径**：保持沙盒自包含——知识库只依赖 `data/`，用户删了原文件夹不影响已建索引；也避免助手乱写用户目录。

---

## 4. RAG 子系统（沿用前面评审过的设计，落地细节）

- 解析：`.pdf`(pypdf 按页) / `.txt` / `.md` / `.csv`；
- 切分：递归切分（先空行→段落→句号），chunk 约 500 字、重叠 80 字；
- 向量化：`bge-small-zh-v1.5`（本地，中文好，~100MB，走 HF 镜像下载）；
- 存储：Chroma 持久化 `data/kb/chroma/`，每 chunk 元数据 = {文件名, 页码(如有), 块序号}；
- 检索工具 `search_knowledge_base(query, top_k=4)`：返回带"（来源：文件·页码·块号）"的片段；
- 生成约束（prompts.py）：只依据片段回答；查不到就明说；引用标 `[来源]`。

---

## 5. 识图设计（v2.1）

### 5.1 视觉桥 analyze_image（默认，免费）
```text
Agent 判断需要看图 → 工具 analyze_image(图片路径, 可选问题)
  → 读图转 base64 → 调 VISION_PROVIDER 的免费视觉模型
  → 返回图片的文字描述 → 主模型基于描述继续回答
```
- 实现即 OpenAI 兼容消息：`content=[{type:"text"},{type:"image_url", image_url:{url:"data:image/png;base64,..."}}]`；
- 未配置识图 Key → 工具返回友好提示，不让程序崩；
- **截图里的文字描述可以顺手存进知识库**（"看图→文字化→可检索"），作为 stretch 目标。

### 5.2 直连视觉模型（可选开关）
当用户把"主力模型"切成 DeepSeek-vision-exp / Qwen-VL / GPT 等视觉模型时，图片直接以消息形式发给主力模型（provider 带 `supports_vision` 标记）。

### 5.3 网页交互
上传图片 → 预览缩略图 → 自动以用户消息"请分析这张图片"发出 → 回答展示 + 轨迹展开。

---

## 6. 安全边界（延续并强化）

1. 所有写入只发生在 `data/{uploads,kb}`，`.gitignore` 已忽略；
2. 工具读取限制在项目目录（现有 `read_file` 逻辑）+ 上传目录；
3. 本地文件夹导入：**只读**原目录 + 仅复制白名单类型，绝不动原文件；
4. API Key 只存 `.env`，`.env.example` 提供模板，不入库；
5. 本期不开放"任意命令执行 / 任意路径写文件"工具（v3 再考虑，届时也要做确认机制）。

---

## 7. 新增/修改文件清单

```
model_providers.py       (新)  Provider 注册表 + create_chat_model/create_vision_model
rag/
  __init__.py            (新)
  config.py              (新)  chunk 大小 / top_k / embedding 模型名
  ingest.py              (新)  解析→切分→向量化→入库（含文件夹批量）
  store.py               (新)  Chroma 封装（add/query/list/delete）
  search.py              (新)  search_knowledge_base 工具实现
  prompts.py             (新)  带出处生成模板
agent/
  tools.py               (改)  注册表加 search_knowledge_base、analyze_image
  graph.py               (改)  模型改走 model_providers 工厂；系统提示词更新
  config.py              (改)  读新增 .env 项
app.py                   (改)  侧栏模型设置/知识库管理/文件夹导入 + 图片上传识图
requirements.txt         (改)  + chromadb, sentence-transformers, pypdf, pydantic(已有)
tests/
  test_rag.py            (新)  离线：小文本入库→检索取回→带出处
  test_vision.py         (新)  离线：analyze_image 无 Key 时优雅报错（有 Key 可跳过）
scripts/e2e_test.py      (改)  增加 RAG + 识图用例
DESIGN.md                (本文件)
```

---

## 8. 里程碑与验收标准

### M1 模型抽象层
- [ ] UI 可切换 Provider；未配 Key 置灰提示；
- [ ] 原 3 个工具用例（读文件/天气/表格）**回归通过**（不回归是底线）。

### M2 文件上传 + 本地文件夹导入 + RAG
- [ ] 上传一个 PDF/笔记 → 入库成功，显示块数；
- [ ] 输入本地文件夹路径 → 确认后批量入库 → 报告清单；
- [ ] 问"文档里讲了什么" → 回答引用原文且带出处；
- [ ] 问库外内容 → 回答"知识库未找到"，不编造；
- [ ] 离线单测（ingest→retrieve）通过。

### M3 识图（视觉桥）
- [ ] 上传图片 → 工具返回可用描述（需你提供任一免费识图 Key：智谱/通义）；
- [ ] 未配置 Key 时页面友好提示、不崩溃；
- [ ] 端到端（真调 DeepSeek）综合用例跑通：查知识库 + 看图 + 调天气 串联。

### 总验收
- [ ] 老功能零回归；新增功能各有离线测试或可复现演示；
- [ ] README 更新架构图与"如何配 Key / 如何换模型"说明。

---

## 9. 明确不做（防失控，也便于面试讲边界）

- 语音输入/输出、任意命令执行、自动改文件、多用户、云端部署、OCR 复杂版面、rerank/混合检索、生产级向量库。

---

## 10. 简历价值（对应最终写法）

> "基于 LangGraph 搭建**多模型可切换的智能体助手**：接入 DeepSeek 与免费视觉模型，实现 RAG 知识库问答（带出处）、Function Calling 多工具（读文件/查天气/整理表格/看图）、本地资料批量导入与沙盒安全边界，端到端验证通过。"

覆盖考点：RAG 全流程 / Function Calling / LangGraph 编排 / 多模态（视觉桥）/ 模型抽象与切换 / 工程安全规范。