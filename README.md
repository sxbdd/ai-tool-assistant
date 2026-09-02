# AI 工具调用小助手（mini-Codex · Agent + RAG + 识图）

一个"会查资料、会看图、会调工具、多模型可切换"的网页智能体，尽量贴近 Codex 的
**决策(模型) + 执行(工具) + 循环(LangGraph) + 安全(沙盒)** 运行思路。

> 当前版本 v2.1：Function Calling 多工具 + RAG 知识库（带出处）+ 免费识图（视觉桥）+ 本地文件夹导入。
> 详细设计见 [DESIGN.md](DESIGN.md)。

## ✨ 功能

| 能力 | 说明 | 示例 |
|---|---|---|
| 🛠️ 工具调用 | 读文件 / 查天气(Open-Meteo) / 整理表格(pandas) | "整理 data/示例-销售数据.csv" |
| 📚 RAG 知识库 | 上传/导入笔记→切分→向量化(Chroma)→检索**带出处**回答 | "根据我的笔记解释 RAG" |
| 🖼️ 识图 | 视觉桥：文本模型不会看图？调免费视觉模型转成描述 | 上传图片问"里面写了什么" |
| 🔀 多模型切换 | DeepSeek / 智谱 / 通义 / OpenAI 下拉即换（OpenAI 兼容协议） | 侧栏切换主力与识图模型 |
| 📁 本地文件夹导入 | 填绝对路径→只读扫描→复制进沙盒→批量入库 | 导入整文件夹课件 |
| 🛡️ 安全边界 | 写入只在 data/ 沙盒；Key 只存 .env；越界检索有相似度阈值 | — |

## 🧱 技术栈

Python 3.12 · LangGraph · langchain-openai · DeepSeek API · Chroma · sentence-transformers(bge-small-zh-v1.5) · Streamlit · pandas

## 📁 项目结构

```
AI 工具调用小助手/
├── app.py                 # Streamlit 网页（模型切换/知识库管理/识图/对话）
├── DESIGN.md              # 设计说明书（架构/里程碑/验收/边界）
├── agent/
│   ├── graph.py           # LangGraph 编排：agent↔tools 条件循环
│   ├── tools.py           # 5 个工具 + Function Calling 注册表
│   ├── model_providers.py # 多模型抽象层（Provider 注册表）
│   └── config.py          # .env 配置
├── rag/                   # RAG 子系统
│   ├── ingest.py          # 解析(PDF按页)→切分(递归+重叠)→入库
│   ├── store.py           # Chroma 封装 + embedding
│   ├── search.py          # 检索工具（相似度阈值过滤）
│   └── config.py
├── data/
│   ├── 示例-待办.txt / 示例-销售数据.csv / 示例-课程笔记.md
│   ├── uploads/  kb/      # 上传与知识库（沙盒，gitignore）
│   └── output/            # 表格整理产物（gitignore）
├── scripts/e2e_test.py    # 端到端冒烟（真调 DeepSeek）
└── tests/                 # 离线单元测试（9 个）
```

## 🚀 快速开始

```powershell
# 1) 依赖（本机虚拟环境已装好；新机器先建 venv 再装）
python -m venv .venv
pip install -r requirements.txt

# 2) 配置 Key：复制 .env.example 为 .env 并填写
#    - DEEPSEEK_API_KEY：主力对话（必填）
#    - ZHIPU_API_KEY / DASHSCOPE_API_KEY：识图（免费档，任一即可，视觉桥用）

# 3) 启动网页
streamlit run app.py
```

打开 http://localhost:8501 后可试：
- `根据我的笔记，什么是 RAG？`（会先自动把示例笔记入库，然后检索回答）
- 上传一张图 → `让 AI 识别这张图片`
- `北京和上海今天天气怎么样？`、`整理 data/示例-销售数据.csv`

### embedding 模型下载（首次）
知识库向量化用 `BAAI/bge-small-zh-v1.5`，首次使用自动从 HuggingFace 下载（约 100MB）。
- 国内可设 `HF_ENDPOINT=https://hf-mirror.com`；
- 走代理直连失败时设 `HF_HUB_DISABLE_XET=1` 再试。

## ✅ 测试

```powershell
# 离线单测（不需要 API Key）：切分/入库/检索/阈值 + 老工具回归
python -m unittest discover -s tests -v

# 端到端（需要 .env 的 DeepSeek Key）：工具调用 + RAG + 识图
python scripts/e2e_test.py
```

## 🧠 Agent 怎么工作（面试要点）

1. 工具注册：`tools.py` 把每个工具写成 JSON Schema 说明书；
2. 模型决策：DeepSeek 返回 `tool_calls`（调哪个工具、传什么参数）；
3. 本地执行：LangGraph `tools` 节点跑真 Python 函数，结果回传；
4. 循环：还要调工具就再进 tools，不需要就输出最终答案；
5. RAG：检索片段带（来源：文件·页码），回答只依据片段并标注，防幻觉（还有相似度阈值兜底）；
6. 识图：视觉桥 `analyze_image` 调免费视觉模型把图转文字，主模型再回答。

## 📚 参考学习资料

- datawhalechina/hello-agents、datawhalechina/llm-universe
- langchain-ai/rag-from-scratch
- AFPyannian/lifepilot-agent（LangGraph+Streamlit+DeepSeek+RAG 参考）