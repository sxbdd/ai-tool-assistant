# AI 工具调用小助手（Agent + Function Calling）

让大模型（DeepSeek）**学会调用工具**，自动完成多步任务：读文件、查天气、整理表格，并通过网页对话使用。

> 一句话：这是一个从零手写的 **Agent 智能体项目**，核心是 **Function Calling（函数调用）** —— 大模型不直接执行操作，而是"决定调哪个工具、传什么参数"，由本地 Python 真正执行，安全可控、过程可解释。

---

## ✨ 功能

| 工具 | 作用 | 示例指令 |
|---|---|---|
| `read_file` | 读取项目内文本文件 | "帮我读一下 `data/示例-待办.txt`" |
| `get_weather` | 查询城市实时天气（Open-Meteo 免费接口） | "北京和上海今天天气怎么样？" |
| `tidy_spreadsheet` | 整理 CSV/Excel：清空行、去重、保存新文件并输出摘要 | "整理 `data/示例-销售数据.csv`" |

- **多步任务自动编排**：一次提问可连续多次调用工具（Agent 循环），例如"读待办清单 → 逐个查天气 → 汇总回答"。
- **网页对话界面**：Streamlit 聊天窗口，能展开查看每次工具调用过程，学习/演示直观。
- **安全边界**：工具只允许访问项目目录内文件，不修改应用数据与系统文件。

## 🧱 技术栈

- Python 3.12
- **LangGraph**：状态机编排 Agent（agent 节点 ↔ tools 节点循环）
- **langchain-openai**：OpenAI 兼容协议对接 **DeepSeek API**（`deepseek-chat`）
- **Streamlit**：网页对话界面
- **pandas / openpyxl**：表格读取与整理
- 天气数据：Open-Meteo（免费，无需 Key）

## 📁 项目结构

```
AI 工具调用小助手/
├── app.py                 # Streamlit 网页对话入口
├── agent/
│   ├── config.py          # 读取 .env（DeepSeek Key 等配置）
│   ├── tools.py           # 工具实现 + Function Calling schema 注册表
│   └── graph.py           # LangGraph 状态机：模型/工具节点 + 条件路由
├── data/                  # 示例数据（Agent 可读写的安全目录）
│   ├── 示例-待办.txt
│   ├── 示例-销售数据.csv
│   └── output/            # 工具整理后的产物（自动生成，已 gitignore）
├── tests/test_tools.py    # 离线单元测试
└── requirements.txt
```

## 🚀 快速开始

```powershell
# 1) 准备虚拟环境（本机已建好，位于 D:\Dev\Env\py_venv\ai-tool-assistant）
#    若在其他电脑：python -m venv .venv 后激活

# 2) 安装依赖
& 'D:\Dev\Env\py_venv\ai-tool-assistant\Scripts\python.exe' -m pip install -r requirements.txt

# 3) 配置 DeepSeek API Key
#    复制 .env.example 为 .env，填入 Key（申请：https://platform.deepseek.com）

# 4) 启动网页对话
& 'D:\Dev\Env\py_venv\ai-tool-assistant\Scripts\python.exe' -m streamlit run app.py
```

浏览器会自动打开 http://localhost:8501 ，试试输入：
- `帮我读一下 data/示例-待办.txt`
- `北京和上海今天天气怎么样？`
- `整理 data/示例-销售数据.csv，看看数据有什么问题`

## 🧠 Agent 是怎么工作的（面试/学习要点）

1. **工具注册**：把每个工具的"说明书"（名称、功能、参数 JSON Schema）告诉模型 —— `agent/tools.py` 里的 `TOOLS`。
2. **模型决策**：DeepSeek 看完你的话 + 工具说明书，返回 `tool_calls`（决定调哪个工具、传什么参数）。
3. **本地执行**：LangGraph 的 `tools` 节点真正运行 Python 函数，结果作为 `ToolMessage` 回传。
4. **循环直到完成**：模型看到工具结果后继续推理——还要调工具就再进 `tools` 节点，不需要了就输出最终答案（LangGraph 条件路由控制）。

```
用户提问
   │
   ▼
[agent] 模型判断 ──要调工具──► [tools] 本地执行
   │  ▲                            │
   │  └───────── 结果回传 ──────────┘
   ▼ 不需要调工具
最终回答
```

## ✅ 运行测试（离线，不需要 Key）

```powershell
& 'D:\Dev\Env\py_venv\ai-tool-assistant\Scripts\python.exe' -m unittest discover -s tests -v
```

## 🗺️ 后续规划（Roadmap）

- [ ] 接入 LangGraph 记忆（checkpointer），支持多轮上下文记忆
- [ ] 增加 RAG 知识库问答（参考 lifepilot-agent）
- [ ] 增加"网页搜索 / 生成图表"等更多工具
- [ ] 支持本地大模型（Ollama）切换，降低成本

## 📚 参考学习资料

- datawhalechina/hello-agents ——《从零开始构建智能体》中文教程
- 2182977liu-bit/awesome-ai-agent-learning —— 不依赖框架从零搭 Agent 的保姆级中文教程
- AFPyannian/lifepilot-agent —— LangGraph + Streamlit + DeepSeek + RAG 的中文 Agent 参考实现