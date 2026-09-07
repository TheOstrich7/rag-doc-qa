# 星海 RAG · 文档问答系统

> LangChain + FAISS + DeepSeek 做的最小可用 RAG，覆盖 **上传 PDF → 切分 → 检索 → 大模型回答**、**纯大模型 vs RAG 评测**、**ReAct Agent 自主决定查不查知识库**三件事。

## 阶段进度

| 阶段 | 目标 | 状态 |
| --- | --- | --- |
| [1] 最小可用版 | 上传 PDF → 切分 → 向量化 → 检索 → 大模型回答 | ✅ 完成 |
| [2] 检索评估 | 20–30 个问题，纯大模型 vs RAG 数字对比 | ✅ 完成（数字待你填 API Key 后跑） |
| [3] Agent 能力 | Function Calling：让模型自己决定查知识库 / 算数 / 查时间 | ✅ 完成 |

## 目录结构

```
rag-doc-qa/
├── README.md
├── requirements.txt            # 依赖
├── .env.example                # API Key 模板
├── pytest.ini
├── data/                       # 语料 PDF（生成后会放这里）
├── eval/questions.jsonl        # 29 题评测集（27 可答 + 2 越界）
├── scripts/
│   ├── make_sample_pdf.py      # 生成《星海科技内部知识库手册》合成 PDF
│   └── patch_readme.py         # 把评测结果回填到 README
├── ragdoc/
│   ├── config.py               # pydantic-settings 配置
│   ├── embeddings.py           # 三种 Embedding 后端（fastembed/hf/hash + auto 降级）
│   ├── loaders.py              # PDF 加载 + 标题感知归一化 + 中文切分
│   ├── vectorstore.py          # FAISS 构建/落盘/检索
│   ├── llm.py                  # DeepSeek 工厂（OpenAI 兼容协议）
│   ├── chains/
│   │   ├── prompts.py          # RAG / Naive / Judge 三套 prompt
│   │   ├── rag.py              # RAG 主链路
│   │   └── naive.py            # 纯大模型基线
│   ├── agent/
│   │   ├── tools.py            # 知识库检索 / 计算器 / 当前时间
│   │   └── graph.py            # LangGraph ReAct Agent
│   ├── evals/                  # 评测：数据集 / 指标 / 执行器
│   └── cli.py                  # 命令行入口
├── app/streamlit_app.py        # Web 界面（上传 + 聊天）
├── tests/                      # 25 个单元测试（不依赖 API Key）
├── docs/开发流程.md            # 完整软件工程记录
└── results/                    # 评测产出（report.md / report.json / compare.png）
```

## 快速开始

```bash
# 1) 装依赖
python -m venv .venv && source .venv/Scripts/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# 2) 填 API Key
cp .env.example .env                                       # Linux/macOS
copy .env.example .env                                     # Windows
# 编辑 .env，把 DEEPSEEK_API_KEY=sk-xxx 填好

# 3) 生成合成语料（第一次跑就行；以后换自己的 PDF 也可）
python scripts/make_sample_pdf.py

# 4) 建索引
python -m ragdoc.cli ingest                                 # 默认读 data/*.pdf
# 或：python -m ragdoc.cli ingest --pdf your.pdf

# 5) 问
python -m ragdoc.cli ask "去深圳出差住宿标准是多少？"
python -m ragdoc.cli ask "P0 故障响应时间？" --show-context

# 6) 跑评测 + 回填 README
python -m ragdoc.cli eval                                   # 跑全部 29 题
python scripts/patch_readme.py                              # 数字回填到下方表格

# 7) 看 Agent
python -m ragdoc.cli agent "差旅补贴每天多少？出差 3 天 5 晚能领多少？" --show-trace
python -m ragdoc.cli agent "公司附近有什么好吃的" --show-trace     # Agent 决定不查知识库

# 8) 起 Web 界面（可选）
streamlit run app/streamlit_app.py
```

## 架构

```
                          ┌────────────────────────┐
                          │  PDF 文档（data/）     │
                          └──────────┬─────────────┘
                                     │ PyPDFLoader
                                     ▼
                          ┌────────────────────────┐
                          │  标题感知归一化         │  修掉 PDF 断行
                          │  + 中文递归切分         │  切到 chunk_size=400
                          └──────────┬─────────────┘
                                     │  Document[]
                                     ▼
   ┌────────────┐   build_index    ┌────────────────────────┐
   │  Embedding │ ───────────────▶ │  FAISS 向量库           │
   │  fastembed │                  │  ~/.ragdoc_index/       │  ← 用 ASCII 路径，
   │  / hf / hash                  │  index.faiss            │    避开 faiss-cpu
   │   (auto)    │                  │  index.pkl              │    在 Windows 中文
   └────────────┘                  └──────────┬─────────────┘    路径下写不进去
                                               │ retriever (MMR, top_k=4)
                                               ▼
   ┌────────────┐                  ┌────────────────────────┐
   │ DeepSeek   │ ◀── context ──── │   RAG 链路              │
   │ (ChatOpenAI│                  │   format_docs → prompt  │
   │  兼容协议) │ ──── answer ────▶ │   → llm → str          │
   └────────────┘                  └────────┬───────────────┘
                                              │
                                              ▼
   [3] ReAct Agent：再在 RAG 之上套一层"决策"
       search_knowledge_base / calculator / get_current_datetime
       模型自己选
```

## 阶段 1：最小可用版

- `python -m ragdoc.cli ingest` 一行建索引
- `python -m ragdoc.cli ask "..."` 单轮问答，可选 `--show-context` 看召回片段
- `streamlit run app/streamlit_app.py` 上传 PDF → 聊天
- **检索方式**：MMR（默认 fetch_k=20, λ=0.5），降低结果同质化
- **Embedding**：默认 fastembed + `BAAI/bge-small-zh-v1.5`（约 90MB ONNX 权重）。在没有网络 / 想冒烟时切到 `hash` 后端
- **注意**：faiss-cpu 在 Windows 含中文路径下会写入失败（已知问题），所以索引默认放在 `~/.ragdoc_index/`。如果项目根目录是纯 ASCII，可用 `INDEX_DIR` 覆盖

## 阶段 2：检索评估

29 题评测集（27 可回答 + 2 越界）位于 `eval/questions.jsonl`，每条都标了 `gold_evidence`（用于自动判定检索是否命中）和 `keywords`（用于自动判分关键数字）。

```bash
python -m ragdoc.cli eval                     # 跑全量（默认同时跑基线和 RAG）
python -m ragdoc.cli eval --limit 10          # 快速试跑前 10 题
python -m ragdoc.cli eval --no-baseline       # 只跑 RAG（省一半 token）
python -m ragdoc.cli eval --category "差旅报销"   # 按类别过滤
```

输出写到 `results/`：
- `report.md`：人可读的对比表
- `report.json`：完整 JSON
- `raw_records.jsonl`：每题明细（含基线 / RAG 答案 + 判分）
- `compare.png`：柱状图（依赖 matplotlib，没装也能跑，只是少一张图）

<!-- EVAL_TABLE_START -->
> ⚠️ 下面这张表是 `python -m ragdoc.cli eval` 跑出来的最新数字。如果显示全 0 / 100% 幻觉，**通常是因为还没在 `.env` 里填 DEEPSEEK_API_KEY**，框架跑通但大模型那边没真正接上。

## 总体对比



| 指标 | 纯大模型 | RAG | 提升 |
| --- | --- | --- | --- |
| 答案正确率（全量，LLM 判分） | 41.4% | 91.4% | +50.0pp |
| 答案正确率（可回答题） | 37.0% | 90.7% | +53.7pp |
| 关键数字覆盖率 | 40.7% | 64.8% | +24.1pp |
| 越界问题正确拒答率 | 100.0% | 100.0% | +0.0pp |
| 有据可依率（忠实度，越低越爱编） | - | 96.6% | - |
| 幻觉率 | - | 3.4% | - |
| 检索召回率@k | - | 48.1% | - |
| 检索 MRR | - | 48.1% | - |
| 可回答题误拒答率 | - | 33.3% | - |
| 平均延迟 | 1902 ms | 758 ms | - |

耗时 132.3 秒。

<!-- EVAL_TABLE_END -->

### 指标怎么算的

| 指标 | 类型 | 怎么算的 | 为什么用它 |
| --- | --- | --- | --- |
| 答案正确率 | LLM-as-Judge | DeepSeek 给每题打 0/1/2 分（对比参考答案），折成 0/0.5/1 | 真正的事实可比，纯关键词覆盖会漏掉语义对的句子 |
| 关键数字覆盖率 | 确定性 | `keywords` 中出现在答案里的比例 | 一眼能看出"具体数字答出来没有"，避免 Judge 主观漂移 |
| 检索召回率@k | 确定性 | top-k 里是否包含 `gold_evidence` 所在 chunk | 直接反映"找没找到" |
| MRR | 确定性 | 首个包含 gold_evidence 的排序倒数 | 越相关排得越前，分数越高 |
| RAG 忠实度 / 幻觉率 | LLM-as-Judge | 让 DeepSeek 判：答案中每条事实是否都能在召回原文里找到依据 | 这是 RAG 存在的根本理由——不能凭空捏造 |
| 越界题拒答率 | 规则 + LLM | 答案包含"无法回答/不确定/未找到"等标记算拒答 | 体现"不知道就说不"的能力，比在测试集上过拟合强 |
| 平均延迟 | 计时 | 端到端 wall-clock | 工程视角：够不够快 |

## 阶段 3：Agent 能力

用 LangGraph 的 `create_react_agent` 把以下工具挂上：

| 工具 | 作用 | 何时会被调用 |
| --- | --- | --- |
| `search_knowledge_base(query)` | 查知识库并返回 top-4 原文片段 | 问题涉及公司制度/金额/天数/SLA 等 |
| `calculator(expr)` | 用 AST 解析做安全算术，**绝不用 eval** | 需要数字计算时 |
| `get_current_datetime()` | 当前时间 | 涉及"今天/还剩多少天"等 |

```bash
python -m ragdoc.cli agent "去深圳出差 5 天 4 晚，住宿能报多少钱？" --show-trace
# Agent 轨迹：
#   1. 调用 search_knowledge_base("深圳 出差 住宿")
#   2. 调用 calculator("600 * 4")
#   3. 合成答案

python -m ragdoc.cli agent "公司附近有什么好吃的" --show-trace
# Agent 决定：直接回答（不调任何工具）
```

## 设计取舍

- **Embedding 用 fastembed 而非 OpenAI**——DeepSeek 不提供 embedding，单独再调一个 OpenAI Key 增加耦合。fastembed 跑 bge-small-zh 中文表现不差，下载量小，不依赖 torch
- **切分用中文标点优先级的递归切分**——`["\n\n","\n","。","！","？","；","，","、"," ",""]`，配合"标题感知归一化"，避免把"第 X 章 差旅与费用报销"和正文拼成一段
- **评测用 LLM-as-Judge + 关键词覆盖双轨**——Judge 看语义、关键词卡数字，互补
- **RAG 和 Naive 用同一套 LLM 同一份 prompt 模板**——唯一的变量是"有没有给参考资料"，对比才有说服力
- **faiss 索引默认放 `~/.ragdoc_index`**——faiss-cpu 在 Windows 中文路径下 C++ I/O 报错，挪到家目录就稳了
- **Agent 走 LangGraph 而不是自己实现 ReAct**——稳定、可观测、能复用 checkpoint

## 已知限制

- LLM-as-Judge 自身有 5-10% 噪音（不同 temperature 跑同一题会判分不同），所以**单次跑的结果别太较真**；要看趋势
- `langchain-community` 已被官方 sunset，未来要切到独立集成包（`langchain-faiss` / `langchain-huggingface` 等），代码改动很小
- 中文 PDF 排版千奇百怪，本归一化对"句末标点+换行"的假设在某些扫描件上不成立

## 测试

```bash
pytest                                    # 25 个单测，不依赖 API Key
```

涵盖：归一化边界、HashingEmbeddings 稳定性、端到端冒烟、计算器安全性、确定性指标。
