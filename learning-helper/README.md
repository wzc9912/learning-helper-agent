# 学习助手 Agent

基于 Langgraph 的学习助手 Agent，使用 OpenAI API，支持 RAG（检索增强生成）系统学习任何主题。

## 功能特性

- 🔍 **智能搜索**: 使用 Tavily API 自动搜索相关学习材料
- 📥 **网页下载**: 使用 Playwright 下载支持动态 JavaScript 内容的网页
- 📚 **知识库管理**: 基于 ChromaDB 的向量存储和 BM25 稀疏检索
- 🔄 **RRF 融合**: 结合密集向量和稀疏检索的混合搜索
- 🗂️ **智能分块**: 基于 Markdown 结构的文档分块
- 💬 **多轮对话**: 支持上下文的多轮对话
- 📝 **引用标注**: 自动标注回答的内容来源

## 安装步骤

1. **克隆项目**
```bash
git clone <repository-url>
cd learning-helper
```

2. **安装 Python 依赖**
```bash
pip install -r requirements.txt
```

3. **配置环境变量**
```bash
cp .env.example .env
```

编辑 `.env` 文件，填入你的 API keys:
```
OPENAI_API_KEY=your_openai_api_key_here
TAVILY_API_KEY=your_tavily_api_key_here
```

## 运行程序

```bash
python -m src.main
```

## 项目结构

```
learning-helper/
├── src/
│   ├── main.py                    # 程序入口
│   ├── config.py                  # 配置文件
│   ├── graph/
│   │   ├── state.py               # Langgraph 状态定义
│   │   ├── graph.py               # Langgraph 状态图构建
│   │   └── nodes.py               # 各个节点函数
│   ├── tools/
│   │   ├── search.py              # 网页搜索工具
│   │   ├── scraper.py             # 网页下载工具
│   │   └── embeddings.py          # 嵌入模型封装
│   ├── retrieval/
│   │   ├── vector_store.py        # ChromaDB 封装
│   │   ├── bm25.py                # BM25 稀疏检索
│   │   └── fusion.py              # RRF 融合
│   ├── processing/
│   │   ├── chunking.py            # 文档分块策略
│   │   └── kb_summary.py          # 知识库总结管理
│   └── utils/
│       └── prompts.py             # LLM 提示词模板
├── data/
│   ├── knowledge_base/            # ChromaDB 数据存储
│   ├── documents/                 # 下载的原始文档
│   └── kb_summary.txt             # 知识库总结文件
├── requirements.txt
└── README.md
```

## 技术栈

- **Langgraph**: 构建状态机和 Agent 工作流
- **Langchain**: LLM 和向量存储集成
- **ChromaDB**: 向量数据库
- **Requests + BeautifulSoup**: 网页抓取
- **Rank-BM25**: BM25 稀疏检索
- **OpenAI API**: LLM 和嵌入模型

## 使用示例

```
💬 请输入你的问题: 我想学习 Langgraph 中的 Checkpointer

🔍 优化后的查询: Langgraph Checkpointer 使用教程
🌐 知识库暂无相关内容，开始搜索...

============================================================
找到以下相关网页：
============================================================

1. Langgraph Checkpointer 指南
   URL: https://langchain-ai.github.io/langgraph/concepts/low_level/#checkpointer
   简介: Checkpointer 是 Langgraph 中用于状态持久化的核心组件...

============================================================
输入 'y' 下载该页面
输入 'n' 或 'n: 拒绝理由' 重新搜索（例如：n: 结果不相关）
============================================================
你的选择: y

📥 开始下载网页...
✓ 成功下载 1 个网页
✓ 文档存储完成
✓ 检索到 5 个相关文档块

============================================================
🤖 助手回答:
============================================================
Checkpointer 是 Langgraph 中用于状态持久化的核心组件...
============================================================
```

### 用户交互说明

- **y** - 确认下载该网页
- **n** - 取消下载，系统会重新搜索
- **n: 拒绝理由** - 取消下载并提供原因，系统会根据原因优化搜索
  - 例如：`n: 结果不相关`
  - 例如：`n: 需要更详细的教程`

## License

MIT
