# MemoryChat - AI 纪念聊伴

基于 RAG (检索增强生成) 技术的 AI 纪念聊伴系统,为逝者家属提供温暖的陪伴。

## 🌟 项目特点

- **RAG 架构**: 检索增强生成,避免 AI 幻觉
- **多 LLM 支持**: OpenAI, Anthropic Claude
- **多向量库**: ChromaDB, 内存存储
- **安全优先**: 内置敏感内容过滤,紧急情况检测
- **隐私保护**: 本地部署选项,数据加密
- **Persona 学习**: 自动提取语言特征

## 🚀 快速开始

### 1. 安装依赖

```bash
# 基础依赖
pip install -r requirements.txt

# 或单独安装
pip install python-dotenv sentence-transformers chromadb numpy

# 可选: LLM 提供商
pip install openai anthropic
```

### 2. 配置环境变量

```bash
# 复制示例配置
cp .env.example .env

# 编辑 .env 文件,填入 API Key
# 如果使用本地 SBERT,无需 API Key
```

### 3. 运行演示

```bash
python demo.py
```

## 📁 项目结构

```
MemoryChat/
├── src/
│   ├── parsers/          # 数据解析器
│   │   └── whatsapp_parser.py
│   ├── rag/              # RAG 核心
│   │   ├── vectorizer.py
│   │   └── rag_pipeline.py
│   └── utils/            # 工具模块
│       └── safety_filter.py
├── data/
│   └── sample/           # 示例数据
├── tests/                # 测试文件
├── demo.py               # 演示脚本
├── requirements.txt      # 依赖列表
├── .env.example          # 配置模板
└── README.md
```

## 🔧 核心模块

### 1. WhatsApp 解析器

解析 WhatsApp 导出的聊天记录:

```python
from src.parsers import WhatsAppParser

parser = WhatsAppParser()
messages = parser.parse_file("chat.txt")
cleaned = parser.clean_messages(messages)
persona = parser.extract_persona_features(cleaned, "张三")
```

### 2. 向量化

支持多种 embedding 和向量库:

```python
from src.rag import create_vectorizer

vectorizer = create_vectorizer(
    provider="sbert",  # 或 "openai"
    store="chroma"     # 或 "simple"
)

documents = vectorizer.process_messages(messages)
vectorizer.index_documents(documents)
```

### 3. RAG Pipeline

检索增强生成:

```python
from src.rag import create_rag_pipeline
from src.utils import create_safety_filter

safety_filter = create_safety_filter(persona_name="张三")

rag = create_rag_pipeline(
    vectorizer=vectorizer,
    llm_provider="openai",
    persona_name="张三",
    persona_features=persona,
    safety_filter=safety_filter
)

response = rag.query("你最喜欢什么?")
print(response.response)
```

## 🛡️ 安全与伦理

### 内置安全机制

1. **敏感话题检测**: 自杀、自残、暴力等
2. **专业建议拦截**: 医疗、法律、财务建议
3. **隐私保护**: 自动检测并过滤敏感信息
4. **AI 身份标识**: 所有响应明确标注 AI 身份
5. **紧急情况响应**: 提供专业求助热线

### 使用原则

- ✅ 必须获得明确法律授权
- ✅ 家属可随时删除数据
- ✅ 透明标识 AI 身份
- ✅ 配合心理咨询资源
- ❌ 不提供专业建议
- ❌ 不冒充真人

## 📊 配置选项

### Embedding 提供商

- `sbert`: 本地模型,无需 API (推荐用于测试)
- `openai`: OpenAI embeddings (需要 API Key)

### 向量存储

- `simple`: 内存存储,适合测试
- `chroma`: ChromaDB,持久化存储

### LLM 提供商

- `openai`: GPT-3.5/GPT-4
- `anthropic`: Claude 3

## 🔍 示例用法

### 基础问答

```python
# 单轮问答
response = rag.query("你喜欢什么花?")

# 多轮对话
from src.rag import ChatMessage

conversation = [
    ChatMessage(role="user", content="你好"),
    ChatMessage(role="assistant", content="你好!"),
    ChatMessage(role="user", content="你最近在做什么?")
]

response = rag.chat(conversation)
```

### 数据导入

```python
# 导入 WhatsApp 聊天记录
parser = WhatsAppParser()
messages = parser.parse_file("WhatsApp Chat.txt")

# 清洗数据
cleaned = parser.clean_messages(
    messages,
    remove_system=True,
    remove_media=True,
    min_length=1
)

# 提取特征
persona = parser.extract_persona_features(cleaned, "逝者姓名")
```

## 🧪 测试

```bash
# 运行演示
python demo.py

# 测试安全过滤器
python src/utils/safety_filter.py
```

## 📝 开发路线图

- [x] WhatsApp 解析器
- [x] RAG Pipeline
- [x] 安全过滤模块
- [x] 交互式 Demo
- [ ] Web API (FastAPI)
- [ ] Web UI
- [ ] 数据管理控制台
- [ ] WhatsApp Business API 集成
- [ ] 多语言支持
- [ ] 用户认证与权限

## ⚠️ 重要提醒

本项目涉及敏感的情感与伦理问题:

1. **法律合规**: 使用前请咨询法律顾问
2. **心理健康**: 建议配合专业心理咨询
3. **数据安全**: 务必加密存储,设置访问控制
4. **透明度**: 始终明确标识 AI 身份
5. **可控性**: 家属随时可删除所有数据

## 📄 许可证

本项目仅供学习与研究使用,不得用于商业目的。

## 🤝 贡献

欢迎提交 Issue 和 Pull Request。

## 📞 支持

如遇到问题,请提交 Issue 或联系开发者。

---

**重要**: 本系统不能替代专业的心理咨询和丧亲辅导服务。
