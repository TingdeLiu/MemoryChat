"""
MemoryChat RAG Demo
完整的 RAG pipeline 演示脚本
"""
import os
from pathlib import Path
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()

from src.parsers import WhatsAppParser
from src.rag import create_vectorizer, create_rag_pipeline
from src.utils import create_safety_filter


def main():
    """主演示函数"""
    print("=" * 60)
    print("MemoryChat RAG Demo")
    print("=" * 60)

    # ==================== 配置 ====================
    PERSONA_NAME = os.getenv("DEFAULT_PERSONA_NAME", "张三")
    EMBEDDING_PROVIDER = os.getenv("EMBEDDING_PROVIDER", "sbert")
    VECTOR_STORE = os.getenv("VECTOR_STORE", "simple")
    LLM_PROVIDER = os.getenv("LLM_PROVIDER", "openai")

    print(f"\n配置:")
    print(f"  Persona: {PERSONA_NAME}")
    print(f"  Embedding: {EMBEDDING_PROVIDER}")
    print(f"  Vector Store: {VECTOR_STORE}")
    print(f"  LLM: {LLM_PROVIDER}")

    # ==================== 1. 解析 WhatsApp 数据 ====================
    print("\n" + "-" * 60)
    print("步骤 1: 解析 WhatsApp 聊天记录")
    print("-" * 60)

    parser = WhatsAppParser()

    # 检查示例数据文件
    sample_file = Path("data/sample/chat.txt")
    if not sample_file.exists():
        print(f"⚠️  示例文件不存在: {sample_file}")
        print("\n创建示例数据...")
        create_sample_data()

    # 解析文件
    messages = parser.parse_file(str(sample_file))
    print(f"✓ 解析到 {len(messages)} 条消息")

    # 清洗数据
    cleaned_messages = parser.clean_messages(
        messages,
        remove_system=True,
        remove_media=True,
        min_length=1
    )
    print(f"✓ 清洗后 {len(cleaned_messages)} 条消息")

    # 提取 Persona 特征
    persona_features = parser.extract_persona_features(cleaned_messages, PERSONA_NAME)
    print(f"✓ 提取 Persona 特征:")
    print(f"  - 消息数: {persona_features.get('total_messages', 0)}")
    print(f"  - 平均长度: {persona_features.get('avg_message_length', 0):.1f} 字符")
    print(f"  - 常用词: {', '.join(persona_features.get('common_words', [])[:5])}")

    # ==================== 2. 向量化与索引 ====================
    print("\n" + "-" * 60)
    print("步骤 2: 向量化与索引")
    print("-" * 60)

    # 创建 vectorizer
    try:
        vectorizer = create_vectorizer(
            provider=EMBEDDING_PROVIDER,
            store=VECTOR_STORE,
            openai_api_key=os.getenv("OPENAI_API_KEY"),
            sbert_model=os.getenv("SBERT_MODEL"),
            collection_name=os.getenv("CHROMA_COLLECTION_NAME"),
            persist_directory=os.getenv("CHROMA_PERSIST_DIR")
        )
        print(f"✓ Vectorizer 创建成功")
    except Exception as e:
        print(f"❌ 创建 Vectorizer 失败: {e}")
        print("\n提示: 请确保安装了必要的依赖:")
        print("  pip install sentence-transformers chromadb")
        return

    # 处理消息并创建文档
    message_dicts = [msg.to_dict() for msg in cleaned_messages]
    documents = vectorizer.process_messages(
        message_dicts,
        chunk_size=int(os.getenv("CHUNK_SIZE", 5)),
        overlap=int(os.getenv("CHUNK_OVERLAP", 1))
    )
    print(f"✓ 创建 {len(documents)} 个文档块")

    # 索引文档
    vectorizer.index_documents(documents)
    print(f"✓ 文档已索引到向量库")

    # ==================== 3. 创建 RAG Pipeline ====================
    print("\n" + "-" * 60)
    print("步骤 3: 创建 RAG Pipeline")
    print("-" * 60)

    # 创建安全过滤器
    safety_filter = create_safety_filter(
        persona_name=PERSONA_NAME,
        enable_emergency_alert=os.getenv("ENABLE_EMERGENCY_ALERT", "true").lower() == "true"
    )
    print(f"✓ 安全过滤器已创建")

    # 创建 RAG pipeline
    try:
        rag = create_rag_pipeline(
            vectorizer=vectorizer,
            llm_provider=LLM_PROVIDER,
            persona_name=PERSONA_NAME,
            persona_features=persona_features,
            safety_filter=safety_filter,
            openai_api_key=os.getenv("OPENAI_API_KEY"),
            openai_model=os.getenv("OPENAI_MODEL"),
            anthropic_api_key=os.getenv("ANTHROPIC_API_KEY"),
            anthropic_model=os.getenv("ANTHROPIC_MODEL")
        )
        print(f"✓ RAG Pipeline 创建成功")
    except Exception as e:
        print(f"❌ 创建 RAG Pipeline 失败: {e}")
        print(f"\n提示: 请确保设置了 API Key:")
        print(f"  - OpenAI: 在 .env 中设置 OPENAI_API_KEY")
        print(f"  - Anthropic: 在 .env 中设置 ANTHROPIC_API_KEY")
        return

    # ==================== 4. 交互式问答 ====================
    print("\n" + "-" * 60)
    print("步骤 4: 交互式问答")
    print("-" * 60)
    print("\n输入你的问题,或输入 'quit' 退出\n")

    while True:
        try:
            user_input = input(f"你: ")

            if user_input.lower() in ["quit", "exit", "q", "退出"]:
                print("\n再见!")
                break

            if not user_input.strip():
                continue

            # 执行查询
            response = rag.query(
                user_question=user_input,
                top_k=int(os.getenv("RAG_TOP_K", 5)),
                max_tokens=int(os.getenv("MAX_TOKENS", 500))
            )

            # 显示响应
            print(f"\n{PERSONA_NAME} AI: {response.response}\n")

            # 显示检索的文档 (可选)
            if os.getenv("SHOW_RETRIEVED_DOCS", "false").lower() == "true":
                print("\n[检索的文档]")
                for i, doc in enumerate(response.retrieved_docs[:3]):
                    print(f"{i+1}. (相似度: {doc['score']:.3f}) {doc['content'][:100]}...")
                print()

        except KeyboardInterrupt:
            print("\n\n再见!")
            break
        except Exception as e:
            print(f"\n❌ 错误: {e}\n")


def create_sample_data():
    """创建示例数据文件"""
    sample_file = Path("data/sample/chat.txt")
    sample_file.parent.mkdir(parents=True, exist_ok=True)

    # 示例 WhatsApp 聊天记录
    sample_content = """01/01/2024, 10:30 - 张三: 早上好!今天天气真不错
01/01/2024, 10:31 - 李四: 是啊,我们去公园散步吧
01/01/2024, 10:32 - 张三: 好主意!我最喜欢在公园散步了,可以看到很多花
01/01/2024, 10:35 - 李四: 你最喜欢什么花?
01/01/2024, 10:36 - 张三: 我最喜欢玫瑰和向日葵,它们都很美
01/01/2024, 11:00 - 张三: 对了,你还记得我们上次一起去的那家咖啡馆吗?
01/01/2024, 11:01 - 李四: 记得,叫什么来着?
01/01/2024, 11:02 - 张三: 叫"时光咖啡",那里的拿铁超好喝
01/01/2024, 11:05 - 李四: 哈哈,你总是记得吃的
01/01/2024, 11:06 - 张三: 当然啦,美食是生活的一部分嘛
01/02/2024, 09:00 - 张三: 昨天散步太开心了!
01/02/2024, 09:05 - 李四: 是啊,下次再一起去
01/02/2024, 09:10 - 张三: 好的!我一直觉得和朋友在一起的时光最珍贵
01/02/2024, 09:15 - 李四: 我也是这么想的
01/02/2024, 09:20 - 张三: 对了,我最近在学吉他,虽然弹得不太好,但很享受
"""

    sample_file.write_text(sample_content, encoding="utf-8")
    print(f"✓ 示例数据已创建: {sample_file}")


if __name__ == "__main__":
    main()
