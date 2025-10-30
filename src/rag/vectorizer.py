"""
向量化模块
支持文本 embedding 和向量数据库操作
"""
import os
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
import json

# 支持多种 embedding 提供商
try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

try:
    from sentence_transformers import SentenceTransformer
    SBERT_AVAILABLE = True
except ImportError:
    SBERT_AVAILABLE = False

try:
    import chromadb
    from chromadb.config import Settings
    CHROMA_AVAILABLE = True
except ImportError:
    CHROMA_AVAILABLE = False


@dataclass
class Document:
    """文档数据结构"""
    id: str
    content: str
    metadata: Dict
    embedding: Optional[List[float]] = None


class EmbeddingProvider:
    """Embedding 提供商基类"""

    def embed_text(self, text: str) -> List[float]:
        raise NotImplementedError

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        raise NotImplementedError


class OpenAIEmbedding(EmbeddingProvider):
    """OpenAI Embedding"""

    def __init__(self, api_key: str, model: str = "text-embedding-3-small"):
        if not OPENAI_AVAILABLE:
            raise ImportError("需要安装 openai: pip install openai")

        self.client = OpenAI(api_key=api_key)
        self.model = model

    def embed_text(self, text: str) -> List[float]:
        response = self.client.embeddings.create(
            input=text,
            model=self.model
        )
        return response.data[0].embedding

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        response = self.client.embeddings.create(
            input=texts,
            model=self.model
        )
        return [item.embedding for item in response.data]


class SBERTEmbedding(EmbeddingProvider):
    """Sentence-BERT Embedding (本地)"""

    def __init__(self, model_name: str = "paraphrase-multilingual-MiniLM-L12-v2"):
        if not SBERT_AVAILABLE:
            raise ImportError("需要安装 sentence-transformers: pip install sentence-transformers")

        self.model = SentenceTransformer(model_name)

    def embed_text(self, text: str) -> List[float]:
        return self.model.encode(text).tolist()

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        embeddings = self.model.encode(texts)
        return [emb.tolist() for emb in embeddings]


class VectorStore:
    """向量数据库基类"""

    def add_documents(self, documents: List[Document]):
        raise NotImplementedError

    def search(self, query_embedding: List[float], top_k: int = 5) -> List[Tuple[Document, float]]:
        raise NotImplementedError

    def delete_all(self):
        raise NotImplementedError


class ChromaVectorStore(VectorStore):
    """ChromaDB 向量存储"""

    def __init__(self, collection_name: str = "memory_chat", persist_directory: str = "./chroma_db"):
        if not CHROMA_AVAILABLE:
            raise ImportError("需要安装 chromadb: pip install chromadb")

        self.client = chromadb.PersistentClient(path=persist_directory)
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"}
        )

    def add_documents(self, documents: List[Document]):
        """添加文档到向量库"""
        ids = [doc.id for doc in documents]
        embeddings = [doc.embedding for doc in documents]
        contents = [doc.content for doc in documents]
        metadatas = [doc.metadata for doc in documents]

        self.collection.add(
            ids=ids,
            embeddings=embeddings,
            documents=contents,
            metadatas=metadatas
        )

    def search(self, query_embedding: List[float], top_k: int = 5) -> List[Tuple[Document, float]]:
        """搜索相似文档"""
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k
        )

        documents = []
        if results['ids'] and len(results['ids'][0]) > 0:
            for i in range(len(results['ids'][0])):
                doc = Document(
                    id=results['ids'][0][i],
                    content=results['documents'][0][i],
                    metadata=results['metadatas'][0][i] if results['metadatas'] else {},
                    embedding=results['embeddings'][0][i] if results['embeddings'] else None
                )
                score = results['distances'][0][i] if results['distances'] else 0.0
                documents.append((doc, score))

        return documents

    def delete_all(self):
        """删除所有文档"""
        self.client.delete_collection(self.collection.name)
        self.collection = self.client.create_collection(
            name=self.collection.name,
            metadata={"hnsw:space": "cosine"}
        )


class SimpleVectorStore(VectorStore):
    """简单的内存向量存储 (用于演示/测试)"""

    def __init__(self):
        self.documents: List[Document] = []

    def add_documents(self, documents: List[Document]):
        self.documents.extend(documents)

    def search(self, query_embedding: List[float], top_k: int = 5) -> List[Tuple[Document, float]]:
        """使用余弦相似度搜索"""
        import numpy as np

        scored_docs = []
        for doc in self.documents:
            if doc.embedding:
                # 计算余弦相似度
                similarity = np.dot(query_embedding, doc.embedding) / (
                    np.linalg.norm(query_embedding) * np.linalg.norm(doc.embedding)
                )
                scored_docs.append((doc, float(similarity)))

        # 按相似度降序排序
        scored_docs.sort(key=lambda x: x[1], reverse=True)
        return scored_docs[:top_k]

    def delete_all(self):
        self.documents = []


class Vectorizer:
    """统一的向量化接口"""

    def __init__(
        self,
        embedding_provider: EmbeddingProvider,
        vector_store: VectorStore
    ):
        self.embedding_provider = embedding_provider
        self.vector_store = vector_store

    def process_messages(
        self,
        messages: List[Dict],
        chunk_size: int = 5,
        overlap: int = 1
    ) -> List[Document]:
        """
        处理消息并创建文档

        Args:
            messages: 消息列表
            chunk_size: 每个文档包含的消息数
            overlap: 重叠消息数

        Returns:
            文档列表
        """
        documents = []
        i = 0

        while i < len(messages):
            # 提取消息块
            chunk_messages = messages[i:i + chunk_size]

            if not chunk_messages:
                break

            # 构建文档内容
            content_parts = []
            for msg in chunk_messages:
                sender = msg.get("sender", "Unknown")
                text = msg.get("content", "")
                content_parts.append(f"{sender}: {text}")

            content = "\n".join(content_parts)

            # 创建文档元数据
            metadata = {
                "start_time": chunk_messages[0].get("timestamp", ""),
                "end_time": chunk_messages[-1].get("timestamp", ""),
                "message_count": len(chunk_messages),
                "senders": list(set(msg.get("sender", "") for msg in chunk_messages))
            }

            # 生成唯一 ID
            doc_id = f"doc_{i}_{i + len(chunk_messages)}"

            # 生成 embedding
            embedding = self.embedding_provider.embed_text(content)

            doc = Document(
                id=doc_id,
                content=content,
                metadata=metadata,
                embedding=embedding
            )

            documents.append(doc)

            # 移动到下一个块 (考虑重叠)
            i += chunk_size - overlap

        return documents

    def index_documents(self, documents: List[Document]):
        """索引文档到向量库"""
        self.vector_store.add_documents(documents)

    def search(self, query: str, top_k: int = 5) -> List[Tuple[Document, float]]:
        """搜索相关文档"""
        query_embedding = self.embedding_provider.embed_text(query)
        return self.vector_store.search(query_embedding, top_k)


def create_vectorizer(
    provider: str = "sbert",
    store: str = "simple",
    **kwargs
) -> Vectorizer:
    """
    工厂函数创建 Vectorizer

    Args:
        provider: embedding 提供商 (openai, sbert)
        store: 向量存储 (chroma, simple)
        **kwargs: 额外参数

    Returns:
        Vectorizer 实例
    """
    # 创建 embedding provider
    if provider == "openai":
        api_key = kwargs.get("openai_api_key") or os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("需要提供 openai_api_key 或设置 OPENAI_API_KEY 环境变量")
        embedding_provider = OpenAIEmbedding(api_key)
    elif provider == "sbert":
        model_name = kwargs.get("sbert_model", "paraphrase-multilingual-MiniLM-L12-v2")
        embedding_provider = SBERTEmbedding(model_name)
    else:
        raise ValueError(f"不支持的 provider: {provider}")

    # 创建 vector store
    if store == "chroma":
        collection_name = kwargs.get("collection_name", "memory_chat")
        persist_directory = kwargs.get("persist_directory", "./chroma_db")
        vector_store = ChromaVectorStore(collection_name, persist_directory)
    elif store == "simple":
        vector_store = SimpleVectorStore()
    else:
        raise ValueError(f"不支持的 store: {store}")

    return Vectorizer(embedding_provider, vector_store)


if __name__ == "__main__":
    print("Vectorizer 模块已就绪")
    print(f"OpenAI 可用: {OPENAI_AVAILABLE}")
    print(f"SBERT 可用: {SBERT_AVAILABLE}")
    print(f"ChromaDB 可用: {CHROMA_AVAILABLE}")
