"""
RAG (检索增强生成) Pipeline
核心业务逻辑: 检索 -> 构建 Prompt -> LLM 生成 -> 安全过滤
"""
import os
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
import json

# LLM 提供商
try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

try:
    from anthropic import Anthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False


@dataclass
class ChatMessage:
    """聊天消息"""
    role: str  # user, assistant, system
    content: str


@dataclass
class RAGResponse:
    """RAG 响应"""
    response: str
    retrieved_docs: List[Dict]
    persona_name: str
    metadata: Dict


class LLMProvider:
    """LLM 提供商基类"""

    def generate(self, messages: List[ChatMessage], max_tokens: int = 500) -> str:
        raise NotImplementedError


class OpenAIProvider(LLMProvider):
    """OpenAI LLM"""

    def __init__(self, api_key: str, model: str = "gpt-3.5-turbo"):
        if not OPENAI_AVAILABLE:
            raise ImportError("需要安装 openai: pip install openai")

        self.client = OpenAI(api_key=api_key)
        self.model = model

    def generate(self, messages: List[ChatMessage], max_tokens: int = 500) -> str:
        openai_messages = [
            {"role": msg.role, "content": msg.content}
            for msg in messages
        ]

        response = self.client.chat.completions.create(
            model=self.model,
            messages=openai_messages,
            max_tokens=max_tokens
        )

        return response.choices[0].message.content


class AnthropicProvider(LLMProvider):
    """Anthropic Claude LLM"""

    def __init__(self, api_key: str, model: str = "claude-3-haiku-20240307"):
        if not ANTHROPIC_AVAILABLE:
            raise ImportError("需要安装 anthropic: pip install anthropic")

        self.client = Anthropic(api_key=api_key)
        self.model = model

    def generate(self, messages: List[ChatMessage], max_tokens: int = 500) -> str:
        # Claude 需要分离 system message
        system_message = None
        claude_messages = []

        for msg in messages:
            if msg.role == "system":
                system_message = msg.content
            else:
                claude_messages.append({
                    "role": msg.role,
                    "content": msg.content
                })

        kwargs = {
            "model": self.model,
            "messages": claude_messages,
            "max_tokens": max_tokens
        }

        if system_message:
            kwargs["system"] = system_message

        response = self.client.messages.create(**kwargs)
        return response.content[0].text


class PromptBuilder:
    """Prompt 构建器"""

    def __init__(self, persona_name: str, persona_features: Optional[Dict] = None):
        self.persona_name = persona_name
        self.persona_features = persona_features or {}

    def build_system_prompt(self) -> str:
        """构建系统 prompt"""
        common_words = self.persona_features.get("common_words", [])
        common_words_str = "、".join(common_words[:10]) if common_words else "无特定常用词"

        prompt = f"""你是一个基于已故 {self.persona_name} 的历史聊天记录与笔记构建的对话助手。

重要规则:
1. 在语气与用词上尽量贴合 {self.persona_name},但不要声称你是真人 {self.persona_name}
2. 在每次回答开头标注: [基于 {self.persona_name} 历史的 AI 模拟]
3. 如果涉及医疗/法律/财务建议,请拒绝并建议咨询专业人士
4. 如果检索到的内容涉及他人隐私且没有授权,避免透露
5. 基于检索到的历史对话进行回答,不要虚构信息
6. 如果检索结果与问题不相关,请诚实说明无法回答

{self.persona_name} 的语言特征:
- 常用词汇: {common_words_str}
- 平均消息长度: {self.persona_features.get('avg_message_length', '未知')} 字符

请以温暖、真诚的态度帮助用户缅怀 {self.persona_name}。"""

        return prompt

    def build_user_prompt(self, query: str, retrieved_contexts: List[str]) -> str:
        """构建用户 prompt"""
        if not retrieved_contexts:
            context_str = "未找到相关的历史对话记录。"
        else:
            context_str = "\n\n---\n\n".join(
                f"历史记录 {i+1}:\n{ctx}"
                for i, ctx in enumerate(retrieved_contexts)
            )

        prompt = f"""以下是从 {self.persona_name} 的历史记录中检索到的相关内容:

{context_str}

用户问题: {query}

请基于上述历史记录,以 {self.persona_name} 的语气和风格回答用户的问题。记住要在回答开头标注 AI 身份。"""

        return prompt


class RAGPipeline:
    """RAG Pipeline"""

    def __init__(
        self,
        vectorizer,
        llm_provider: LLMProvider,
        persona_name: str,
        persona_features: Optional[Dict] = None,
        safety_filter=None
    ):
        self.vectorizer = vectorizer
        self.llm_provider = llm_provider
        self.persona_name = persona_name
        self.persona_features = persona_features or {}
        self.safety_filter = safety_filter
        self.prompt_builder = PromptBuilder(persona_name, persona_features)

    def query(
        self,
        user_question: str,
        top_k: int = 5,
        max_tokens: int = 500
    ) -> RAGResponse:
        """
        执行 RAG 查询

        Args:
            user_question: 用户问题
            top_k: 检索文档数量
            max_tokens: 生成最大 token 数

        Returns:
            RAG 响应
        """
        # 1. 检索相关文档
        retrieved_docs = self.vectorizer.search(user_question, top_k=top_k)

        # 2. 提取文档内容
        contexts = [doc.content for doc, score in retrieved_docs]

        # 3. 构建 prompt
        system_prompt = self.prompt_builder.build_system_prompt()
        user_prompt = self.prompt_builder.build_user_prompt(user_question, contexts)

        messages = [
            ChatMessage(role="system", content=system_prompt),
            ChatMessage(role="user", content=user_prompt)
        ]

        # 4. LLM 生成
        response = self.llm_provider.generate(messages, max_tokens=max_tokens)

        # 5. 安全过滤 (如果有)
        if self.safety_filter:
            filtered_response = self.safety_filter.filter(response, user_question)
            if filtered_response.get("blocked", False):
                response = f"[系统提示] 出于安全考虑,无法生成此回答。原因: {filtered_response.get('reason', '未知')}"

        # 6. 构建响应
        return RAGResponse(
            response=response,
            retrieved_docs=[
                {
                    "content": doc.content,
                    "score": score,
                    "metadata": doc.metadata
                }
                for doc, score in retrieved_docs
            ],
            persona_name=self.persona_name,
            metadata={
                "query": user_question,
                "retrieved_count": len(retrieved_docs),
                "max_tokens": max_tokens
            }
        )

    def chat(
        self,
        conversation_history: List[ChatMessage],
        top_k: int = 3
    ) -> RAGResponse:
        """
        支持多轮对话的聊天

        Args:
            conversation_history: 对话历史
            top_k: 检索文档数量

        Returns:
            RAG 响应
        """
        # 提取最后一条用户消息
        last_user_message = None
        for msg in reversed(conversation_history):
            if msg.role == "user":
                last_user_message = msg.content
                break

        if not last_user_message:
            raise ValueError("对话历史中没有用户消息")

        # 使用最后的用户消息进行检索
        retrieved_docs = self.vectorizer.search(last_user_message, top_k=top_k)
        contexts = [doc.content for doc, score in retrieved_docs]

        # 构建完整的对话上下文
        system_prompt = self.prompt_builder.build_system_prompt()

        # 在第一条用户消息中注入检索内容
        enhanced_history = [ChatMessage(role="system", content=system_prompt)]

        for i, msg in enumerate(conversation_history):
            if msg.role == "user" and msg.content == last_user_message:
                # 增强最后一条用户消息
                enhanced_content = self.prompt_builder.build_user_prompt(msg.content, contexts)
                enhanced_history.append(ChatMessage(role="user", content=enhanced_content))
            else:
                enhanced_history.append(msg)

        # 生成响应
        response = self.llm_provider.generate(enhanced_history)

        # 安全过滤
        if self.safety_filter:
            filtered_response = self.safety_filter.filter(response, last_user_message)
            if filtered_response.get("blocked", False):
                response = f"[系统提示] 出于安全考虑,无法生成此回答。原因: {filtered_response.get('reason', '未知')}"

        return RAGResponse(
            response=response,
            retrieved_docs=[
                {
                    "content": doc.content,
                    "score": score,
                    "metadata": doc.metadata
                }
                for doc, score in retrieved_docs
            ],
            persona_name=self.persona_name,
            metadata={
                "query": last_user_message,
                "retrieved_count": len(retrieved_docs),
                "conversation_turns": len(conversation_history)
            }
        )


def create_rag_pipeline(
    vectorizer,
    llm_provider: str = "openai",
    persona_name: str = "逝者",
    persona_features: Optional[Dict] = None,
    safety_filter=None,
    **kwargs
) -> RAGPipeline:
    """
    工厂函数创建 RAG Pipeline

    Args:
        vectorizer: Vectorizer 实例
        llm_provider: LLM 提供商 (openai, anthropic)
        persona_name: Persona 名称
        persona_features: Persona 特征
        safety_filter: 安全过滤器
        **kwargs: 额外参数

    Returns:
        RAGPipeline 实例
    """
    # 创建 LLM provider
    if llm_provider == "openai":
        api_key = kwargs.get("openai_api_key") or os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("需要提供 openai_api_key 或设置 OPENAI_API_KEY 环境变量")
        model = kwargs.get("openai_model", "gpt-3.5-turbo")
        llm = OpenAIProvider(api_key, model)
    elif llm_provider == "anthropic":
        api_key = kwargs.get("anthropic_api_key") or os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError("需要提供 anthropic_api_key 或设置 ANTHROPIC_API_KEY 环境变量")
        model = kwargs.get("anthropic_model", "claude-3-haiku-20240307")
        llm = AnthropicProvider(api_key, model)
    else:
        raise ValueError(f"不支持的 llm_provider: {llm_provider}")

    return RAGPipeline(
        vectorizer=vectorizer,
        llm_provider=llm,
        persona_name=persona_name,
        persona_features=persona_features,
        safety_filter=safety_filter
    )


if __name__ == "__main__":
    print("RAG Pipeline 已就绪")
    print(f"OpenAI 可用: {OPENAI_AVAILABLE}")
    print(f"Anthropic 可用: {ANTHROPIC_AVAILABLE}")
