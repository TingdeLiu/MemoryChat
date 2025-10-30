"""
WhatsApp 聊天记录解析器
支持标准 WhatsApp 导出格式的解析与清洗
"""
import re
from datetime import datetime
from typing import List, Dict, Optional
from dataclasses import dataclass


@dataclass
class Message:
    """聊天消息数据结构"""
    timestamp: datetime
    sender: str
    content: str
    message_type: str = "text"  # text, media, system

    def to_dict(self) -> Dict:
        return {
            "timestamp": self.timestamp.isoformat(),
            "sender": self.sender,
            "content": self.content,
            "message_type": self.message_type
        }


class WhatsAppParser:
    """WhatsApp 聊天记录解析器"""

    # WhatsApp 导出格式正则表达式
    # 支持格式: [dd/mm/yyyy, hh:mm:ss] Sender: Message
    # 或: dd/mm/yyyy, hh:mm - Sender: Message
    PATTERNS = [
        r'\[(\d{1,2}/\d{1,2}/\d{4},\s\d{1,2}:\d{2}:\d{2})\]\s([^:]+):\s(.+)',
        r'(\d{1,2}/\d{1,2}/\d{4},\s\d{1,2}:\d{2})\s-\s([^:]+):\s(.+)',
        r'(\d{4}-\d{2}-\d{2},\s\d{1,2}:\d{2}:\d{2})\s-\s([^:]+):\s(.+)',
    ]

    # 系统消息关键词
    SYSTEM_KEYWORDS = [
        "Messages and calls are end-to-end encrypted",
        "创建了群组",
        "added",
        "left",
        "changed the subject",
        "changed this group's icon",
        "<Media omitted>",
        "图片已省略",
        "视频已省略",
        "音频已省略",
        "文件已省略"
    ]

    # 媒体占位符
    MEDIA_PATTERNS = [
        r"<Media omitted>",
        r"<attached:.*?>",
        r"\[图片\]",
        r"\[视频\]",
        r"\[音频\]",
        r"\[文件\]"
    ]

    def __init__(self):
        self.compiled_patterns = [re.compile(p) for p in self.PATTERNS]
        self.media_pattern = re.compile("|".join(self.MEDIA_PATTERNS))

    def parse_file(self, file_path: str, encoding: str = "utf-8") -> List[Message]:
        """
        解析 WhatsApp 导出文件

        Args:
            file_path: 文件路径
            encoding: 文件编码 (默认 utf-8, 可尝试 utf-8-sig)

        Returns:
            消息列表
        """
        messages = []

        try:
            with open(file_path, 'r', encoding=encoding) as f:
                content = f.read()
        except UnicodeDecodeError:
            # 尝试其他编码
            with open(file_path, 'r', encoding='utf-8-sig') as f:
                content = f.read()

        # 按行分割,但处理多行消息
        lines = content.split('\n')
        current_message = None

        for line in lines:
            line = line.strip()
            if not line:
                continue

            # 尝试匹配新消息
            parsed = self._parse_line(line)

            if parsed:
                # 保存之前的消息
                if current_message:
                    messages.append(current_message)
                current_message = parsed
            else:
                # 多行消息的延续部分
                if current_message:
                    current_message.content += "\n" + line

        # 添加最后一条消息
        if current_message:
            messages.append(current_message)

        return messages

    def _parse_line(self, line: str) -> Optional[Message]:
        """解析单行消息"""
        for pattern in self.compiled_patterns:
            match = pattern.match(line)
            if match:
                timestamp_str, sender, content = match.groups()

                # 解析时间戳
                timestamp = self._parse_timestamp(timestamp_str)
                if not timestamp:
                    continue

                # 判断消息类型
                message_type = self._classify_message(content, sender)

                return Message(
                    timestamp=timestamp,
                    sender=sender.strip(),
                    content=content.strip(),
                    message_type=message_type
                )

        return None

    def _parse_timestamp(self, timestamp_str: str) -> Optional[datetime]:
        """解析时间戳字符串"""
        # 尝试不同的时间格式
        formats = [
            "%d/%m/%Y, %H:%M:%S",
            "%m/%d/%Y, %H:%M:%S",
            "%d/%m/%Y, %H:%M",
            "%m/%d/%Y, %H:%M",
            "%Y-%m-%d, %H:%M:%S",
            "%Y-%m-%d, %H:%M",
        ]

        for fmt in formats:
            try:
                return datetime.strptime(timestamp_str, fmt)
            except ValueError:
                continue

        return None

    def _classify_message(self, content: str, sender: str) -> str:
        """分类消息类型"""
        # 检查是否是系统消息
        for keyword in self.SYSTEM_KEYWORDS:
            if keyword in content:
                return "system"

        # 检查是否是媒体消息
        if self.media_pattern.search(content):
            return "media"

        return "text"

    def clean_messages(
        self,
        messages: List[Message],
        remove_system: bool = True,
        remove_media: bool = True,
        min_length: int = 1
    ) -> List[Message]:
        """
        清洗消息列表

        Args:
            messages: 原始消息列表
            remove_system: 是否移除系统消息
            remove_media: 是否移除媒体消息
            min_length: 消息最小长度

        Returns:
            清洗后的消息列表
        """
        cleaned = []

        for msg in messages:
            # 过滤系统消息
            if remove_system and msg.message_type == "system":
                continue

            # 过滤媒体消息
            if remove_media and msg.message_type == "media":
                continue

            # 过滤短消息
            if len(msg.content) < min_length:
                continue

            cleaned.append(msg)

        return cleaned

    def extract_persona_features(self, messages: List[Message], target_sender: str) -> Dict:
        """
        提取目标发送者的 Persona 特征

        Args:
            messages: 消息列表
            target_sender: 目标发送者名称

        Returns:
            Persona 特征字典
        """
        target_messages = [m for m in messages if m.sender == target_sender]

        if not target_messages:
            return {}

        # 统计特征
        total_messages = len(target_messages)
        total_chars = sum(len(m.content) for m in target_messages)
        avg_length = total_chars / total_messages if total_messages > 0 else 0

        # 提取常用词 (简单实现,可优化)
        words = []
        for msg in target_messages:
            words.extend(msg.content.split())

        word_freq = {}
        for word in words:
            word_freq[word] = word_freq.get(word, 0) + 1

        # 取前20个高频词
        common_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)[:20]

        return {
            "sender": target_sender,
            "total_messages": total_messages,
            "avg_message_length": avg_length,
            "common_words": [w[0] for w in common_words],
            "time_range": {
                "start": min(m.timestamp for m in target_messages).isoformat(),
                "end": max(m.timestamp for m in target_messages).isoformat()
            }
        }


def main():
    """示例用法"""
    parser = WhatsAppParser()

    # 解析示例文件
    # messages = parser.parse_file("data/sample/chat.txt")
    # cleaned = parser.clean_messages(messages)

    # print(f"解析到 {len(messages)} 条消息")
    # print(f"清洗后 {len(cleaned)} 条消息")

    # 提取 Persona
    # persona = parser.extract_persona_features(cleaned, "张三")
    # print(f"Persona 特征: {persona}")

    print("WhatsApp Parser 已就绪")


if __name__ == "__main__":
    main()
