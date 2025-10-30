"""
安全过滤模块
检测并过滤敏感内容,防止有害输出
"""
import re
from typing import Dict, List, Optional
from dataclasses import dataclass


@dataclass
class FilterResult:
    """过滤结果"""
    blocked: bool
    reason: Optional[str] = None
    category: Optional[str] = None
    confidence: float = 0.0


class SafetyFilter:
    """安全过滤器"""

    # 敏感话题关键词
    SENSITIVE_KEYWORDS = {
        "suicide": [
            "自杀", "轻生", "不想活", "去死", "结束生命",
            "suicide", "kill myself", "end my life"
        ],
        "self_harm": [
            "自残", "割腕", "伤害自己", "self harm", "hurt myself"
        ],
        "medical": [
            "诊断", "处方", "用药建议", "治疗方案", "医学建议",
            "diagnosis", "prescription", "medical advice"
        ],
        "legal": [
            "法律建议", "起诉", "合同建议", "legal advice", "sue", "lawsuit"
        ],
        "financial": [
            "投资建议", "理财建议", "股票推荐", "financial advice", "investment"
        ],
        "violence": [
            "暴力", "伤害他人", "攻击", "violence", "harm others", "attack"
        ],
        "privacy": [
            "密码", "银行账号", "身份证号", "password", "account number", "ssn"
        ]
    }

    # 禁止词 (绝对不能出现)
    FORBIDDEN_WORDS = [
        "我就是{name}本人",
        "我是真的{name}",
        "I am the real {name}",
    ]

    # 专业建议模板
    PROFESSIONAL_ADVICE_PATTERNS = [
        r"你应该服用",
        r"我建议你.*药",
        r"诊断.*为",
        r"法律上.*你可以",
        r"投资.*股票",
        r"you should take.*medication",
        r"I diagnose",
        r"legally.*you can"
    ]

    def __init__(self, persona_name: str, enable_emergency_alert: bool = True):
        self.persona_name = persona_name
        self.enable_emergency_alert = enable_emergency_alert
        self.compiled_patterns = [re.compile(p, re.IGNORECASE) for p in self.PROFESSIONAL_ADVICE_PATTERNS]

    def filter(self, response: str, user_query: str) -> Dict:
        """
        过滤响应内容

        Args:
            response: AI 生成的响应
            user_query: 用户查询

        Returns:
            过滤结果字典
        """
        # 1. 检查用户查询中的敏感话题
        query_check = self._check_sensitive_topics(user_query)
        if query_check.blocked and self.enable_emergency_alert:
            return {
                "blocked": True,
                "reason": f"检测到敏感话题: {query_check.category}",
                "category": query_check.category,
                "emergency_alert": True,
                "suggested_action": self._get_emergency_action(query_check.category)
            }

        # 2. 检查响应中的禁止词
        forbidden_check = self._check_forbidden_words(response)
        if forbidden_check.blocked:
            return {
                "blocked": True,
                "reason": forbidden_check.reason,
                "category": "forbidden_content"
            }

        # 3. 检查响应中的专业建议
        advice_check = self._check_professional_advice(response)
        if advice_check.blocked:
            return {
                "blocked": True,
                "reason": "检测到未授权的专业建议",
                "category": "professional_advice",
                "details": advice_check.reason
            }

        # 4. 检查响应中的隐私信息
        privacy_check = self._check_privacy_info(response)
        if privacy_check.blocked:
            return {
                "blocked": True,
                "reason": "检测到隐私信息泄露",
                "category": "privacy_violation"
            }

        # 通过所有检查
        return {
            "blocked": False,
            "reason": None
        }

    def _check_sensitive_topics(self, text: str) -> FilterResult:
        """检查敏感话题"""
        text_lower = text.lower()

        for category, keywords in self.SENSITIVE_KEYWORDS.items():
            for keyword in keywords:
                if keyword.lower() in text_lower:
                    return FilterResult(
                        blocked=True,
                        reason=f"检测到敏感话题关键词: {keyword}",
                        category=category,
                        confidence=1.0
                    )

        return FilterResult(blocked=False)

    def _check_forbidden_words(self, text: str) -> FilterResult:
        """检查禁止词"""
        for template in self.FORBIDDEN_WORDS:
            forbidden_word = template.format(name=self.persona_name)
            if forbidden_word in text:
                return FilterResult(
                    blocked=True,
                    reason=f"包含禁止的声明: {forbidden_word}",
                    category="identity_confusion"
                )

        # 检查是否缺少 AI 身份标识
        if "[基于" not in text and "AI 模拟" not in text:
            # 如果响应较长但没有标识,可能有问题
            if len(text) > 50:
                return FilterResult(
                    blocked=True,
                    reason="响应缺少 AI 身份标识",
                    category="missing_ai_label"
                )

        return FilterResult(blocked=False)

    def _check_professional_advice(self, text: str) -> FilterResult:
        """检查专业建议"""
        for pattern in self.compiled_patterns:
            if pattern.search(text):
                return FilterResult(
                    blocked=True,
                    reason=f"检测到专业建议模式: {pattern.pattern}",
                    category="professional_advice"
                )

        return FilterResult(blocked=False)

    def _check_privacy_info(self, text: str) -> FilterResult:
        """检查隐私信息"""
        # 检查常见隐私模式
        privacy_patterns = [
            (r'\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b', "信用卡号"),
            (r'\b\d{3}-\d{2}-\d{4}\b', "社会安全号"),
            (r'\b\d{15,18}\b', "身份证号"),
            (r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}', "邮箱地址"),
        ]

        for pattern, desc in privacy_patterns:
            if re.search(pattern, text):
                return FilterResult(
                    blocked=True,
                    reason=f"检测到可能的隐私信息: {desc}",
                    category="privacy"
                )

        return FilterResult(blocked=False)

    def _get_emergency_action(self, category: str) -> str:
        """获取紧急情况的建议行动"""
        actions = {
            "suicide": "建议立即联系心理危机热线: 400-161-9995 (中国) 或 988 (美国)",
            "self_harm": "建议联系专业心理咨询师或前往医院",
            "medical": "请咨询专业医生,不要依赖 AI 建议",
            "legal": "请咨询专业律师,不要依赖 AI 建议",
            "financial": "请咨询专业理财顾问,不要依赖 AI 建议"
        }
        return actions.get(category, "请咨询相关专业人士")

    def add_ai_label(self, response: str) -> str:
        """
        确保响应包含 AI 标识

        Args:
            response: 原始响应

        Returns:
            添加标识后的响应
        """
        if "[基于" in response or "AI 模拟" in response:
            return response

        label = f"[基于 {self.persona_name} 历史的 AI 模拟]\n\n"
        return label + response


class ContentModerator:
    """内容审核器 (可选,用于更高级的过滤)"""

    def __init__(self):
        # 可以集成第三方内容审核 API
        pass

    def moderate(self, text: str) -> Dict:
        """
        审核文本内容

        Returns:
            审核结果
        """
        # 这里可以调用 OpenAI Moderation API 或其他服务
        # 示例实现:
        return {
            "flagged": False,
            "categories": {},
            "category_scores": {}
        }


def create_safety_filter(
    persona_name: str,
    enable_emergency_alert: bool = True,
    use_moderation: bool = False
) -> SafetyFilter:
    """
    工厂函数创建安全过滤器

    Args:
        persona_name: Persona 名称
        enable_emergency_alert: 是否启用紧急情况警报
        use_moderation: 是否使用内容审核 API

    Returns:
        SafetyFilter 实例
    """
    return SafetyFilter(persona_name, enable_emergency_alert)


if __name__ == "__main__":
    # 测试示例
    filter = SafetyFilter("张三")

    # 测试敏感话题
    test_cases = [
        ("我想自杀", "应该被拦截"),
        ("今天天气不错", "应该通过"),
        ("你建议我服用什么药?", "应该被拦截"),
        ("[基于张三历史的 AI 模拟] 我觉得...", "应该通过"),
    ]

    for query, expected in test_cases:
        result = filter.filter(query, query)
        print(f"查询: {query}")
        print(f"预期: {expected}")
        print(f"结果: {'拦截' if result['blocked'] else '通过'}")
        if result.get('reason'):
            print(f"原因: {result['reason']}")
        print("-" * 50)
