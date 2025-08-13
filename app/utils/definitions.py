import os
import re
import logging
from typing import Any


logger = logging.getLogger(__name__)


def is_definitional_question(question: str) -> bool:
    """判断是否为“定义/科普”类问题，而非求助/建议类。

    逻辑与旧版服务保持一致，优先匹配定义类模式，并排除求助类表述。
    """
    if not question or not question.strip():
        return False

    question_lower = question.lower().strip()

    definitional_patterns = [
        r"what is\s+\w+",
        r"what are\s+\w+",
        r"define\s+\w+",
        r"meaning of\s+\w+",
        r"definition of\s+\w+",
        r"what does\s+\w+\s+mean",
        r"explain\s+\w+",
        r"tell me about\s+\w+",
        r"describe\s+\w+",
        r"what is the\s+\w+",
        r"what are the\s+\w+",
        r"how is\s+\w+\s+defined",
        r"what constitutes\s+\w+",
    ]
    for pattern in definitional_patterns:
        if re.search(pattern, question_lower):
            return True

    medical_terms = [
        "depression",
        "anxiety",
        "ptsd",
        "ocd",
        "bipolar",
        "schizophrenia",
        "panic disorder",
        "social anxiety",
        "generalized anxiety",
        "major depressive disorder",
        "dysthymia",
        "cyclothymia",
        "borderline personality",
        "narcissistic personality",
        "antisocial personality",
        "avoidant personality",
    ]

    help_seeking_patterns = [
        r"what should i do\s+\w+",
        r"what can i do\s+\w+",
        r"how can i\s+\w+",
        r"how do i\s+\w+",
        r"i need help\s+\w+",
        r"can you help\s+\w+",
        r"help me\s+\w+",
        r"struggling with\s+\w+",
        r"dealing with\s+\w+",
        r"coping with\s+\w+",
    ]
    for pattern in help_seeking_patterns:
        if re.search(pattern, question_lower):
            return False

    has_medical_term = any(term in question_lower for term in medical_terms)
    is_information_request = any(
        word in question_lower for word in ["what", "define", "explain", "tell", "describe"]
    )
    return has_medical_term and is_information_request


def load_definitional_prompt() -> str:
    """加载定义类问题的提示词模板。路径沿用旧版以保持兼容。"""
    definitional_prompt_path = "OPRO_Streamlined/prompts/definitional_prompt.txt"
    try:
        if os.path.exists(definitional_prompt_path):
            with open(definitional_prompt_path, "r", encoding="utf-8") as f:
                return f.read().strip()
        else:
            logger.warning(
                "Definitional prompt not found at %s", definitional_prompt_path
            )
            return ""
    except Exception as e:
        logger.error("Error loading definitional prompt: %s", e)
        return ""


def get_medical_context_for_definition(question: str, store: Any) -> str:
    """为定义类问题检索医学上下文（通过向量库相似度搜索）。

    期望 `store` 兼容 LangChain FAISS 接口，具备 `similarity_search(term, k)` 方法。
    """
    if not store:
        return ""

    try:
        question_lower = question.lower()
        medical_terms = {
            "depression": ["depression", "depressive", "major depressive", "dysthymia"],
            "anxiety": ["anxiety", "anxious", "panic", "generalized anxiety"],
            "ptsd": ["ptsd", "post traumatic", "trauma", "traumatic"],
            "ocd": ["ocd", "obsessive", "compulsive"],
            "bipolar": ["bipolar", "manic", "mania", "cyclothymia"],
            "schizophrenia": ["schizophrenia", "psychotic", "psychosis"],
            "personality": [
                "personality disorder",
                "borderline",
                "narcissistic",
                "antisocial",
            ],
        }

        search_terms: list[str] = []
        for _category, terms in medical_terms.items():
            if any(term in question_lower for term in terms):
                search_terms.extend(terms)
                break

        if not search_terms:
            search_terms = [question]

        context_parts: list[str] = []
        for term in search_terms[:3]:  # 限制搜索次数
            try:
                results = store.similarity_search(term, k=2)
                for result in results:
                    content = getattr(result, "page_content", "") or ""
                    if content and len(content) > 50:
                        context_parts.append(content[:300])
            except Exception as e:
                logger.debug("Error searching for term '%s': %s", term, e)
                continue

        if context_parts:
            combined = " ".join(context_parts)
            combined = re.sub(r"\s+", " ", combined).strip()
            return combined[:800]

        return ""
    except Exception as e:
        logger.error("Error getting medical context: %s", e)
        return ""


