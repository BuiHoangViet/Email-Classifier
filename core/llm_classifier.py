"""
core/llm_classifier.py
Phân loại spam bằng LangChain + Google Gemini.
- Chỉ khởi tạo LLM một lần (cached qua st.cache_resource ở app.py)
- Tự động retry khi gặp lỗi 429 RESOURCE_EXHAUSTED (rate limit)
"""
from __future__ import annotations

import re
import time
import logging
from typing import Callable, TypeVar

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import Runnable

from config import GEMINI_MODEL, LLM_TEMPERATURE, LLM_MAX_RETRIES, LLM_RETRY_DELAY

logger = logging.getLogger(__name__)
T = TypeVar("T")

# ── Prompt ──────────────────────────────────────────────────────────────────────
_PROMPT = PromptTemplate.from_template(
    """Bạn là chuyên gia bảo mật email và an ninh mạng người Việt.
Phân loại đoạn văn bản sau thành MỘT trong hai nhãn:

• SPAM – thư rác, lừa đảo tài chính, quảng cáo vay tiền/cờ bạc, link đáng ngờ, tin nhắn khuyến mãi không xác thực.
• HAM  – trao đổi bình thường, thông báo hành chính, học tập, công việc.

---
{text}
---

Chỉ trả về đúng một từ: SPAM hoặc HAM. Không giải thích.
"""
)


def build_llm(api_key: str, temperature: float = LLM_TEMPERATURE) -> ChatGoogleGenerativeAI:
    return ChatGoogleGenerativeAI(
        model=GEMINI_MODEL,
        temperature=temperature,
        google_api_key=api_key,
    )


def build_chain(api_key: str) -> Runnable:
    """Tạo LangChain pipeline: Prompt → LLM → StrOutputParser."""
    return _PROMPT | build_llm(api_key) | StrOutputParser()


def _parse_rate_limit(exc: Exception) -> tuple[bool, int]:
    """
    Kiểm tra lỗi 429 và đọc thời gian retry gợi ý từ message.
    Trả về (is_rate_limit, wait_seconds).
    """
    err_str = str(exc)
    if "429" in err_str or "RESOURCE_EXHAUSTED" in err_str or "quota" in err_str.lower():
        match = re.search(r"retry.*?(\d+)s", err_str, re.IGNORECASE)
        wait = int(match.group(1)) + 2 if match else LLM_RETRY_DELAY
        return True, wait
    return False, 0


def call_with_retry(fn: Callable[[], T]) -> T:
    """
    Gọi fn(), tự động retry tối đa LLM_MAX_RETRIES lần khi gặp lỗi 429.
    Raise RuntimeError với thông báo tiếng Việt rõ ràng nếu thất bại.
    """
    for attempt in range(1, LLM_MAX_RETRIES + 1):
        try:
            return fn()

        except Exception as exc:
            is_rate_limit, wait_sec = _parse_rate_limit(exc)

            if is_rate_limit and attempt < LLM_MAX_RETRIES:
                logger.warning(
                    "Rate limit 429 – lần thử %d/%d, chờ %ds...",
                    attempt, LLM_MAX_RETRIES, wait_sec,
                )
                time.sleep(wait_sec)
                continue

            # Hết lượt retry hoặc lỗi khác
            if is_rate_limit:
                raise RuntimeError(
                    f"⏱️ Vượt quota Gemini free tier (lỗi 429) sau {LLM_MAX_RETRIES} lần thử.\n\n"
                    "**Cách khắc phục:**\n"
                    "- Chờ ~1 phút rồi thử lại\n"
                    "- Hoặc chuyển sang engine **Naive Bayes** (không cần API, chạy offline)\n"
                    "- Hoặc nâng cấp Google AI plan: https://ai.google.dev"
                ) from exc
            raise RuntimeError(f"Lỗi khi gọi Gemini API: {exc}") from exc

    raise AssertionError("unreachable")


def classify_with_llm(text: str, chain: Runnable) -> str:
    """Gọi chain và trả về 'spam' hoặc 'ham' (lowercase)."""
    truncated = text[:3000]  # giới hạn token input
    result = call_with_retry(lambda: chain.invoke({"text": truncated}))
    return "spam" if "SPAM" in result.upper() else "ham"
