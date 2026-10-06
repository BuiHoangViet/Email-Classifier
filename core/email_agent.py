"""
core/email_agent.py
AI Agent soạn & gửi email: người dùng chat yêu cầu (vd. "soạn mail xin nghỉ phép
gửi tới sep@congty.com"), Gemini tự soạn nội dung, kiểm tra spam bằng Naive Bayes
rồi gọi tool gửi email.

Agent dùng vòng lặp tool-calling thủ công (bind_tools) để không phụ thuộc vào
API agent hay thay đổi giữa các phiên bản LangChain.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Callable

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import (
    AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage,
)
from langchain_core.tools import tool

from config import AGENT_MAX_STEPS
from core.llm_classifier import call_with_retry

# (to, subject, body) -> thông báo kết quả cho LLM
SendFn = Callable[[str, str, str], str]
# text -> (label, spam_probability) hoặc None nếu chưa có model
SpamFn = Callable[[str], tuple[str, float] | None]

_SYSTEM_PROMPT = """Bạn là trợ lý soạn và gửi email tiếng Việt chuyên nghiệp.

Khi người dùng yêu cầu GỬI email:
1. Xác định địa chỉ email người nhận. Nếu người dùng chưa cung cấp địa chỉ hợp lệ, hãy hỏi lại — TUYỆT ĐỐI không tự bịa địa chỉ.
2. Soạn tiêu đề ngắn gọn và nội dung rõ ràng, lịch sự, đúng mục đích, có lời chào và lời kết.{signature}
3. Gọi tool `check_spam` với tiêu đề và nội dung. Nếu kết quả là spam, viết lại cho tự nhiên hơn (bớt từ ngữ quảng cáo, khuyến mãi, ký tự đặc biệt, chữ in hoa) rồi kiểm tra lại, tối đa 2 lần.
4. Gọi tool `send_email` đúng một lần cho mỗi email.
5. Trả lời người dùng ngắn gọn: đã gửi cho ai, tiêu đề gì. Nếu tool báo lỗi, giải thích lỗi và cách khắc phục.

Khi người dùng chỉ nhờ GỢI Ý / SOẠN / VIẾT tin nhắn hoặc email mà không yêu cầu gửi: chỉ trả về bản nháp, không gọi `send_email`.
Khi người dùng muốn sửa bản nháp trước đó rồi gửi: dùng bản đã sửa.
Luôn trả lời bằng tiếng Việt."""


@dataclass
class AgentResult:
    reply: str
    tool_calls: list[dict] = field(default_factory=list)


def _text_of(message: AIMessage) -> str:
    """Lấy phần text từ AIMessage (content có thể là str hoặc list các part)."""
    content = message.content
    if isinstance(content, str):
        return content
    parts = []
    for part in content:
        if isinstance(part, str):
            parts.append(part)
        elif isinstance(part, dict) and part.get("type") == "text":
            parts.append(part.get("text", ""))
    return "".join(parts)


class EmailAgent:
    def __init__(
        self,
        llm: BaseChatModel,
        send_fn: SendFn,
        spam_fn: SpamFn,
        sender_name: str = "",
        max_steps: int = AGENT_MAX_STEPS,
    ) -> None:
        self._tools = self._make_tools(send_fn, spam_fn)
        self._llm = llm.bind_tools(list(self._tools.values()))
        signature = f' Ký tên cuối thư là "{sender_name}".' if sender_name else ""
        self._system = _SYSTEM_PROMPT.format(signature=signature)
        self._max_steps = max_steps

    @staticmethod
    def _make_tools(send_fn: SendFn, spam_fn: SpamFn) -> dict:
        sent: set[tuple[str, str]] = set()

        @tool
        def check_spam(subject: str, body: str) -> str:
            """Kiểm tra email có bị bộ lọc Naive Bayes đánh giá là spam hay không."""
            result = spam_fn(f"{subject}\n{body}")
            if result is None:
                return "Chưa có model Naive Bayes, bỏ qua bước kiểm tra spam."
            label, prob = result
            return json.dumps(
                {"label": label, "spam_probability": round(prob, 3)},
                ensure_ascii=False,
            )

        @tool
        def send_email(to: str, subject: str, body: str) -> str:
            """Gửi email. `to`: một hoặc nhiều địa chỉ email, cách nhau bởi dấu phẩy."""
            key = (to.strip().lower(), subject.strip())
            if key in sent:
                return "Email này đã được xử lý trong yêu cầu hiện tại, không gửi lại."
            sent.add(key)
            return send_fn(to, subject, body)

        return {t.name: t for t in (check_spam, send_email)}

    def run(self, history: list[BaseMessage]) -> AgentResult:
        """Chạy agent trên lịch sử hội thoại (tin nhắn cuối là của người dùng)."""
        messages: list[BaseMessage] = [SystemMessage(self._system), *history]
        calls: list[dict] = []

        for _ in range(self._max_steps):
            ai: AIMessage = call_with_retry(lambda: self._llm.invoke(messages))
            messages.append(ai)

            if not ai.tool_calls:
                return AgentResult(_text_of(ai).strip(), calls)

            for call in ai.tool_calls:
                tool_fn = self._tools.get(call["name"])
                if tool_fn is None:
                    output = f"Không có tool tên {call['name']}."
                else:
                    try:
                        output = str(tool_fn.invoke(call["args"]))
                    except Exception as exc:  # lỗi tool trả lại cho LLM tự xử lý
                        output = f"Lỗi: {exc}"
                calls.append({"name": call["name"], "args": call["args"], "output": output})
                messages.append(ToolMessage(content=output, tool_call_id=call["id"]))

        return AgentResult(
            "Xin lỗi, yêu cầu này cần quá nhiều bước xử lý. Bạn thử diễn đạt lại ngắn gọn hơn nhé.",
            calls,
        )


def to_langchain_history(chat: list[dict]) -> list[BaseMessage]:
    """Chuyển lịch sử chat lưu trong session (role/content) sang message LangChain."""
    out: list[BaseMessage] = []
    for msg in chat:
        if msg["role"] == "user":
            out.append(HumanMessage(msg["content"]))
        else:
            content = msg["content"]
            # Cho LLM biết các email đã soạn/gửi ở lượt trước để xử lý câu hỏi tiếp theo
            for e in msg.get("emails", []):
                content += (
                    f"\n[Email {e['status']}: tới {e['to']} | tiêu đề: {e['subject']}"
                    f" | nội dung: {e['body']}]"
                )
            out.append(AIMessage(content))
    return out
