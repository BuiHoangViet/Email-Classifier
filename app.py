"""
app.py  –  Entry point của ứng dụng Streamlit.
Chạy: streamlit run app.py
"""
from __future__ import annotations

import pickle
import sys
from pathlib import Path
from uuid import uuid4

import streamlit as st

# ── sys.path ────────────────────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent))

from config import (
    APP_TITLE, APP_ICON, MODEL_PATH, DATASET_PATH,
    SUPPORTED_UPLOAD_TYPES, GOOGLE_API_KEY, AGENT_TEMPERATURE,
)
from core.naive_bayes import predict as nb_predict, spam_probability
from core.llm_classifier import build_chain, build_llm, classify_with_llm
from core.email_agent import EmailAgent, to_langchain_history
from utils.file_reader import read_uploaded_file
from utils.mailer import SmtpSettings, parse_recipients, send_email

# ═══════════════════════════════════════════════════════════════════════════════
# Page config
# ═══════════════════════════════════════════════════════════════════════════════
st.set_page_config(page_title=APP_TITLE, page_icon=APP_ICON, layout="centered")

# ── Custom CSS ──────────────────────────────────────────────────────────────────
st.markdown(
    """
    <style>
    .stTextArea textarea { font-size: 0.95rem; }
    .result-box { border-radius: 10px; padding: 1rem 1.4rem; margin-top: 1rem; font-size: 1.1rem; }
    .spam-box   { background: #fff0f0; border-left: 5px solid #e53935; color: #b71c1c; }
    .ham-box    { background: #f0fff4; border-left: 5px solid #2e7d32; color: #1b5e20; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ═══════════════════════════════════════════════════════════════════════════════
# Cached resources
# ═══════════════════════════════════════════════════════════════════════════════
@st.cache_resource
def load_nb_model():
    if not MODEL_PATH.exists():
        return None
    with open(MODEL_PATH, "rb") as f:
        return pickle.load(f)


@st.cache_resource
def load_llm_chain(api_key: str):
    return build_chain(api_key)


@st.cache_resource
def load_agent_llm(api_key: str):
    return build_llm(api_key, temperature=AGENT_TEMPERATURE)


def nb_spam_check(text: str) -> tuple[str, float] | None:
    model_data = load_nb_model()
    if model_data is None:
        return None
    return nb_predict(text, **model_data), spam_probability(text, **model_data)


def train_now_button() -> None:
    if st.button("🏋️ Huấn luyện ngay"):
        from train import run as train_model
        with st.spinner("Đang huấn luyện Naive Bayes..."):
            train_model(DATASET_PATH, MODEL_PATH)
        load_nb_model.clear()
        st.rerun()


# ═══════════════════════════════════════════════════════════════════════════════
# Sidebar
# ═══════════════════════════════════════════════════════════════════════════════
st.title(APP_TITLE)

with st.sidebar:
    st.header("⚙️ Cấu hình")

    page = st.radio("Chức năng", ["🔍 Phân loại spam", "🤖 Trợ lý Email AI"], index=0)
    agent_page = page.startswith("🤖")

    use_llm = False
    if not agent_page:
        engine = st.radio(
            "Chọn engine phân loại",
            ["🤖 Naive Bayes (local)", "✨ Gemini LLM (LangChain)"],
            index=0,
        )
        use_llm = engine.startswith("✨")

    api_key_input = ""
    if use_llm or agent_page:
        st.divider()
        api_key_input = st.text_input(
            "Google API Key",
            value=GOOGLE_API_KEY,
            type="password",
            placeholder="AIza...",
            help="Lấy key miễn phí tại https://aistudio.google.com/app/apikey",
        ).strip()

    auto_send = True
    if agent_page:
        auto_send = not st.toggle(
            "Xác nhận trước khi gửi",
            value=False,
            help="Bật: AI chỉ soạn bản nháp, bạn bấm Gửi mới gửi. Tắt: AI tự gửi luôn.",
        )
        if st.button("🗑️ Xoá hội thoại", use_container_width=True):
            st.session_state.chat = []
            st.rerun()

    st.divider()
    if agent_page:
        st.info(
            "Nhắn yêu cầu như: *“Soạn mail xin nghỉ phép ngày mai gửi tới "
            "sep@congty.com”*. AI sẽ soạn, kiểm tra spam rồi gửi.",
            icon="ℹ️",
        )
    else:
        st.info(
            "**Naive Bayes**: nhanh, chạy offline, cần train trước.\n\n"
            "**Gemini LLM**: hiểu ngữ cảnh sâu hơn, cần API key.",
            icon="ℹ️",
        )


# ═══════════════════════════════════════════════════════════════════════════════
# Trang 1: Phân loại spam
# ═══════════════════════════════════════════════════════════════════════════════
def render_classifier_page() -> None:
    st.caption("Phân loại tin nhắn / email tiếng Việt: **Spam** hay **Ham**")
    st.subheader("📥 Nhập nội dung")

    tab_text, tab_file = st.tabs(["✏️ Nhập tay", "📎 Tải file"])

    with tab_text:
        input_text = st.text_area(
            "Nội dung cần phân loại:",
            height=220,
            placeholder="Dán tin nhắn hoặc nội dung email vào đây...",
        )

    with tab_file:
        uploaded = st.file_uploader(
            "Chọn file", type=SUPPORTED_UPLOAD_TYPES,
            help=f"Hỗ trợ: {', '.join(SUPPORTED_UPLOAD_TYPES)}",
        )
        if uploaded:
            file_text = ""
            with st.spinner("Đang đọc file..."):
                try:
                    file_text = read_uploaded_file(uploaded)
                    st.text_area("Nội dung file:", file_text, height=220)
                except ValueError as e:
                    st.error(str(e))
            input_text = file_text

    st.divider()
    if not st.button("🔍 Phân loại", type="primary", use_container_width=True):
        return

    text = (input_text or "").strip()
    if not text:
        st.warning("⚠️ Vui lòng nhập nội dung hoặc tải file.")
        return

    prob = None
    if not use_llm:
        result = nb_spam_check(text)
        if result is None:
            st.error("❌ Chưa tìm thấy `models/model.pkl`. Hãy huấn luyện model trước.")
            train_now_button()
            return
        label, prob = result
    else:
        if not api_key_input:
            st.error("❌ Cần nhập Google API Key ở sidebar.")
            return
        with st.spinner("Gemini đang phân tích ngữ cảnh..."):
            try:
                label = classify_with_llm(text, load_llm_chain(api_key_input))
            except RuntimeError as e:
                st.error(f"❌ {e}")
                return

    st.subheader("📌 Kết quả")
    if label == "spam":
        st.markdown(
            '<div class="result-box spam-box">🚨 <strong>SPAM</strong> — Thư / tin nhắn rác</div>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            '<div class="result-box ham-box">✅ <strong>HAM</strong> — Thư / tin nhắn bình thường</div>',
            unsafe_allow_html=True,
        )

    engine_label = "Gemini LLM" if use_llm else "Naive Bayes"
    extra = f" · Xác suất spam: {prob:.1%}" if prob is not None else ""
    st.caption(f"Engine: {engine_label} · {len(text.split())} từ đầu vào{extra}")


# ═══════════════════════════════════════════════════════════════════════════════
# Trang 2: Trợ lý Email AI
# ═══════════════════════════════════════════════════════════════════════════════
STATUS_LABELS = {
    "sent":      "✅ Đã gửi",
    "pending":   "⏳ Chờ xác nhận",
    "failed":    "❌ Gửi thất bại",
    "cancelled": "🚫 Đã huỷ",
}

EXAMPLES = [
    "Soạn mail mời họp dự án lúc 9h sáng thứ Hai gửi tới a@example.com",
    "Gợi ý tin nhắn xin lỗi khách hàng vì giao hàng trễ",
    "Viết mail cảm ơn đối tác sau buổi gặp hôm nay",
]


def _deliver(rec: dict, smtp: SmtpSettings) -> str:
    """Gửi một email record, cập nhật trạng thái, trả về thông báo cho LLM."""
    try:
        send_email(rec["to"], rec["subject"], rec["body"], smtp)
    except (RuntimeError, ValueError) as e:
        rec["status"], rec["error"] = "failed", str(e)
        return f"Lỗi: {e}"
    rec["status"], rec["error"] = "sent", ""
    return f"Đã gửi thành công tới {rec['to']}."


def make_send_fn(records: list[dict], smtp: SmtpSettings):
    def send_fn(to: str, subject: str, body: str) -> str:
        rec = {
            "id": uuid4().hex, "to": to, "subject": subject, "body": body,
            "status": "pending", "error": "",
            "spam": nb_spam_check(f"{subject}\n{body}"),
        }
        records.append(rec)
        try:
            rec["to"] = ", ".join(parse_recipients(to))
        except ValueError as e:
            rec["status"], rec["error"] = "failed", str(e)
            return f"Lỗi: {e}"

        if not auto_send:
            return ("Đã soạn xong. Email đang chờ người dùng kiểm tra và bấm nút "
                    "'Gửi' trên giao diện — hãy nhắc người dùng làm việc này.")
        return _deliver(rec, smtp)

    return send_fn


def _confirm_send(rec: dict, smtp: SmtpSettings) -> None:
    rec["subject"] = st.session_state[f"subj_{rec['id']}"]
    rec["body"]    = st.session_state[f"body_{rec['id']}"]
    _deliver(rec, smtp)


def _cancel(rec: dict) -> None:
    rec["status"] = "cancelled"


def render_email_card(rec: dict, smtp: SmtpSettings) -> None:
    with st.container(border=True):
        st.markdown(f"**{STATUS_LABELS[rec['status']]}** · Tới: `{rec['to']}`")

        if rec["status"] == "pending":
            st.text_input("Tiêu đề", rec["subject"], key=f"subj_{rec['id']}")
            st.text_area("Nội dung", rec["body"], height=220, key=f"body_{rec['id']}")
        else:
            st.markdown(f"**Tiêu đề:** {rec['subject']}")
            st.text(rec["body"])

        if rec.get("spam"):
            label, prob = rec["spam"]
            icon = "⚠️" if label == "spam" else "🛡️"
            st.caption(f"{icon} Kiểm tra Naive Bayes: {label.upper()} · xác suất spam {prob:.0%}")
        if rec["error"]:
            st.error(rec["error"])

        if rec["status"] == "pending":
            c1, c2 = st.columns(2)
            c1.button("📤 Gửi", key=f"send_{rec['id']}", type="primary",
                      use_container_width=True, on_click=_confirm_send, args=(rec, smtp))
            c2.button("Huỷ", key=f"cancel_{rec['id']}",
                      use_container_width=True, on_click=_cancel, args=(rec,))


def render_agent_page() -> None:
    st.caption("Chat với AI để soạn tin nhắn / email — AI có thể tự gửi email giúp bạn.")
    smtp = SmtpSettings.from_config()
    chat: list[dict] = st.session_state.setdefault("chat", [])

    if not api_key_input:
        st.warning("⚠️ Cần Google API Key (sidebar hoặc file `.env`) để dùng trợ lý.")
    if not smtp.is_configured:
        st.warning(
            "📭 Chưa cấu hình SMTP nên chưa gửi được email (vẫn soạn được). "
            "Điền `SMTP_USER`, `SMTP_PASSWORD` trong file `.env` rồi khởi động lại app."
        )
    else:
        st.caption(f"📧 Gửi từ: `{smtp.sender}`")

    if not chat:
        st.markdown("**Thử hỏi:**")
        for ex in EXAMPLES:
            st.markdown(f"- {ex}")

    for msg in chat:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            for rec in msg.get("emails", []):
                render_email_card(rec, smtp)

    prompt = st.chat_input("Nhắn yêu cầu, vd: soạn mail ... gửi tới ...", disabled=not api_key_input)
    if not prompt:
        return

    chat.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    records: list[dict] = []
    with st.chat_message("assistant"), st.spinner("AI đang soạn..."):
        try:
            agent = EmailAgent(
                load_agent_llm(api_key_input),
                send_fn=make_send_fn(records, smtp),
                spam_fn=nb_spam_check,
                sender_name=smtp.sender_name,
            )
            reply = agent.run(to_langchain_history(chat[-20:])).reply or "(Không có phản hồi)"
        except RuntimeError as e:
            reply = f"❌ {e}"

    chat.append({"role": "assistant", "content": reply, "emails": records})
    st.rerun()


if agent_page:
    render_agent_page()
else:
    render_classifier_page()
