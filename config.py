"""
Cấu hình toàn bộ ứng dụng — chỉnh sửa tại đây, không sửa rải rác khắp project.
Các giá trị bí mật (API key, mật khẩu SMTP) đọc từ biến môi trường hoặc file .env.
"""
import os
from pathlib import Path

from dotenv import load_dotenv

# ── Đường dẫn gốc ──────────────────────────────────────────────────────────────
ROOT_DIR   = Path(__file__).parent
DATA_DIR   = ROOT_DIR / "data"
MODELS_DIR = ROOT_DIR / "models"

load_dotenv(ROOT_DIR / ".env")

# ── Dữ liệu ────────────────────────────────────────────────────────────────────
DATASET_PATH   = DATA_DIR / "vi_dataset.csv"       # dataset chính
MODEL_PATH     = MODELS_DIR / "model.pkl"           # file model Naive Bayes
TEST_SIZE      = 0.3
RANDOM_STATE   = 42
MAX_WORDS      = 200                                # cắt câu quá dài
MIN_WORDS      = 3                                  # loại mẫu quá ngắn

# ── LangChain / Gemini ─────────────────────────────────────────────────────────
# Đặt GOOGLE_API_KEY trong file .env hoặc biến môi trường. Không ghi key vào code.
GOOGLE_API_KEY: str = os.getenv("GOOGLE_API_KEY", "")
GEMINI_MODEL    = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
LLM_TEMPERATURE = 0

# ── Retry config (chống lỗi 429 Rate Limit) ────────────────────────────────────
LLM_MAX_RETRIES = 3       # số lần thử lại tối đa
LLM_RETRY_DELAY = 30      # giây chờ giữa mỗi lần thử lại

# ── AI Agent soạn & gửi email ──────────────────────────────────────────────────
AGENT_TEMPERATURE = 0.4   # cao hơn một chút để văn phong tự nhiên
AGENT_MAX_STEPS   = 8     # số vòng gọi tool tối đa cho một yêu cầu
MAX_RECIPIENTS    = 10    # chặn gửi hàng loạt

# ── SMTP (gửi email) ───────────────────────────────────────────────────────────
# Gmail: SMTP_HOST=smtp.gmail.com, SMTP_PORT=587, SMTP_PASSWORD = App Password
# (tạo tại https://myaccount.google.com/apppasswords, cần bật xác minh 2 bước).
SMTP_HOST        = os.getenv("SMTP_HOST", "smtp.gmail.com")
SMTP_PORT        = int(os.getenv("SMTP_PORT", "587"))
SMTP_USER        = os.getenv("SMTP_USER", "")
SMTP_PASSWORD    = os.getenv("SMTP_PASSWORD", "")
SMTP_FROM        = os.getenv("SMTP_FROM", "") or SMTP_USER
SMTP_SENDER_NAME = os.getenv("SMTP_SENDER_NAME", "")

# ── Streamlit UI ───────────────────────────────────────────────────────────────
APP_TITLE   = "🛡️ Vietnamese Spam Classifier"
APP_ICON    = "🛡️"
SUPPORTED_UPLOAD_TYPES = ["txt", "pdf", "docx"]
