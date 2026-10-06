"""
utils/mailer.py
Gửi email qua SMTP (Gmail, Outlook, ...). Cấu hình lấy từ config / .env.
"""
from __future__ import annotations

import re
import smtplib
import ssl
from dataclasses import dataclass
from email.message import EmailMessage
from email.utils import formataddr

from config import (
    SMTP_HOST, SMTP_PORT, SMTP_USER, SMTP_PASSWORD,
    SMTP_FROM, SMTP_SENDER_NAME, MAX_RECIPIENTS,
)

_EMAIL_RE = re.compile(r"^[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}$")


@dataclass(frozen=True)
class SmtpSettings:
    host: str
    port: int
    user: str
    password: str
    sender: str
    sender_name: str = ""

    @classmethod
    def from_config(cls) -> "SmtpSettings":
        return cls(SMTP_HOST, SMTP_PORT, SMTP_USER, SMTP_PASSWORD,
                   SMTP_FROM, SMTP_SENDER_NAME)

    @property
    def is_configured(self) -> bool:
        return bool(self.host and self.user and self.password and self.sender)


def parse_recipients(raw: str | list[str]) -> list[str]:
    """Tách danh sách người nhận (phẩy / chấm phẩy / khoảng trắng) và kiểm tra hợp lệ."""
    items = raw if isinstance(raw, list) else re.split(r"[,;\s]+", raw or "")
    recipients = list(dict.fromkeys(a.strip() for a in items if a and a.strip()))

    if not recipients:
        raise ValueError("Chưa có địa chỉ email người nhận.")
    invalid = [a for a in recipients if not _EMAIL_RE.match(a)]
    if invalid:
        raise ValueError(f"Địa chỉ email không hợp lệ: {', '.join(invalid)}")
    if len(recipients) > MAX_RECIPIENTS:
        raise ValueError(f"Tối đa {MAX_RECIPIENTS} người nhận mỗi lần gửi.")
    return recipients


def send_email(
    to: str | list[str],
    subject: str,
    body: str,
    settings: SmtpSettings | None = None,
) -> list[str]:
    """Gửi email dạng text. Trả về danh sách người nhận đã gửi. Raise RuntimeError nếu lỗi."""
    settings = settings or SmtpSettings.from_config()
    if not settings.is_configured:
        raise RuntimeError(
            "Chưa cấu hình SMTP. Điền SMTP_USER, SMTP_PASSWORD trong file .env."
        )
    recipients = parse_recipients(to)

    msg = EmailMessage()
    msg["From"]    = formataddr((settings.sender_name, settings.sender))
    msg["To"]      = ", ".join(recipients)
    msg["Subject"] = subject.strip() or "(Không có tiêu đề)"
    msg.set_content(body)

    context = ssl.create_default_context()
    try:
        if settings.port == 465:
            with smtplib.SMTP_SSL(settings.host, settings.port, context=context, timeout=30) as smtp:
                smtp.login(settings.user, settings.password)
                smtp.send_message(msg)
        else:
            with smtplib.SMTP(settings.host, settings.port, timeout=30) as smtp:
                smtp.starttls(context=context)
                smtp.login(settings.user, settings.password)
                smtp.send_message(msg)
    except smtplib.SMTPAuthenticationError as exc:
        raise RuntimeError(
            "Đăng nhập SMTP thất bại. Kiểm tra SMTP_USER / SMTP_PASSWORD "
            "(Gmail cần dùng App Password)."
        ) from exc
    except (smtplib.SMTPException, OSError) as exc:
        raise RuntimeError(f"Gửi email thất bại: {exc}") from exc

    return recipients
