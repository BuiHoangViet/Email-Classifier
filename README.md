# 🛡️ Vietnamese Spam Classifier

Ứng dụng phân loại thư rác / tin nhắn tiếng Việt với **hai engine**:
- **Naive Bayes** — chạy hoàn toàn offline, không cần API key
- **Gemini LLM (LangChain)** — dùng Google Gemini, hiểu ngữ cảnh sâu hơn

---

## 📁 Cấu trúc thư mục

```
spam_classifier/
│
├── app.py                  ← Entry point (Streamlit UI)
├── train.py                ← Script huấn luyện Naive Bayes
├── config.py               ← Toàn bộ cấu hình tập trung (đọc .env)
├── requirements.txt
├── .env.example            ← Mẫu biến môi trường
│
├── core/
│   ├── __init__.py
│   ├── naive_bayes.py      ← Thuật toán Naive Bayes (train + predict)
│   └── llm_classifier.py  ← LangChain pipeline (Gemini)
│   └── email_agent.py      ← AI Agent soạn & gửi email
│
├── utils/
│   ├── __init__.py
│   ├── data_loader.py      ← Đọc CSV dataset
│   └── file_reader.py      ← Đọc .txt / .pdf / .docx (LangChain Loaders)
│   └── mailer.py           ← Gửi email qua SMTP
│
├── data/
│   ├── vi_dataset.csv      ← Dataset chính
│   └── sms_spam_vi.csv     ← Dataset bổ sung
│
└── models/
    └── model.pkl           ← Model đã train (tự sinh sau bước 3)
```

---

## 🚀 Hướng dẫn chạy từ đầu

### Bước 1 — Yêu cầu hệ thống

| Công cụ | Phiên bản tối thiểu |
|---------|---------------------|
| Python  | 3.11+               |
| pip     | 23+                 |

> Kiểm tra: `python --version` và `pip --version`

---

### Bước 2 — Tải project về máy

```bash
# Nếu dùng Git
git clone <repo_url>
cd spam_classifier

# Hoặc giải nén file ZIP rồi vào thư mục
cd spam_classifier
```

---

### Bước 3 — Tạo môi trường ảo & cài thư viện

```bash
# Tạo môi trường ảo (khuyến nghị)
python -m venv .venv

# Kích hoạt
# Windows:
.venv\Scripts\activate
# macOS / Linux:
source .venv/bin/activate

# Cài thư viện
pip install -r requirements.txt
```

---

### Bước 4 — Cấu hình API Key *(chỉ cần nếu dùng Gemini LLM)*

**Cách A — Biến môi trường (khuyến nghị)**

```bash
# macOS / Linux
export GOOGLE_API_KEY="AIza..."

# Windows (PowerShell)
$env:GOOGLE_API_KEY="AIza..."
```

**Cách B — File .env**

```bash
cp .env.example .env
# Mở .env và điền key thật vào
```

**Cách C — Nhập trực tiếp trên giao diện**
- Mở sidebar → ô "Google API Key" → dán key vào.

> 🔑 Lấy key miễn phí tại: https://aistudio.google.com/app/apikey

---

### Bước 5 — Huấn luyện Naive Bayes model

```bash
python train.py
```

Kết quả mong đợi:
```
[INFO] Đọc dữ liệu : data/vi_dataset.csv
[INFO] Tổng mẫu   : 4733  (sau lọc từ 5120)
[INFO] Train / Test: 3313 / 1420
[INFO] Accuracy    : 94.44%
[INFO] ✔ Model lưu : models/model.pkl
```

> Muốn dùng dataset khác:
> ```bash
> python train.py --dataset data/sms_spam_vi.csv
> ```

---

### Bước 6 — Chạy ứng dụng

```bash
streamlit run app.py
```

Trình duyệt tự mở tại **http://localhost:8501**

---

## 🖥️ Hướng dẫn sử dụng giao diện

1. **Sidebar** → chọn engine: *Naive Bayes* hoặc *Gemini LLM*
2. **Tab "Nhập tay"** → dán nội dung tin nhắn / email
3. **Tab "Tải file"** → upload file `.txt`, `.pdf`, hoặc `.docx`
4. Bấm **🔍 Phân loại**
5. Xem kết quả: 🚨 SPAM hoặc ✅ HAM

---

## ⚙️ Tuỳ chỉnh nhanh

Mở `config.py` để thay đổi:

| Biến | Mô tả |
|------|-------|
| `DATASET_PATH` | Đường dẫn file CSV train |
| `MODEL_PATH` | Nơi lưu model.pkl |
| `GEMINI_MODEL` | Tên model Gemini (mặc định gemini-2.5-flash, đổi qua .env) |
| `MAX_WORDS` | Giới hạn số từ mỗi mẫu train |
| `TEST_SIZE` | Tỷ lệ test khi train (mặc định 0.3) |

---

## 🐛 Lỗi thường gặp

| Lỗi | Nguyên nhân | Cách sửa |
|-----|-------------|----------|
| `model.pkl not found` | Chưa train | Chạy `python train.py` |
| `ModuleNotFoundError` | Chưa cài thư viện | Chạy `pip install -r requirements.txt` |
| `API key invalid` | Key sai hoặc hết quota | Kiểm tra lại key tại aistudio.google.com |
| `Streamlit not found` | Chưa kích hoạt venv | Chạy `source .venv/bin/activate` |

---

## 🤖 Trợ lý Email AI (AI Agent)

Sidebar → **Chức năng** → *🤖 Trợ lý Email AI*. Chat bằng tiếng Việt, ví dụ:

- `Soạn mail xin nghỉ phép ngày mai gửi tới sep@congty.com` → AI soạn, kiểm tra spam bằng Naive Bayes, rồi **tự gửi**
- `Gợi ý tin nhắn xin lỗi khách vì giao hàng trễ` → AI chỉ soạn bản nháp, không gửi
- `Sửa lại ngắn hơn rồi gửi cho cả b@congty.com` → AI dùng ngữ cảnh lượt trước

Bật **"Xác nhận trước khi gửi"** ở sidebar nếu muốn xem/sửa bản nháp và tự bấm **📤 Gửi**.

### Cấu hình gửi mail (file `.env`)

```bash
SMTP_HOST=smtp.gmail.com
SMTP_PORT=587            # 465 = SSL
SMTP_USER=ban@gmail.com
SMTP_PASSWORD=xxxx xxxx xxxx xxxx   # Gmail App Password, không phải mật khẩu đăng nhập
SMTP_SENDER_NAME=Tên của bạn        # dùng để ký cuối thư
```

Gmail App Password: bật xác minh 2 bước → https://myaccount.google.com/apppasswords

Giới hạn an toàn: tối đa 10 người nhận/lần, AI không tự bịa địa chỉ email, không gửi trùng trong một yêu cầu.
