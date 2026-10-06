"""
utils/file_reader.py
Đọc nội dung văn bản từ các định dạng file được hỗ trợ.
Dùng LangChain Document Loaders làm backend chính.
"""
from __future__ import annotations

import os
import tempfile
from pathlib import Path

from langchain_community.document_loaders import (
    PyPDFLoader,
    Docx2txtLoader,
    TextLoader,
)


_LOADERS = {
    "pdf":  PyPDFLoader,
    "docx": Docx2txtLoader,
    "txt":  lambda p: TextLoader(p, encoding="utf-8"),
}


def read_uploaded_file(uploaded_file) -> str:
    """
    Nhận một Streamlit UploadedFile, trả về chuỗi văn bản.
    Raise ValueError nếu định dạng không hỗ trợ.
    """
    ext = Path(uploaded_file.name).suffix.lstrip(".").lower()
    if ext not in _LOADERS:
        raise ValueError(f"Định dạng '.{ext}' chưa được hỗ trợ.")

    # LangChain loaders cần đường dẫn thật trên ổ đĩa
    suffix = f".{ext}"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(uploaded_file.getvalue())
        tmp_path = tmp.name

    try:
        loader = _LOADERS[ext](tmp_path)
        docs   = loader.load()
        return "\n".join(doc.page_content for doc in docs if doc.page_content)
    finally:
        os.remove(tmp_path)
