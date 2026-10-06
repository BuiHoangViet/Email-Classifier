"""
utils/data_loader.py
Đọc và chuẩn hoá các file CSV dataset.
Hỗ trợ hai schema:
  (a) labels / texts_vi   ← vi_dataset.csv
  (b) label  / text       ← sms_spam_vi.csv  (hoặc v1/v2)
"""
from __future__ import annotations

from pathlib import Path
import pandas as pd


_SCHEMA_MAP: list[tuple[list[str], list[str]]] = [
    (["labels", "texts_vi"], ["labels", "texts_vi"]),
    (["label",  "text"],     ["label",  "text"]),
    (["v1",     "v2"],       ["v1",     "v2"]),
]


def _detect_columns(df: pd.DataFrame) -> tuple[str, str]:
    for cols, _ in _SCHEMA_MAP:
        if all(c in df.columns for c in cols):
            return cols[0], cols[1]
    raise ValueError(
        f"Không nhận ra schema CSV. Cột hiện có: {list(df.columns)}"
    )


def load_csv(path: str | Path) -> list[tuple[str, str]]:
    """
    Trả về list[(label, text)].
    label đã được chuẩn hoá thành 'spam' hoặc 'ham'.
    """
    df = pd.read_csv(path, encoding="utf-8")
    label_col, text_col = _detect_columns(df)
    df = df[[label_col, text_col]].dropna()
    df.columns = ["label", "text"]

    df["label"] = df["label"].astype(str).str.lower().str.strip()
    df = df[df["label"].isin(["spam", "ham"])]

    return list(df.itertuples(index=False, name=None))
