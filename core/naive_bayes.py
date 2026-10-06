"""
core/naive_bayes.py
Toàn bộ logic Naive Bayes: tiền xử lý → train → predict.
"""
from __future__ import annotations

import math
import re
from typing import Iterable

# ── Pattern một lần ────────────────────────────────────────────────────────────
_URL_RE      = re.compile(r"https?://\S+|www\.\S+")
_DIGIT_RE    = re.compile(r"\d+")
_NON_WORD_RE = re.compile(
    r"[^\w\sàáạảãâầấậẩẫăằắặẳẵèéẹẻẽêềếệểễìíịỉĩ"
    r"òóọỏõôồốộổỗơờớợởỡùúụủũưừứựửữỳýỵỷỹđ]"
)
_SPACE_RE    = re.compile(r"\s+")


def clean_text(text: str | None) -> str:
    """Chuẩn hoá văn bản tiếng Việt."""
    if not text:
        return ""
    if not isinstance(text, str):
        text = str(text)
    text = text.lower()
    text = _URL_RE.sub(" ", text)
    text = _DIGIT_RE.sub(" ", text)
    text = _NON_WORD_RE.sub(" ", text)
    return _SPACE_RE.sub(" ", text).strip()


def tokenize(text: str) -> list[str]:
    return clean_text(text).split()


# ── Kiểu dữ liệu trả về từ train ───────────────────────────────────────────────
type WordCounts  = dict[str, dict[str, int]]
type ClassCounts = dict[str, int]
type Vocab       = set[str]


def train(
    dataset: Iterable[tuple[str, str]],
) -> tuple[WordCounts, ClassCounts, Vocab]:
    """
    Huấn luyện Multinomial Naive Bayes với Laplace smoothing.

    Parameters
    ----------
    dataset : iterable of (label, text)  –  label phải là 'spam' hoặc 'ham'

    Returns
    -------
    word_counts, class_counts, vocab
    """
    word_counts:  WordCounts  = {"spam": {}, "ham": {}}
    class_counts: ClassCounts = {"spam": 0,  "ham": 0}
    vocab:        Vocab       = set()

    for label, text in dataset:
        words = tokenize(text)
        if not words:
            continue
        class_counts[label] += 1
        for word in words:
            vocab.add(word)
            word_counts[label][word] = word_counts[label].get(word, 0) + 1

    return word_counts, class_counts, vocab


def _log_scores(
    text: str,
    word_counts: WordCounts,
    class_counts: ClassCounts,
    vocab: Vocab,
) -> dict[str, float]:
    """Log-xác suất (chưa chuẩn hoá) của từng nhãn."""
    words       = tokenize(text)
    total_docs  = sum(class_counts.values())
    vocab_size  = len(vocab)
    log_probs:  dict[str, float] = {}

    for label in ("spam", "ham"):
        if total_docs == 0 or class_counts[label] == 0:
            log_probs[label] = float("-inf")
            continue

        lp          = math.log(class_counts[label] / total_docs)
        total_words = sum(word_counts[label].values())

        for word in words:
            if word not in vocab:          # bỏ qua từ OOV
                continue
            freq  = word_counts[label].get(word, 0) + 1          # Laplace +1
            lp   += math.log(freq / (total_words + vocab_size))

        log_probs[label] = lp

    return log_probs


def predict(
    text: str,
    word_counts: WordCounts,
    class_counts: ClassCounts,
    vocab: Vocab,
) -> str:
    """Trả về 'spam' hoặc 'ham'."""
    log_probs = _log_scores(text, word_counts, class_counts, vocab)
    return max(log_probs, key=log_probs.get)


def spam_probability(
    text: str,
    word_counts: WordCounts,
    class_counts: ClassCounts,
    vocab: Vocab,
) -> float:
    """Xác suất (0–1) văn bản là spam."""
    lp = _log_scores(text, word_counts, class_counts, vocab)
    if lp["spam"] == lp["ham"]:
        return 0.5
    # sigmoid(spam - ham), tránh tràn số khi chênh lệch lớn
    diff = lp["spam"] - lp["ham"]
    if diff >= 0:
        return 1 / (1 + math.exp(-diff))
    e = math.exp(diff)
    return e / (1 + e)
