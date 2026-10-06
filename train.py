"""
train.py
Script huấn luyện Naive Bayes và lưu model.pkl.
Chạy: python train.py [--dataset data/vi_dataset.csv] [--model models/model.pkl]
"""
from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path

from sklearn.model_selection import train_test_split

# Thêm thư mục gốc vào sys.path để import config / core / utils
sys.path.insert(0, str(Path(__file__).parent))

from config import (
    DATASET_PATH, MODEL_PATH,
    TEST_SIZE, RANDOM_STATE, MAX_WORDS, MIN_WORDS,
)
from core.naive_bayes import train, predict
from utils.data_loader import load_csv


def preprocess(raw: list[tuple[str, str]]) -> list[tuple[str, str]]:
    result = []
    for label, text in raw:
        words = text.split()
        if len(words) < MIN_WORDS:
            continue
        if len(words) > MAX_WORDS:
            text = " ".join(words[:MAX_WORDS])
        result.append((label, text))
    return result


def evaluate(model_data: dict, test_data: list[tuple[str, str]]) -> float:
    correct = sum(
        1 for label, text in test_data
        if predict(text, **model_data) == label
    )
    return correct / len(test_data) * 100 if test_data else 0.0


def run(dataset_path: Path, model_path: Path) -> None:
    print(f"[INFO] Đọc dữ liệu : {dataset_path}")
    raw  = load_csv(dataset_path)
    data = preprocess(raw)
    print(f"[INFO] Tổng mẫu   : {len(data)}  (sau lọc từ {len(raw)})")

    train_data, test_data = train_test_split(
        data,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=[l for l, _ in data],
    )
    print(f"[INFO] Train / Test: {len(train_data)} / {len(test_data)}")

    word_counts, class_counts, vocab = train(train_data)

    model_data = {
        "word_counts":  word_counts,
        "class_counts": class_counts,
        "vocab":        vocab,
    }

    acc = evaluate(model_data, test_data)
    print(f"[INFO] Accuracy    : {acc:.2f}%")

    model_path.parent.mkdir(parents=True, exist_ok=True)
    with open(model_path, "wb") as f:
        pickle.dump(model_data, f)
    print(f"[INFO] ✔ Model lưu : {model_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Naive Bayes spam classifier")
    parser.add_argument("--dataset", type=Path, default=DATASET_PATH)
    parser.add_argument("--model",   type=Path, default=MODEL_PATH)
    args = parser.parse_args()
    run(args.dataset, args.model)
