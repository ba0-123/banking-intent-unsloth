"""
preprocess_data.py
------------------
Tải BANKING77, lấy tất cả 77 intents, tiền xử lý và lưu train/test CSV.
"""

import os
import re
import yaml
import argparse
import pandas as pd
from datasets import load_dataset
from sklearn.model_selection import train_test_split


def normalize_text(text: str) -> str:
    """Chuẩn hoá văn bản: lowercase, xoá ký tự đặc biệt thừa."""
    text = text.lower().strip()
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[^\w\s\?\.\,\!\'\-]", "", text)
    return text


def load_config(config_path: str) -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def main(config_path: str):
    cfg = load_config(config_path)

    output_dir = cfg.get("output_dir", "sample_data")
    test_size   = cfg.get("test_size", 0.15)
    random_seed = cfg.get("random_seed", 42)
    max_per_intent = cfg.get("max_per_intent", None)  # None = dùng hết

    os.makedirs(output_dir, exist_ok=True)

    print("📥 Đang tải BANKING77 từ HuggingFace...")
    dataset = load_dataset("PolyAI/banking77")

    # Gộp train + test gốc lại rồi tự chia
    df_train_raw = dataset["train"].to_pandas()
    df_test_raw  = dataset["test"].to_pandas()
    df_all = pd.concat([df_train_raw, df_test_raw], ignore_index=True)

    # Lấy label names
    label_names = dataset["train"].features["label"].names  # 77 nhãn
    df_all["intent"] = df_all["label"].apply(lambda x: label_names[x])

    print(f"✅ Tổng số mẫu: {len(df_all)}, số intents: {df_all['label'].nunique()}")

    # Chuẩn hoá text
    df_all["text"] = df_all["text"].apply(normalize_text)

    # Giới hạn số mẫu mỗi intent (nếu cần)
    if max_per_intent:
        df_all = (
            df_all.groupby("label", group_keys=False)
            .apply(lambda g: g.sample(min(len(g), max_per_intent), random_state=random_seed))
            .reset_index(drop=True)
        )
        print(f"📊 Sau khi lấy mẫu: {len(df_all)} mẫu")

    # Tạo label_id liên tục (0-76)
    label2id = {name: idx for idx, name in enumerate(label_names)}
    id2label = {idx: name for name, idx in label2id.items()}
    df_all["label_id"] = df_all["intent"].map(label2id)

    # Chia train / test
    df_train, df_test = train_test_split(
        df_all[["text", "intent", "label_id"]],
        test_size=test_size,
        stratify=df_all["label_id"],
        random_state=random_seed,
    )

    df_train.to_csv(os.path.join(output_dir, "train.csv"), index=False)
    df_test.to_csv(os.path.join(output_dir, "test.csv"),  index=False)

    # Lưu mapping nhãn
    import json
    with open(os.path.join(output_dir, "label_map.json"), "w") as f:
        json.dump({"label2id": label2id, "id2label": id2label}, f, indent=2)

    print(f"💾 Đã lưu:")
    print(f"   train.csv : {len(df_train)} mẫu")
    print(f"   test.csv  : {len(df_test)}  mẫu")
    print(f"   label_map.json : 77 intents")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/train.yaml")
    args = parser.parse_args()
    main(args.config)
