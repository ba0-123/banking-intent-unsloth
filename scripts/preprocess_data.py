"""
preprocess_data.py
------------------
Tải BANKING77 từ CSV, tiền xử lý và lưu train/test CSV.
"""

import os
import re
import yaml
import argparse
import pandas as pd
from sklearn.model_selection import train_test_split
import json


def normalize_text(text: str) -> str:
    """Chuẩn hoá văn bản: lowercase, xoá ký tự đặc biệt thừa."""
    text = str(text).lower().strip()
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
    max_per_intent = cfg.get("max_per_intent", None)

    os.makedirs(output_dir, exist_ok=True)

    print("📥 Đang tải BANKING77 từ CSV...")

    # ✅ Load từ CSV (thay vì HF)
    data_files_urls = {
        "train": "https://raw.githubusercontent.com/PolyAI-LDN/task-specific-datasets/master/banking_data/train.csv",
        "test":  "https://raw.githubusercontent.com/PolyAI-LDN/task-specific-datasets/master/banking_data/test.csv"
    }

    df_train_raw = pd.read_csv(data_files_urls["train"])
    df_test_raw  = pd.read_csv(data_files_urls["test"])

    # Gộp lại
    df_all = pd.concat([df_train_raw, df_test_raw], ignore_index=True)

    # ✅ category chính là intent
    df_all["intent"] = df_all["category"]

    print(f"✅ Tổng số mẫu: {len(df_all)}")
    print(f"✅ Số intents: {df_all['intent'].nunique()}")

    # Chuẩn hoá text
    df_all["text"] = df_all["text"].apply(normalize_text)

    # ✅ Lấy label_names từ data (THIẾU ở code cũ)
    label_names = sorted(df_all["intent"].unique().tolist())

    # Mapping
    label2id = {name: idx for idx, name in enumerate(label_names)}
    id2label = {idx: name for name, idx in label2id.items()}

    df_all["label_id"] = df_all["intent"].map(label2id)

    # Giới hạn số mẫu mỗi intent (nếu cần)
    if max_per_intent:
        df_all = (
            df_all.groupby("intent", group_keys=False)
            .apply(lambda g: g.sample(min(len(g), max_per_intent), random_state=random_seed))
            .reset_index(drop=True)
        )
        print(f"📊 Sau khi sampling: {len(df_all)} mẫu")

    # Split train/test
    df_train, df_test = train_test_split(
        df_all[["text", "intent", "label_id"]],
        test_size=test_size,
        stratify=df_all["label_id"],
        random_state=random_seed,
    )

    # Save
    df_train.to_csv(os.path.join(output_dir, "train.csv"), index=False)
    df_test.to_csv(os.path.join(output_dir, "test.csv"), index=False)

    with open(os.path.join(output_dir, "label_map.json"), "w") as f:
        json.dump(
            {
                "label2id": label2id,
                "id2label": id2label,
                "label_names": label_names
            },
            f,
            indent=2
        )

    print("💾 Saved:")
    print(f"   train.csv : {len(df_train)} samples")
    print(f"   test.csv  : {len(df_test)} samples")
    print(f"   label_map.json : {len(label_names)} intents")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/train.yaml")
    args = parser.parse_args()
    main(args.config)