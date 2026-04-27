"""
train.py
--------
Fine-tune LLaMA-3.2-1B dùng Unsloth cho bài toán Intent Classification (BANKING77).
Chạy trên Google Colab Free (T4 GPU, ~15GB VRAM).
"""

import os
import json
import yaml
import argparse
import numpy as np
import pandas as pd
from datasets import Dataset
from sklearn.metrics import accuracy_score, classification_report


# ── Unsloth (phải import trước transformers) ─────────────────────────────────
from unsloth import FastLanguageModel
import torch

from transformers import TrainingArguments
from trl import SFTTrainer


# ─────────────────────────────────────────────────────────────────────────────
def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def build_prompt(text: str, label: str = "") -> str:
    """
    Định dạng prompt instruction cho sequence classification.
    Khi inference, label để trống.
    """
    instruction = (
        "You are a banking assistant. "
        "Classify the following customer message into one of the banking intent categories.\n\n"
        f"Message: {text}\n\n"
        "Intent:"
    )
    if label:
        return instruction + f" {label}" + tokenizer.eos_token
    return instruction


def preprocess_dataset(df: pd.DataFrame) -> Dataset:
    """Chuyển DataFrame thành HuggingFace Dataset với prompt."""
    records = []
    for _, row in df.iterrows():
        prompt = build_prompt(row["text"], row["intent"])
        records.append({"text": prompt})
    return Dataset.from_list(records)


def evaluate_model(model, tokenizer, df_test: pd.DataFrame, label2id: dict, id2label: dict, max_new_tokens: int = 20):
    """Chạy inference trên test set và tính accuracy."""
    FastLanguageModel.for_inference(model)
    model.eval()

    preds = []
    labels = []

    print(f"\n🔍 Đánh giá trên {len(df_test)} mẫu test...")
    for i, (_, row) in enumerate(df_test.iterrows()):
        prompt = build_prompt(row["text"])
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=256).to("cuda")

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=0.01,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )

        generated = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        pred_label = generated.strip().lower().replace("-", "_").replace(" ", "_")

        # Map về label hợp lệ gần nhất
        if pred_label not in label2id:
            # Tìm label gần nhất (prefix match)
            matched = [k for k in label2id if k.startswith(pred_label[:5])]
            pred_label = matched[0] if matched else list(label2id.keys())[0]

        preds.append(label2id[pred_label])
        labels.append(row["label_id"])

        if (i + 1) % 100 == 0:
            print(f"  {i+1}/{len(df_test)} done...")

    acc = accuracy_score(labels, preds)
    print(f"\n✅ Test Accuracy: {acc:.4f} ({acc*100:.2f}%)")
    return acc, preds, labels


def main(config_path: str):
    global tokenizer  # dùng trong build_prompt

    cfg = load_config(config_path)

    # ── Config ──────────────────────────────────────────────────────────────
    model_name      = cfg["model_name"]          # "unsloth/Llama-3.2-1B-Instruct"
    max_seq_length  = cfg["max_seq_length"]       # 256
    load_in_4bit    = cfg["load_in_4bit"]         # True
    lora_r          = cfg["lora_r"]               # 16
    lora_alpha      = cfg["lora_alpha"]           # 16
    lora_dropout    = cfg["lora_dropout"]         # 0.05
    batch_size      = cfg["batch_size"]           # 8
    grad_accum      = cfg["gradient_accumulation_steps"]  # 4
    learning_rate   = cfg["learning_rate"]        # 2e-4
    num_epochs      = cfg["num_epochs"]           # 3
    warmup_ratio    = cfg["warmup_ratio"]         # 0.1
    weight_decay    = cfg["weight_decay"]         # 0.01
    output_dir      = cfg["output_dir"]           # "outputs/llama32-banking77"
    data_dir        = cfg["data_dir"]             # "sample_data"
    run_eval        = cfg.get("run_eval", True)

    os.makedirs(output_dir, exist_ok=True)

    # ── Load label map ───────────────────────────────────────────────────────
    with open(os.path.join(data_dir, "label_map.json")) as f:
        label_map = json.load(f)
    label2id: dict = label_map["label2id"]
    id2label: dict = {int(k): v for k, v in label_map["id2label"].items()}
    num_labels = len(label2id)
    print(f"📌 Số intents: {num_labels}")

    # ── Load model + tokenizer (Unsloth) ────────────────────────────────────
    print(f"\n🚀 Tải model: {model_name}")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=max_seq_length,
        dtype=None,          # auto-detect (bfloat16 nếu có)
        load_in_4bit=load_in_4bit,
    )

    # ── Thêm LoRA adapters ───────────────────────────────────────────────────
    model = FastLanguageModel.get_peft_model(
        model,
        r=lora_r,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                         "gate_proj", "up_proj", "down_proj"],
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        bias="none",
        use_gradient_checkpointing="unsloth",  # tiết kiệm VRAM
        random_state=42,
        use_rslora=False,
    )

    print(model.print_trainable_parameters())

    # ── Load data ────────────────────────────────────────────────────────────
    df_train = pd.read_csv(os.path.join(data_dir, "train.csv"))
    df_test  = pd.read_csv(os.path.join(data_dir, "test.csv"))
    print(f"📊 Train: {len(df_train)} | Test: {len(df_test)}")

    train_dataset = preprocess_dataset(df_train)

    # ── Training Arguments ───────────────────────────────────────────────────
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=grad_accum,
        learning_rate=learning_rate,
        warmup_ratio=warmup_ratio,
        weight_decay=weight_decay,
        lr_scheduler_type="cosine",
        optim="adamw_8bit",             # tiết kiệm VRAM
        fp16=not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_bf16_supported(),
        logging_steps=50,
        save_strategy="epoch",
        save_total_limit=1,
        report_to="none",
        seed=42,
        dataloader_num_workers=2,
    )

    # ── SFT Trainer ──────────────────────────────────────────────────────────
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        dataset_text_field="text",
        max_seq_length=max_seq_length,
        dataset_num_proc=2,
        packing=True,   # ghép nhiều mẫu ngắn vào 1 sequence → tăng tốc
        args=training_args,
    )

    # ── Train ────────────────────────────────────────────────────────────────
    print("\n🏋️  Bắt đầu training...")
    trainer_stats = trainer.train()
    print(f"\n✅ Training hoàn tất!")
    print(f"   Time: {trainer_stats.metrics['train_runtime']:.1f}s")
    print(f"   Loss: {trainer_stats.metrics['train_loss']:.4f}")

    # ── Save checkpoint ──────────────────────────────────────────────────────
    checkpoint_dir = os.path.join(output_dir, "checkpoint-final")
    model.save_pretrained(checkpoint_dir)
    tokenizer.save_pretrained(checkpoint_dir)
    print(f"\n💾 Checkpoint lưu tại: {checkpoint_dir}")

    # Lưu label map vào checkpoint
    import shutil
    shutil.copy(
        os.path.join(data_dir, "label_map.json"),
        os.path.join(checkpoint_dir, "label_map.json"),
    )

    # ── Evaluate ─────────────────────────────────────────────────────────────
    if run_eval:
        acc, preds, true_labels = evaluate_model(
            model, tokenizer, df_test, label2id, id2label
        )
        # Lưu kết quả
        report = classification_report(
            true_labels, preds,
            target_names=[id2label[i] for i in sorted(id2label)],
            zero_division=0,
        )
        with open(os.path.join(output_dir, "eval_results.txt"), "w") as f:
            f.write(f"Test Accuracy: {acc:.4f}\n\n")
            f.write(report)
        print(f"\n📄 Kết quả đã lưu: {output_dir}/eval_results.txt")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/train.yaml")
    args = parser.parse_args()
    main(args.config)
