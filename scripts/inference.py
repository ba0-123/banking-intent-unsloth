"""
inference.py
------------
Standalone inference file cho bài toán Intent Classification (BANKING77).
Đúng format yêu cầu: class IntentClassification với __init__ và __call__.
"""

import json
import yaml
import torch
from unsloth import FastLanguageModel


class IntentClassification:
    """
    Lớp inference cho mô hình phân loại intent ngân hàng.

    Ví dụ sử dụng:
        classifier = IntentClassification("configs/inference.yaml")
        label = classifier("I lost my credit card")
        print(label)  # -> "lost_or_stolen_card"
    """

    def __init__(self, model_path: str):
        """
        Tải config, tokenizer và model checkpoint.

        Args:
            model_path: Đường dẫn tới file config YAML (inference.yaml).
                        File config phải chứa trường 'checkpoint_dir'.
        """
        # Đọc config
        with open(model_path, "r") as f:
            cfg = yaml.safe_load(f)

        checkpoint_dir    = cfg["checkpoint_dir"]
        self.max_new_tokens = cfg.get("max_new_tokens", 20)
        self.max_seq_length = cfg.get("max_seq_length", 256)
        self.temperature    = cfg.get("temperature", 0.01)

        # Tải label map
        label_map_path = cfg.get(
            "label_map_path",
            f"{checkpoint_dir}/label_map.json"
        )
        with open(label_map_path, "r") as f:
            label_map = json.load(f)
        self.label2id: dict = label_map["label2id"]
        self.id2label: dict = {int(k): v for k, v in label_map["id2label"].items()}

        # Tải model & tokenizer bằng Unsloth
        print(f"[IntentClassification] Đang tải model từ: {checkpoint_dir}")
        self.model, self.tokenizer = FastLanguageModel.from_pretrained(
            model_name=checkpoint_dir,
            max_seq_length=self.max_seq_length,
            dtype=None,
            load_in_4bit=cfg.get("load_in_4bit", True),
        )
        FastLanguageModel.for_inference(self.model)
        self.model.eval()
        print("[IntentClassification] ✅ Model đã sẵn sàng!")

    def _build_prompt(self, message: str) -> str:
        return (
            "You are a banking assistant. "
            "Classify the following customer message into one of the banking intent categories.\n\n"
            f"Message: {message}\n\n"
            "Intent:"
        )

    def _postprocess(self, raw: str) -> str:
        """Chuẩn hoá output của model về dạng label hợp lệ."""
        pred = raw.strip().lower()
        pred = pred.replace("-", "_").replace(" ", "_")
        # Lấy từ đầu tiên (tránh model sinh thêm chữ)
        pred = pred.split("\n")[0].strip()

        if pred in self.label2id:
            return pred

        # Fallback: prefix match
        candidates = [k for k in self.label2id if k.startswith(pred[:6])]
        if candidates:
            return candidates[0]

        # Fallback cuối: lấy label có substring match dài nhất
        best, best_len = list(self.label2id.keys())[0], 0
        for k in self.label2id:
            common = sum(a == b for a, b in zip(k, pred))
            if common > best_len:
                best, best_len = k, common
        return best

    def __call__(self, message: str) -> str:
        """
        Phân loại intent của một tin nhắn khách hàng.

        Args:
            message: Câu hỏi / yêu cầu của khách hàng (tiếng Anh).

        Returns:
            predicted_label: Tên intent (ví dụ: "card_payment_fee_charged").
        """
        prompt = self._build_prompt(message)
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_seq_length,
        ).to("cuda" if torch.cuda.is_available() else "cpu")

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                temperature=self.temperature,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        # Chỉ lấy phần model sinh ra (sau prompt)
        generated_ids = outputs[0][inputs["input_ids"].shape[1]:]
        raw_output = self.tokenizer.decode(generated_ids, skip_special_tokens=True)

        predicted_label = self._postprocess(raw_output)
        return predicted_label


# ─── Chạy trực tiếp để demo ──────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Banking Intent Inference")
    parser.add_argument("--config", default="configs/inference.yaml",
                        help="Path to inference config YAML")
    parser.add_argument("--message", type=str,
                        help="Input message (nếu không truyền sẽ vào interactive mode)")
    args = parser.parse_args()

    classifier = IntentClassification(args.config)

    if args.message:
        # Single prediction
        label = classifier(args.message)
        print(f"\nInput  : {args.message}")
        print(f"Intent : {label}")
    else:
        # Interactive mode
        print("\n💬 Interactive Mode - nhập 'quit' để thoát\n")
        while True:
            msg = input("Message: ").strip()
            if msg.lower() in ("quit", "exit", "q"):
                break
            if not msg:
                continue
            label = classifier(msg)
            print(f"Intent : {label}\n")
