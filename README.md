# Banking Intent Classification with Unsloth

Fine-tuning **LLaMA-3.2-1B-Instruct** trên dataset **BANKING77** (77 intents) sử dụng **Unsloth** + LoRA.

> Project 2 — NLP Industry | HCMUS Faculty of Information Technology

---

## Cấu trúc thư mục

```
banking-intent-unsloth/
├── scripts/
│   ├── preprocess_data.py   # Tải & tiền xử lý BANKING77
│   ├── train.py             # Fine-tune với Unsloth + LoRA
│   └── inference.py         # Inference class (IntentClassification)
├── configs/
│   ├── train.yaml           # Hyperparameters training
│   └── inference.yaml       # Config inference
├── sample_data/             # Sinh ra sau khi chạy preprocess
│   ├── train.csv
│   ├── test.csv
│   └── label_map.json
│
└── README.md
```

---

## Cài đặt môi trường (Google Colab)

### Bước 1 — Cài Unsloth (bắt buộc chạy đầu tiên)

```python
# Cell 1 trong Colab notebook
%%capture
import torch
cuda_version = torch.version.cuda.replace(".", "")

# Cài Unsloth phù hợp với CUDA version
!pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git" --quiet
!pip install --no-deps "xformers<0.0.29" peft accelerate bitsandbytes --quiet
!pip install datasets transformers scikit-learn pandas PyYAML --quiet
```

### Bước 2 — Clone repo

```bash
!git clone https://github.com/ba0-123/banking-intent-unsloth.git
%cd banking-intent-unsloth
!ls -la
```

### Bước 3 — Chạy scripts

```bash
# Bước 1: Tiền xử lý dữ liệu
!python scripts/preprocess_data.py --config configs/train.yaml

# Bước 2: Fine-tuning
!python scripts/train.py --config configs/train.yaml
```

---

## Inference

### Dùng script

```bash
# Predict một câu
!python scripts/inference.py \
    --config configs/inference.yaml \
    --message "I lost my credit card"

# Interactive mode
!python scripts/inference.py --config configs/inference.yaml
```

### Dùng class trực tiếp (trong Python / Colab)

```python
from scripts.inference import IntentClassification

# Khởi tạo (tải model một lần)
classifier = IntentClassification("configs/inference.yaml")

# Predict
print(classifier("I lost my credit card"))
# → lost_or_stolen_card

print(classifier("What is my current balance?"))
# → balance_inquiry

print(classifier("I want to cancel my transaction"))
# → cancel_transfer
```

---

## Hyperparameters

| Tham số | Giá trị | Ghi chú |
|---|---|---|
| Model | `unsloth/Llama-3.2-1B-Instruct` | 1B params, 4-bit quantized |
| LoRA rank (r) | 16 | Cân bằng tốt cho task này |
| LoRA alpha | 16 | = r để stable training |
| LoRA dropout | 0.05 | Regularization nhẹ |
| Batch size | 8 | per device |
| Gradient accum | 4 | Effective batch = 32 |
| Learning rate | 2e-4 | AdamW 8-bit optimizer |
| Epochs | 3 | Đủ cho ~13k mẫu train |
| Warmup ratio | 0.1 | 10% steps warm up |
| Weight decay | 0.01 | L2 regularization |
| Max seq length | 256 | Câu banking ngắn |
| LR scheduler | cosine | Giảm dần sau warmup |
| Optimizer | adamw_8bit | Tiết kiệm VRAM |

---

## Dataset

- **BANKING77**: 13,083 câu hỏi khách hàng ngân hàng, 77 intents
- **Train**: ~11,120 mẫu (85%)
- **Test**: ~1,963 mẫu (15%)
- **Nguồn**: [HuggingFace - PolyAI/banking77](https://huggingface.co/datasets/PolyAI/banking77)

### Một số intents mẫu

| Intent | Ví dụ câu hỏi |
|---|---|
| `card_payment_fee_charged` | "Why was I charged a fee for my card payment?" |
| `lost_or_stolen_card` | "My card was stolen, what should I do?" |
| `balance_inquiry` | "What's my current account balance?" |
| `cancel_transfer` | "I need to cancel a transfer I just made" |
| `exchange_rate` | "What is the exchange rate for USD to EUR?" |

---

## Kết quả

Sau khi training, kết quả được lưu tại `outputs/llama32-banking77/eval_results.txt`.

---

## Tác giả

- **Sinh viên**: Nguyễn Quốc Bảo
- **MSSV**: 23127329
- **Giảng viên**: Dr. Nguyen Hong Buu Long
