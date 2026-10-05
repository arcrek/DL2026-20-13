---
phase: 1
title: "Module Scaffold & Data Pipeline"
status: pending
priority: P1
effort: "2h"
dependencies: []
---

# Phase 1: Module Scaffold & Data Pipeline

## Overview
Khởi tạo cấu trúc module độc lập cho thành viên B tại `src/models/text.py` nhằm cách ly hoàn toàn với các module của thành viên C (`image.py`) và D (`fusion.py`), đồng thời xây dựng quy trình nạp dữ liệu văn bản đảm bảo tuyệt đối không rò rỉ meme template.

## Requirements
- **Functional:**
  - Nạp dữ liệu từ `data/memotion/*.jsonl` thông qua hàm tiện ích `load_jsonl()` trong [`src/data.py`](../../src/data.py).
  - Phân chia tập dữ liệu huấn luyện thành `fit` và `holdout` dựa trên [`features/train_holdout.json`](../../features/train_holdout.json).
  - Trích xuất nhãn phân loại 3 lớp: Negative (0), Neutral (1), Positive (2).
  - Trích xuất nhãn `sarcasm` để phục vụ hợp đồng dữ liệu cho phân tích chuyên sâu của thành viên E.
  - Sử dụng `AutoTokenizer.from_pretrained("bert-base-uncased")` với cấu hình `padding=True`, `truncation=True`, `max_length=128`.
- **Non-functional:**
  - **Bảo mật dữ liệu & Kiểm soát rò rỉ:** Tuyệt đối không đọc trường `text_ocr`; chỉ dùng trường `text` (được trích từ `text_corrected`).
  - **Tách biệt mã nguồn:** Không chỉnh sửa trực tiếp vào [`src/finetune.py`](../../src/finetune.py) để tránh xung đột git giữa các thành viên.

## Architecture
```
data/memotion/train.jsonl ──┐
                            ├──> TextMemeDataset ──> DataLoader(batch=32, shuffle=True)
features/train_holdout.json ┘            │
                                         ▼
                            Tokenizer('bert-base-uncased')
                               (max_len=128, return_pt)
                                         │
                                         ▼
                             Batch: {ids, input_ids, mask, y, sarcasm}
```

## Related Code Files
- Create: [`src/models/text.py`](../../src/models/text.py)
- Reference: [`src/data.py`](../../src/data.py)
- Reference: [`configs/base.yaml`](../../configs/base.yaml)

## Implementation Steps
1. Tạo thư mục `src/models/` nếu chưa tồn tại: `mkdir -p src/models`.
2. Tạo file `src/models/__init__.py`.
3. Trong `src/models/text.py`, định nghĩa class `TextMemeDataset(Dataset)` nạp `id`, `text`, `label` và `sarcasm`.
4. Viết hàm `make_collate_fn(tokenizer, max_length=128)` để gom batch và xử lý padding/truncation bằng PyTorch tensors.
5. Viết hàm `get_dataloaders(cfg, seed)` khởi tạo DataLoader cho `fit`, `holdout`, và `test` kèm `torch.Generator` để kiểm soát ngẫu nhiên.
6. Chạy thử nghiệm sanity check: Lấy 1 batch từ `fit_loader`, in kích thước `input_ids` và `attention_mask` (phải là `[32, 128]` hoặc nhỏ hơn theo batch padding) và kiểm tra nhãn `labels`.

## Success Criteria
- [ ] File `src/models/text.py` được tạo và import thành công.
- [ ] Batch sinh ra có đầy đủ các keys: `ids`, `input_ids`, `attention_mask`, `y`, `sarcasms`.
- [ ] Tập `fit` chứa đúng 5.033 mẫu, tập `holdout` chứa đúng 560 mẫu (tổng 5.593 mẫu train).
- [ ] Sanity check chạy không phát sinh lỗi hoặc cảnh báo tokenizer.

## Risk Assessment
- **Nguy cơ:** Tokenizer tải chậm hoặc lỗi kết nối mạng trên Colab/Kaggle khi tải từ Hugging Face Hub.
  - *Dấu hiệu:* `HTTPError` hoặc `ConnectionTimeout` khi gọi `AutoTokenizer.from_pretrained()`.
  - *Giải pháp:* Thiết lập retry logic hoặc lưu local cache model nếu cần.
- **Nguy cơ:** Tràn bộ nhớ (OOM) nếu meme có caption quá dài.
  - *Dấu hiệu:* Batch kích thước lớn gây tràn RAM/VRAM.
  - *Giải pháp:* Đảm bảo đã bật `truncation=True, max_length=128`.
