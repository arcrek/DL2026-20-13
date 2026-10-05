---
phase: 2
title: "Model Architecture & Differential LR"
status: pending
priority: P1
effort: "2h"
dependencies: ["1"]
---

# Phase 2: Model Architecture & Differential LR

## Overview
Xây dựng kiến trúc mô hình phân loại văn bản `BertMemeClassifier` sử dụng backbone `bert-base-uncased`, đồng thời thiết lập tối ưu hóa với **Differential Learning Rate** và hàm mất mát điều chỉnh theo trọng số lớp (`class_weighted_loss`) để xử lý hiện tượng mất cân bằng dữ liệu nghiêm trọng.

## Requirements
- **Functional:**
  - Backbone: `bert-base-uncased` (từ Hugging Face `AutoModel` hoặc `BertModel`).
  - Representation: Trích xuất vector biểu diễn token `[CLS]` (768 chiều, tương ứng với `pooler_output` hoặc `last_hidden_state[:, 0, :]`).
  - Dropout: Thêm lớp `nn.Dropout(p=0.2)` chống học vẹt (overfitting).
  - Classification Head: Lớp tuyến tính `nn.Linear(768, 3)`.
  - Differential Learning Rate:
    - Tham số của backbone (BERT): $lr = 1.5 \times 10^{-5}$
    - Tham số của classification head: $lr = 5.0 \times 10^{-4}$
    - Weight decay: $0.01$ (loại trừ bias và LayerNorm khỏi weight decay nếu có thể, hoặc áp dụng toàn cục).
  - LR Scheduler: Linear Warmup (10% tổng số bước huấn luyện) và Linear Decay.
  - Loss Function: `nn.CrossEntropyLoss(weight=class_weights)` với trọng số lớp nạp trực tiếp từ `features/train_holdout.json`.
- **Non-functional:**
  - Tách biệt rõ ràng 2 nhóm tham số `backbone_params` và `head_params` trong class mô hình.
  - Hỗ trợ chạy cả trên CPU và GPU CUDA.

## Architecture
```
input_ids, attention_mask
          │
          ▼
   bert-base-uncased (lr=1.5e-5)
          │
          ▼
   [CLS] vector (768d)
          │
          ▼
     Dropout(0.2)
          │
          ▼
Linear(768 -> 3) (lr=5e-4)
          │
          ▼
        Logits ──> CrossEntropyLoss(weight=class_weights)
```

## Related Code Files
- Modify: [`src/models/text.py`](../../src/models/text.py)
- Reference: [`src/finetune.py`](../../src/finetune.py)
- Reference: [`features/train_holdout.json`](../../features/train_holdout.json)

## Implementation Steps
1. Định nghĩa class `BertMemeClassifier(nn.Module)` trong `src/models/text.py`.
2. Viết phương thức `backbone_params()` và `head_params()` trong `BertMemeClassifier` để phân nhóm tham số sạch sẽ:
   ```python
   def head_params(self):
     return [
         p
         for n, p in self.named_parameters()
         if not n.startswith("bert.") and p.requires_grad
     ]

   def backbone_params(self):
     return [
         p
         for n, p in self.named_parameters()
         if n.startswith("bert.") and p.requires_grad
     ]
```
3. Viết hàm `build_optimizer_and_scheduler(model, num_training_steps, lr_backbone=1.5e-5, lr_head=5e-4)`:
   - Khởi tạo `torch.optim.AdamW`.
   - Khởi tạo `get_linear_schedule_with_warmup` từ `transformers` hoặc dùng `LambdaLR`.
4. Viết hàm nạp `class_weights` từ `features/train_holdout.json` và khởi tạo `loss_fn`.
5. Chạy forward pass thử nghiệm trên 1 dummy batch để xác nhận output logits có shape `[batch_size, 3]` và loss là scalar hữu hạn.

## Success Criteria
- [ ] Output logits của forward pass có shape chính xác `[B, 3]`.
- [ ] Tham số backbone và classification head được gán đúng tốc độ học riêng biệt trong optimizer.
- [ ] Loss tính toán thành công với `class_weights` không bị lỗi nan/inf.
- [ ] Gradient tính toán mượt mà khi gọi `loss.backward()`.

## Risk Assessment
- **Nguy cơ:** Catastrophic Forgetting nếu tốc độ học của BERT quá cao.
  - *Dấu hiệu:* Loss dao động mạnh, Macro-F1 không tăng hoặc sụt giảm sau epoch 1.
  - *Giải pháp:* Giữ chặt LR của BERT ở mức $1.5 \times 10^{-5}$, không vượt quá $3.0 \times 10^{-5}$.
- **Nguy cơ:** Sụp hố dự đoán đa số (Majority Class Collapse) do mất cân bằng dữ liệu.
  - *Dấu hiệu:* Mô hình chỉ dự đoán lớp Positive (2), Macro-F1 sụt xuống dưới 0.35.
  - *Giải pháp:* Kiểm tra xem `CrossEntropyLoss` đã truyền đúng `class_weights` chưa.
