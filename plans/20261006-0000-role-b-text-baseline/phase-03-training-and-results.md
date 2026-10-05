---
phase: 3
title: "Training Loop & Contract Export (3 Seeds)"
status: completed
priority: P1
effort: "5h"
dependencies: ["1", "2"]
---

# Phase 3: Training Loop & Contract Export (3 Seeds)

## Overview
Thực hiện toàn bộ quá trình huấn luyện và đánh giá mô hình Text Baseline trên 3 seed ngẫu nhiên độc lập (`seed 0, 1, 2`), tự động chọn checkpoint tốt nhất theo Macro-F1 trên tập `holdout`, thực hiện dự đoán trên tập `test`, và xuất các tệp kết quả tuân thủ nghiêm ngặt hợp đồng dữ liệu tại `results/text_seed{0,1,2}.json`.

## Requirements
- **Functional:**
  - Vòng lặp huấn luyện chuẩn PyTorch với `epochs = 5` (hoặc `6`), `batch_size = 32`.
  - Hỗ trợ Mixed Precision (`torch.autocast("cuda", dtype=torch.bfloat16)` hoặc `float16`) để tiết kiệm VRAM và tăng tốc huấn luyện trên GPU.
  - Sau mỗi epoch, tính toán đánh giá trên tập `holdout`:
    - Macro-F1 (trung bình không trọng số F1 của 3 lớp).
    - Lưu giữ `best_state_dict` tại epoch có `holdout_macro_f1` cao nhất.
  - Khi hoàn thành huấn luyện mỗi seed:
    - Nạp lại `best_state_dict`.
    - Chạy inference trên tập `test` (700 mẫu).
    - Trích xuất: `ids`, `y_true`, `probs` (ma trận softmax kích thước `[700, 3]`), `sarcasm`.
  - Gọi hàm `save_results()` trong [`src/results.py`](../../src/results.py) để xuất kết quả:
    - Đường dẫn: `results/text_seed{seed}.json`.
    - Kiểm tra hợp đồng: `validate_result(res)`.
  - Hỗ trợ CLI arguments: `--seeds 0 1 2`, `--epochs 5`, `--batch 32`, `--lr-backbone 1.5e-5`, `--lr-head 5e-4`.
- **Non-functional:**
  - Tính tái lập (Reproducibility): Cố định chặt chẽ seed cho `random`, `np.random`, `torch.manual_seed`, `torch.cuda.manual_seed_all` và PyTorch DataLoader `generator`.
  - Không làm rò rỉ dữ liệu: Tập `test` chỉ được suy luận 1 lần duy nhất sau khi đã chọn model xong.

## Architecture
```
For seed in [0, 1, 2]:
   Set all RNG seeds
   Train on fit_loader (5 epochs)
   Evaluate holdout_loader each epoch -> Track best holdout Macro-F1
   Load best model checkpoint
   Evaluate test_loader -> Softmax Probs
   Call save_results("text", seed, ids, y_true, probs, sarcasm, split="test")
   Output: results/text_seed{seed}.json
```

## Related Code Files
- Modify: [`src/models/text.py`](../../src/models/text.py)
- Reference: [`src/results.py`](../../src/results.py)
- Reference: [`tests/test_results.py`](../../tests/test_results.py)
- Generate: `results/text_seed0.json`, `results/text_seed1.json`, `results/text_seed2.json`

## Implementation Steps
1. Viết hàm `evaluate(model, dataloader, device)` trả về `all_probs`, `all_ys`, `all_ids`, `all_sarcasms`.
2. Viết hàm tính chỉ số `compute_metrics(y_true, probas)` tính Macro-F1 và Accuracy.
3. Viết hàm `train_seed(seed, cfg, device)`:
   - Cố định toàn bộ seed số ngẫu nhiên.
   - Huấn luyện qua từng epoch với gradient clipping (`clip_grad_norm_ <= 1.0`).
   - Theo dõi F1 trên `holdout` và snapshot state dict tốt nhất.
   - Chạy suy luận trên `test` với state dict tốt nhất.
   - Gọi `save_results("text", seed, test_ids, test_y_true, test_probs, test_sarcasm, split="test")`.
4. Viết hàm `main()` với `argparse` để người dùng có thể chạy dễ dàng:
   ```bash
   python -m src.models.text --seeds 0 1 2
   ```
5. Chạy lệnh kiểm thử hợp đồng:
   ```bash
   python tests/test_results.py
   pytest tests/test_results.py
   ```

## Success Criteria
- [ ] 3 tệp `results/text_seed0.json`, `results/text_seed1.json`, `results/text_seed2.json` được tạo thành công.
- [ ] Hàm `validate_result()` trong `src/results.py` trả về `True` không phát sinh ngoại lệ cho cả 3 file.
- [ ] Độ dài danh sách `ids`, `y_true`, `y_pred`, `probs`, `sarcasm` trong mỗi file đều bằng đúng kích thước tập test (700 mẫu).
- [ ] Xác suất trong `probs` đều là số thực hữu hạn, nằm trong khoảng `[0, 1]` và tổng mỗi dòng bằng đúng 1.0.
- [ ] `pytest tests/test_results.py` vượt qua toàn bộ test cases.

## Risk Assessment
- **Nguy cơ:** Thời gian huấn luyện vượt quá khung 6-18h nếu huấn luyện tuần tự trên GPU yếu.
  - *Dấu hiệu:* Mỗi epoch mất > 10 phút, 3 seed mất > 3 giờ.
  - *Giải pháp:* Kích hoạt `torch.autocast`, tăng `batch_size` lên 32 hoặc 64 (nếu VRAM cho phép), chỉ huấn luyện 5 epochs.
- **Nguy cơ:** Quên lưu nhãn `sarcasm` dẫn đến việc hàm `save_results()` báo lỗi `missing keys`.
  - *Dấu hiệu:* Ngoại lệ `ValueError: missing keys: ['sarcasm']`.
  - *Giải pháp:* Đảm bảo Dataset và Collate trả về trường `sarcasm` lấy từ JSONL gốc.
