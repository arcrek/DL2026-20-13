---
phase: 4
title: "Downstream Handoff & Report Contribution"
status: pending
priority: P2
effort: "3h"
dependencies: ["3"]
---

# Phase 4: Downstream Handoff & Report Contribution

## Overview
Sau khi hoàn thành huấn luyện Text Baseline và xuất 3 file JSON ở Giờ 18, thành viên B tiến hành bàn giao kết quả cho Thành viên E (Phân tích thực nghiệm) và Thành viên F (Soạn thảo báo cáo), đồng thời trực tiếp hỗ trợ phân tích chuyên sâu các trường hợp mô hình Text-only gặp hạn chế (đặc biệt là nhóm meme châm biếm / sarcasm) và soạn thảo các mục báo cáo liên quan.

## Requirements
- **Functional:**
  - Tính toán bảng tổng hợp `mean ± std` của Macro-F1 và Accuracy qua 3 seed của Text Baseline.
  - Phối hợp với Thành viên E:
    - Chạy **Paired Bootstrap Test / Permutation Test** so sánh hiệu năng của Text-only vs Multimodal Fusion (`both_concat`, `both_cross_attn`) để lấy chỉ số $p$-value.
    - Chạy phân tích nhóm con (Sub-group Analysis): Đánh giá Macro-F1 của Text-only trên tập meme có tính châm biếm (`sarcasm` khác `not_sarcastic`) so với nhóm không châm biếm (`not_sarcastic`).
    - Xuất biểu đồ ma trận nhầm lẫn (Confusion Matrix) của Text Baseline vào thư mục `report/figures/cm_text.png`.
  - Phối hợp với Thành viên F:
    - Soạn thảo **Mục 4 (Thiết kế mô hình & Chiến lược huấn luyện Text)**: Trình bày chi tiết cơ chế BERT, phân tích lý do cần Differential LR và hàm loss có trọng số `class_weights`.
    - Cung cấp số liệu chính xác cho **Mục 5 (Kết quả thực nghiệm định lượng)**.
    - Cung cấp 3-5 ca lỗi định tính của Text-only cho **Mục 6 (Phân tích lỗi)**: Chỉ ra các meme mà chữ có nghĩa tích cực/trung tính nhưng ảnh mang hàm ý tiêu cực/châm biếm, dẫn đến Text-only bị đoán sai.
  - Phối hợp với Thành viên A:
    - Tích hợp pipeline Text baseline vào notebook chung [`notebooks/meme_understanding.ipynb`](../../notebooks/meme_understanding.ipynb) đảm bảo notebook chạy từ đầu đến cuối không lỗi trên môi trường Colab/Kaggle sạch.
- **Non-functional:**
  - Bàn giao kết quả đúng hạn tại Giờ 18 để không làm chậm trễ tiến độ phân tích của E.
  - Trình bày mạch lạc, số liệu thống kê đầy đủ độ lệch chuẩn.

## Architecture
```
results/text_seed{0,1,2}.json ──┬──> Thành viên E ──> Bootstrap / Permutation Test (p < 0.05)
                                │                 └──> Sarcasm Sub-group Analysis
                                │                 └──> Confusion Matrix
                                │
                                ├──> Thành viên F ──> Báo cáo Mục 4 (Text Architecture)
                                │                 └──> Báo cáo Mục 5 (Quantitative Table)
                                │                 └──> Báo cáo Mục 6 (Qualitative Error Analysis)
                                │
                                └──> Thành viên A ──> Tích hợp Colab Notebook
```

## Related Code Files
- Reference: `results/text_seed*.json`
- Reference: [`src/results.py`](../../src/results.py)
- Contribute: [`notebooks/meme_understanding.ipynb`](../../notebooks/meme_understanding.ipynb)
- Contribute: `report/` (hoặc tài liệu báo cáo của F)
- Output: `report/figures/cm_text.png`

## Implementation Steps
1. Viết script nhỏ hoặc hàm tổng hợp nhanh bảng số liệu từ 3 file kết quả:
   ```python
   import numpy as np
   from src.results import load_results

   f1s = [load_results("text", s)["macro_f1"] for s in [0, 1, 2]]
   accs = [load_results("text", s)["accuracy"] for s in [0, 1, 2]]
   print(f"Text Macro-F1: {np.mean(f1s):.4f} +/- {np.std(f1s):.4f}")
   print(f"Text Accuracy: {np.mean(accs):.4f} +/- {np.std(accs):.4f}")
```
2. Phối hợp với E để đưa danh sách `probs` và `y_true` vào script kiểm định thống kê `paired_bootstrap_test(y_true, probs_text, probs_fusion)`.
3. Lọc ra các ca lỗi tiêu biểu của Text:
   - Những mẫu có `y_true == 0` (negative) nhưng Text-only dự đoán `y_pred == 2` (positive) do caption chứa từ ngữ tích cực giả tạo (sarcasm).
4. Viết đoạn mô tả phương pháp và kết quả gửi cho F tổng hợp vào bản thảo báo cáo cuối kỳ.
5. Chạy thử kiểm tra lại notebook với A trên môi trường Colab.

## Success Criteria
- [ ] Bảng số liệu `mean ± std` của Text Baseline sẵn sàng trước Giờ 20.
- [ ] Thành viên E hoàn thành kiểm định thống kê $p$-value giữa Multimodal và Text-only.
- [ ] Báo cáo Mục 4, 5, 6 có đầy đủ hình ảnh và phân tích của Text Baseline.
- [ ] Module `src/models/text.py` hoạt động trơn tru trong notebook Colab.

## Risk Assessment
- **Nguy cơ:** Kết quả Text-only có phương sai lớn giữa các seed (std cao).
  - *Dấu hiệu:* Một seed đạt Macro-F1 0.40, seed khác đạt 0.31.
  - *Giải pháp:* Kiểm tra lại quá trình warmup của optimizer hoặc tăng số lượng epoch lên 6 để đảm bảo độ hội tụ đồng đều.
- **Nguy cơ:** Thiếu sự tương thích khi E tích hợp phân tích do khác biệt môi trường.
  - *Dấu hiệu:* E không đọc được file hoặc xung đột thư viện.
  - *Giải pháp:* File xuất ra là JSON thuần (`results/text_seed*.json`) nên độc lập hoàn toàn với framework và thư viện.
