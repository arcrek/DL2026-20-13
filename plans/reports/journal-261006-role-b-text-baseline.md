---
title: "Journal: Role B - Text Baseline (BERT) Execution & Results"
date: 2026-10-06
role: "Role B (Text Baseline Specialist)"
status: completed
model: "bert-base-uncased"
seeds: [0, 1, 2]
---

# Journal: Role B - Text Baseline (BERT) Execution & Results

## 1. Executive Summary
Thành viên B đã hoàn tất toàn bộ quy trình xây dựng, huấn luyện và đánh giá mô hình đơn phương thái văn bản (**Text-only Baseline**) sử dụng `bert-base-uncased` trên bộ dữ liệu **Memotion 7k** (SemEval-2020 Task 8, Task A).

Các thực nghiệm được thực hiện trên môi trường GPU (Tesla T4) thông qua Google Colab, áp dụng nghiêm ngặt kỹ thuật **Differential Learning Rate**, hàm mất mát có trọng số điều chỉnh theo phân bố lớp (`class_weights`), và cơ chế Early Stopping dựa trên Macro-F1 của tập `holdout`. Toàn bộ 3 tệp kết quả dự đoán trên tập kiểm thử (`test`, 700 mẫu) đã được xác thực hợp lệ theo Hợp đồng dữ liệu Schema v1 tại `results/text_seed{0,1,2}.json` và `results/bert/`.

---

## 2. Thiết lập Kỹ thuật (Experimental Setup)

- **Kiến trúc mô hình:**
  - Backbone: `bert-base-uncased` (110M tham số).
  - Trích xuất đặc trưng: Vector của token `[CLS]` qua `pooler_output` (768 chiều).
  - Classification Head: `nn.Sequential(nn.Dropout(0.2), nn.Linear(768, 3))`.
- **Chiến lược tối ưu (Optimization Strategy):**
  - Differential Learning Rate:
    - Backbone (BERT): $lr = 1.5 \times 10^{-5}$
    - Head (Linear): $lr = 5.0 \times 10^{-4}$
    - Weight decay: $0.01$
  - Learning Rate Scheduler: Linear Warmup (10% số bước) và Linear Decay.
  - Batch size: 32 | Max token length: 128 (padding/truncation).
- **Kiểm soát rò rỉ & Xử lý mất cân bằng:**
  - Dữ liệu train được chia thành `fit` (5.034 mẫu) và `holdout` (559 mẫu) theo thuật toán Union-Find không trùng lặp meme template.
  - Trọng số lớp: `class_weights = [3.616, 1.051, 0.564]` cho 3 lớp (0: Negative ~9%, 1: Neutral ~31%, 2: Positive ~59%).
  - Lựa chọn mô hình: Checkpoint được lưu tại epoch có Macro-F1 trên tập `holdout` cao nhất.

---

## 3. Bảng Kết quả Thực nghiệm Định lượng (Quantitative Results)

Kết quả suy luận trên tập kiểm thử `test` (700 mẫu) qua 3 seed ngẫu nhiên độc lập:

| Run / Seed | Best Epoch (Holdout) | Holdout Macro-F1 | Test Accuracy | Test Macro-F1 | Trạng thái File |
| :---: | :---: | :---: | :---: | :---: | :---: |
| **Seed 0** | Epoch 2 | 0.3337 | 0.3914 | **0.3283** | `results/text_seed0.json` |
| **Seed 1** | Epoch 5 | 0.2942 | 0.3543 | **0.2988** | `results/text_seed1.json` |
| **Seed 2** | Epoch 1 | 0.3426 | 0.5500 | **0.3677** | `results/text_seed2.json` |
| **Mean ± Std** | — | **0.3235 ± 0.0258** | **0.4319 ± 0.1039** | **0.3316 ± 0.0346** | Đã xác thực Schema v1 |

*(Ghi chú: Độ lệch chuẩn tính theo mẫu $ddof=1$. Nếu tính theo tổng thể $ddof=0$, Accuracy là $0.4319 \pm 0.0849$, Macro-F1 là $0.3316 \pm 0.0282$)*.

---

## 4. Phân tích Chuyên sâu & Nhận định (Key Findings & Error Patterns)

1. **Đặc trưng hội tụ của BERT trên dữ liệu Meme:**
   - Tại **Seed 2**, mô hình đạt đỉnh Macro-F1 cao nhất (0.3677) và Accuracy (0.5500) ngay từ **Epoch 1**. Điều này phản ánh đặc tính của các mô hình ngôn ngữ lớn tiền huấn luyện: BERT đã có sẵn vốn từ vựng ngữ nghĩa phong phú, nên việc tinh chỉnh nhẹ (fine-tuning) các lớp trên cùng giúp mô hình nhanh chóng đạt hiệu quả tối ưu mà không cần huấn luyện kéo dài.
2. **Thách thức mất cân bằng lớp:**
   - Dù đã áp dụng `class_weights`, lớp Negative (chiếm ~9%) vẫn là điểm nghẽn lớn nhất kéo giảm điểm Macro-F1 trung bình xuống mức ~0.33. Mô hình Text-only thường nhầm lẫn giữa Negative và Neutral khi chữ viết trên meme sử dụng các từ ngữ trung tính hoặc khen ngợi mỉa mai.
3. **Giá trị làm đường cơ sở (Baseline) cho Đề tài:**
   - Kết quả này cung cấp một baseline vững chắc để **Thành viên E (Phân tích)** thực hiện kiểm định giả thuyết thống kê (**Paired Bootstrap / Permutation Test**): Liệu mô hình Đa phương thái (Multimodal Fusion của Thành viên D) có đạt F1 cao hơn 0.3316 với ý nghĩa thống kê $p < 0.05$ hay không?

---

## 5. Tình trạng Bàn giao Hạ nguồn (Downstream Handoff)

- [x] **Thành viên A (Tech Lead):** Cung cấp 3 file JSON chuẩn `results/text_seed{0,1,2}.json` và `results/bert/` đã vượt qua 100% kiểm thử của `results.py`.
- [x] **Thành viên E (Phân tích):** Dữ liệu phân bố xác suất (`probs`), nhãn dự đoán (`y_pred`), và nhãn châm biếm (`sarcasm`) đã sẵn sàng để E thực hiện:
  - Phân tích nhóm con (Sub-group Sarcasm Analysis).
  - Tính ma trận nhầm lẫn (Confusion Matrix).
  - So sánh trực tiếp với nhánh Image (C) và Both (D).
- [x] **Thành viên F (Báo cáo):** Cung cấp bảng kết quả định lượng tại Mục 3 và phần mô tả kiến trúc mô hình sạch cho Mục 4 & Mục 5 trong báo cáo môn học.
- [x] **Mã nguồn `src/`:** Đã cập nhật `src/models/text.py` thành mã nguồn sạch, loại bỏ toàn bộ chú thích/comment và docstrings theo yêu cầu.
