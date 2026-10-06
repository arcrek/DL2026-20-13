# Project Roadmap: Multimodal Meme Understanding

Tài liệu này xác định lộ trình nghiên cứu, triển khai và đánh giá đề tài **"Multimodal Meme Understanding: Image, Text, or Both?"** trên bộ dữ liệu **Memotion 7k** (SemEval-2020 Task 8, Task A).

> **Mục tiêu cốt lõi:** Trả lời bằng thực nghiệm khoa học xem việc kết hợp đa phương thái (Multimodal Fusion) có thực sự vượt trội hơn các mô hình đơn phương thái (Text-only hoặc Image-only) hay không, và trong những ngữ cảnh nào (như châm biếm / sarcasm) thì đa phương thái mang lại giá trị lớn nhất.

---

## 1. Trạng thái tổng quan (Milestone Overview)

| Giai đoạn | Nội dung chính | Trọng tâm kỹ thuật | Trạng thái | Executable Owner |
| :--- | :--- | :--- | :---: | :--- |
| **Phase 1** | Chuẩn bị & Tiền xử lý dữ liệu | Lọc template leakage, dùng `text_corrected`, chia hold-out | ✅ Hoàn thành | [`src/data.py`](file:///home/arcrek/workspace/dl2026/src/data.py), [`DATA.md`](file:///home/arcrek/workspace/dl2026/DATA.md) |
| **Phase 2** | Triển khai 3 mô hình đối chứng | Text-only (BERT), Image-only (ResNet50), Both (Concat & Cross-Attn) | 🔄 Đang triển khai | [`src/models/`](file:///home/arcrek/workspace/dl2026/src/models/) |
| **Phase 3** | Đánh giá, Ablation & Error Analysis | Phân tích Sarcasm, Paired t-test, 20 ca lỗi định tính | ⏳ Chờ Phase 2 | [`plans/20261001-0830-meme-understanding-modality-study/phase-03-analysis.md`](file:///home/arcrek/workspace/dl2026/plans/20261001-0830-meme-understanding-modality-study/phase-03-analysis.md) |
| **Phase 4** | Báo cáo, Notebook Colab & Nghiệm thu | Báo cáo 10-15 trang chuẩn rubric, demo inference | ⏳ Chờ Phase 3 | [`plans/20261001-0830-meme-understanding-modality-study/phase-04-deliverables.md`](file:///home/arcrek/workspace/dl2026/plans/20261001-0830-meme-understanding-modality-study/phase-04-deliverables.md) |

---

## 2. Lộ trình chi tiết từng giai đoạn

### Giai đoạn 1: Chuẩn hóa dữ liệu & Kiểm soát rò rỉ (Hours 0 - 6)
* **Mục tiêu:** Xây dựng tập dữ liệu sạch, ngăn chặn triệt để hiện tượng rò rỉ meme template.
* **Nhiệm vụ cụ thể:**
  - [x] Tải và giải nén Memotion 7k từ nguồn chuẩn SemEval-2020 (`Ahren09/MMSoc_Memotion`).
  - [x] Khắc phục lỗi của notebook mẫu: Chuyển toàn bộ pipeline trích xuất văn bản từ `text_ocr` sang `text_corrected`.
  - [x] Áp dụng thuật toán Union-Find (MD5 ảnh + normalized text) trong [`src/data.py`](file:///home/arcrek/workspace/dl2026/src/data.py) để phân nhóm không để lọt template trùng giữa tập fit và tập hold-out.
  - [x] Tính toán trọng số phân bố lớp (`class_weights`) cho 3 nhãn: Negative (~9%), Neutral (~31%), Positive (~59%), theo phân bố thực tế của train (518/1762/3313).
* **Tiêu chí hoàn thành (Exit Criteria):**
  - [x] Chạy `pytest tests/test_data.py` vượt qua toàn bộ kiểm thử.
  - [x] Sinh file phân tách `features/train_holdout.json` hợp lệ.

---

### Giai đoạn 2: Xây dựng & Huấn luyện 3 Mô hình Đối chứng (Hours 6 - 18)
* **Mục tiêu:** Huấn luyện công bằng 3 cấu hình trên cùng một tập dữ liệu phân tách và cùng các seed thực nghiệm (Seed 0, 1, 2).
* **Nhiệm vụ cụ thể:**
  1. **Nhánh 1 - Text Only Baseline:**
     - Mô hình: `bert-base-uncased` $\to$ Vector `[CLS]` (768d) $\to$ Dropout $\to$ Linear(3 classes).
     - Huấn luyện với Differential LR (`1.5e-5` cho BERT, `5e-4` cho classification head).
  2. **Nhánh 2 - Image Only Baseline:**
     - Mô hình: `ResNet50` (hoặc VGG16) $\to$ Global Average Pooling (2048d) $\to$ Dropout $\to$ Linear(3 classes).
     - Huấn luyện transfer learning 2 giai đoạn (freeze backbone rồi unfreeze fine-tune).
  3. **Nhánh 3 - Both (Multimodal Fusion):**
     - **Cấu hình 3A (Late Concat Fusion):** Ghép vector `[v_text, v_img]` $\to$ MLP Classifier.
     - **Cấu hình 3B (Cross-Attention Fusion):** Query = Text tokens ($L \times 256$), Key/Value = Image feature map ($49 \times 256$) $\to$ Residual LayerNorm $\to$ MLP Classifier.
* **Tiêu chí hoàn thành (Exit Criteria):**
  - Lưu kết quả dự đoán và chỉ số (Accuracy, Macro-F1) của 3 seed vào `results/*.json`.
  - Có bảng tổng hợp `mean ± std` cho cả 3 mục tiêu: Text, Image, Both.

---

### Giai đoạn 3: Phân tích Thực nghiệm, Ablation & Error Analysis (Hours 18 - 34)
* **Mục tiêu:** Đào sâu lý giải *tại sao* và *khi nào* đa phương thái vượt trội.
* **Nhiệm vụ cụ thể:**
  - [ ] **Kiểm định ý nghĩa thống kê:** Chạy Paired Bootstrap Test / Permutation Test giữa `Both` vs `Text-only` để khẳng định mức tăng điểm F1 có ý nghĩa thống kê ($p < 0.05$).
  - [ ] **Phân tích nhóm con (Sub-group Analysis):** Đánh giá riêng hiệu năng trên:
    - Nhóm meme có tính châm biếm (`sarcasm` = true).
    - Nhóm meme nghĩa đen / không châm biếm (`sarcasm` = false).
    - Nhóm meme có text ngắn / phụ thuộc biểu cảm ảnh.
  - [ ] **Ablation Study:** So sánh Concat Fusion vs Product Fusion vs Cross-Attention Fusion.
  - [ ] **Phân tích lỗi định tính (Qualitative Error Analysis):**
    - Trích xuất ít nhất 20 trường hợp dự đoán sai tiêu biểu.
    - Phân loại nguyên nhân lỗi: *Ẩn dụ thị giác phức tạp*, *Nhiễu OCR*, *Ngữ cảnh văn hóa internet đặc thù*, hoặc *Ảnh làm nhiễu chữ*.
  - [ ] **Trực quan hóa:** Sinh ma trận nhầm lẫn (Confusion Matrix) và biểu đồ so sánh giữa 3 mục tiêu.
* **Tiêu chí hoàn thành (Exit Criteria):**
  - Toàn bộ bảng biểu, đồ thị xuất ra định dạng ấn bản tại `report/figures/`.

---

### Giai đoạn 4: Hoàn thiện Deliverables & Báo cáo (Hours 34 - 48)
* **Mục tiêu:** Đóng gói mã nguồn, notebook chạy được từ đầu đến cuối trên Google Colab / Kaggle, và hoàn thành báo cáo môn học.
* **Nhiệm vụ cụ thể:**
  - [ ] Hoàn thiện [`notebooks/meme_understanding.ipynb`](file:///home/arcrek/workspace/dl2026/notebooks/meme_understanding.ipynb) đảm bảo chạy từ đầu đến cuối không lỗi trên môi trường Colab/Kaggle sạch.
  - [ ] Cập nhật [`README.md`](file:///home/arcrek/workspace/dl2026/README.md) hướng dẫn cài đặt, tái lập kết quả chỉ với 1 dòng lệnh.
  - [ ] Soạn thảo báo cáo học phần (10 - 15 trang) bám sát rubric:
    1. Giới thiệu & Câu hỏi nghiên cứu (Text, Image, hay Both?).
    2. Tổng quan nghiên cứu & Cơ sở lý thuyết (CNN, Transformers, Attention theo slide môn học).
    3. Chuẩn bị dữ liệu & Kiểm soát rò rỉ template.
    4. Thiết kế mô hình & Chiến lược huấn luyện 2 giai đoạn.
    5. Kết quả thực nghiệm định lượng (Bảng so sánh 3 mục tiêu).
    6. Phân tích định tính & Phân loại ca lỗi (20+ ví dụ).
    7. Kết luận & Hướng phát triển.
* **Tiêu chí hoàn thành (Exit Criteria):**
  - Báo cáo hoàn chỉnh định dạng PDF và nộp mã nguồn trước thời hạn.

---

## 3. Bản đồ Điều hướng & Quyền quản lý (Navigation & Ownership)

* **Quy chuẩn dữ liệu & Phân tách:** Quản lý tại [`DATA.md`](file:///home/arcrek/workspace/dl2026/DATA.md) và thực thi bởi [`src/data.py`](file:///home/arcrek/workspace/dl2026/src/data.py).
* **Huấn luyện mô hình:** Thực thi tại [`src/models/`](file:///home/arcrek/workspace/dl2026/src/models/) và các runner (như [`src/text.py`](file:///home/arcrek/workspace/dl2026/src/text.py), [`src/image.py`](file:///home/arcrek/workspace/dl2026/src/image.py)).
* **Kế hoạch gốc:** [`plans/20261001-0830-meme-understanding-modality-study/plan.md`](file:///home/arcrek/workspace/dl2026/plans/20261001-0830-meme-understanding-modality-study/plan.md).
* **Notebook tương tác:** [`notebooks/meme_understanding.ipynb`](file:///home/arcrek/workspace/dl2026/notebooks/meme_understanding.ipynb).
