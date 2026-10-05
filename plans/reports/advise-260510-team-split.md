## Yêu cầu đã xác nhận

- **Vấn đề:** 6 người, còn tối đa 48 giờ, mỗi người có GPU Colab/Kaggle riêng. Cần hoàn thành Phase 2 (huấn luyện), Phase 3 (phân tích) và Phase 4 (báo cáo 10-15 trang, notebook chạy được) mà không giẫm chân nhau.
- **Yêu cầu:** (1) Mỗi người có phần việc và file riêng. (2) Phase sau bắt đầu sớm nhờ chốt trước định dạng `results/*.json`. (3) Có mốc kiểm tra và một người tích hợp.
- **Mục tiêu:** Nộp PDF và mã nguồn đúng hạn, notebook chạy từ đầu đến cuối.
- **Ngoài phạm vi:** Thí nghiệm ngoài roadmap; chia theo năng lực cá nhân (chưa biết trình độ nên chia theo vai trò).
- **Ràng buộc:** Mọi mô hình dùng cùng split và seed 0, 1, 2.

## Nhận định

Roadmap có thể chia song song, nhưng không nên chia theo phase vì Phase 3 và 4 sẽ phải chờ Phase 2. Điều kiện để song song là chốt hợp đồng dữ liệu trong giờ đầu: `results/*.json` gồm `y_true`, `y_pred`, `probs`, `seed`, `sarcasm` và id mẫu.

## Phân vai

| # | Vai | Việc chính | File sở hữu | Giờ |
|---|---|---|---|---|
| A | Tech lead / tích hợp | Chốt schema, review PR, hợp nhất, kiểm tra Colab sạch | khung `src/finetune.py`, README | 0-48 |
| B | Text baseline | BERT, differential LR, 3 seed | `src/models/text.py`, `results/text_*.json` | 6-18 |
| C | Image baseline | ResNet50 hai giai đoạn, 3 seed | `src/models/image.py`, `results/image_*.json` | 6-18 |
| D | Fusion | Concat, Cross-Attn, Product fusion, 3 seed | `src/models/fusion.py`, `results/both_*.json` | 6-22 |
| E | Phân tích | Bootstrap/permutation test, sub-group sarcasm, confusion matrix, biểu đồ | `src/analysis.py`, `report/figures/` | 12-34 |
| F | Báo cáo + lỗi định tính | 20 ca lỗi, viết mục 1-4 và 7 | `report/` | 6-48 |

Ghi chú:
- D nặng nhất (3 cấu hình fusion). A nhận thêm việc của D khi rảnh.
- B và C xong sớm (khoảng giờ 18), sau đó B hỗ trợ E, C hỗ trợ F (mục 5, 6, hình).
- E làm trước trên dữ liệu giả ở giờ 12-18, rồi chạy lại trên kết quả thật.

## Không nên làm

- Chia mỗi người một phase.
- Để nhiều người cùng sửa `src/finetune.py`. Tách mỗi mô hình thành một file.
- Để mỗi người tự chọn split hoặc seed, vì làm sai tính công bằng của so sánh.
- Dồn viết báo cáo vào giờ 40.
- Thêm thí nghiệm ngoài roadmap.

## Cách tiết kiệm công sức

1. Một config chung (`configs/*.yaml`) với cùng split và seed.
2. Một hàm `save_results()` duy nhất, do A làm.
3. Họp 15 phút ở giờ 6, 18, 34 và 44.

## Lộ trình

- **Giờ 0-6:** A chốt schema và khung code. B, C, D cài môi trường và chạy thử một epoch. E viết khung phân tích trên dữ liệu giả. F dựng khung báo cáo và viết mục 3.
- **Giờ 6-18:** B, C, D huấn luyện (3 seed song song). F viết mục 1, 2, 4.
- **Giờ 18-22:** D hoàn tất 3A, 3B, Product. Gộp kết quả. A kiểm tra đủ file.
- **Giờ 22-34:** E chạy phân tích thật. F lấy 20 ca lỗi. B và C hỗ trợ.
- **Giờ 34-44:** F hoàn thiện mục 5, 6, 7. A hoàn thiện notebook và README, chạy thử Colab sạch.
- **Giờ 44-48:** Đọc lại, sửa lỗi, xuất PDF và nộp.

## Lợi ích

- Phase 3 và 4 bắt đầu sớm, không dồn việc cuối.
- Mỗi người sở hữu file riêng, ít xung đột.
- A là điểm tích hợp duy nhất, giảm rủi ro lệch kết quả.

## Đánh đổi

- A là điểm nghẽn, cần người chắc tay.
- Tải không đều: D nặng, B và C nhẹ hơn, cần chuyển việc sau giờ 18.
- Nếu một mô hình hỏng ở giờ 18, nên bỏ Product fusion thay vì kéo dài thời hạn.
- Nếu có người chậm hoặc thiếu GPU, dồn việc về A và D.

## Danh sách việc

- [ ] A: chốt schema `results/*.json` và hàm `save_results()`
- [ ] A: tạo config chung (split, seed 0/1/2)
- [ ] A: tạo nhánh theo vai
- [ ] B: huấn luyện BERT, 3 seed
- [x] C: huấn luyện ResNet50 hai giai đoạn, 3 seed
- [x] D: huấn luyện Concat, Cross-Attn, Product fusion, 3 seed
- [x] E: viết `src/analysis.py`, chạy thử trên dữ liệu giả
- [ ] E: bootstrap/permutation test, phân tích sarcasm, confusion matrix, biểu đồ
- [ ] F: soạn khung báo cáo và mục 1-4, 7
- [ ] F: chọn và phân loại 20 ca lỗi
- [ ] A: hoàn thiện notebook và README, chạy thử Colab sạch
- [ ] Cả nhóm: đọc lại, xuất PDF và nộp

## Chỉ tiêu thành công

- 12 file JSON hợp lệ (4 cấu hình × 3 seed) trong `results/` trước giờ 22.
- Có bảng `mean ± std` cho Text, Image và Both.
- `pytest tests/` chạy không lỗi.
- Có p-value cho Both so với Text-only.
- Ít nhất 20 ca lỗi được phân loại.
- Notebook chạy đủ trên Colab sạch.
- Báo cáo 10-15 trang, nộp trước hạn ít nhất 2 giờ.
