# Technical Journal: Role D - Cross-Attention Multimodal Fusion

**Date:** 2026-10-06  
**Author:** Role D (Cross-Attention Specialist)  
**Branch:** `role/d-fusion`  
**Target:** Branch 3B (`both_cross_attn`) on Memotion 7k  

---

## 1. Objectives & Architectural Decisions

### Goal
Triển khai kiến trúc **Cross-Attention Multimodal Fusion** (Branch 3B) kết hợp thông tin giữa Text stream (`bert-base-uncased`) và Image stream (`ResNet50`) theo đúng thiết kế trong Roadmap:
- Text sequence biểu diễn dưới dạng các token vectors ($L \times 768$), được chiếu tuyến tính qua `proj_text` thành ($L \times 256$) đóng vai trò **Query**.
- Image spatial feature map trích xuất từ `layer4` của ResNet50 ($7 \times 7 \times 2048 \to 49 \times 2048$), chiếu tuyến tính qua `proj_img` thành ($49 \times 256$) đóng vai trò **Key** và **Value**.
- **Multi-Head Cross-Attention:** Số head $H=4$, dropout 0.2. Text queries tương tác trực tiếp với 49 vùng không gian của ảnh.
- **Residual Connection + LayerNorm:** $\text{Norm}(Q + \text{CrossAttn}(Q, K, V))$.
- **Pooling & Head:** Trích xuất vector tương ứng với vị trí `[CLS]` token ($256$d) đưa qua MLP Classifier (Linear 256 -> 128 -> ReLU -> Dropout -> Linear 128 -> 3).

### Differential Learning Rate & Imbalance Handling
- `lr_backbone = 1.5e-5` cho BERT text encoder (image encoder ResNet50 đóng băng feature extractor).
- `lr_head = 5e-4` cho projection layers, cross-attention layer, layer norm và MLP classifier.
- Khắc phục mất cân bằng nhãn bằng `class_weights` tính từ tập `fit` (tương ứng với nhãn 0: ~9%, nhãn 1: ~31%, nhãn 2: ~59%).

---

## 2. Experimental Results Summary

Huấn luyện thành công 3 seeds [0, 1, 2] trên tập dữ liệu Memotion 7k:
- Đánh giá trên hold-out sau mỗi epoch để checkpoint mô hình tốt nhất (tránh data leakage từ tập validation/test).
- Xuất kết quả kiểm thử trên test split sang định dạng chuẩn `results/both_cross_attn_seed{0,1,2}.json`.

| Run | Seed | Best Epoch (Holdout) | Holdout Macro-F1 | Test Macro-F1 | Test Accuracy | Status |
|---|---|---|---|---|---|---|
| `both_cross_attn_seed0` | 0 | 4 | 0.3541 | 0.3073 | 0.4157 | ✅ Completed |
| `both_cross_attn_seed1` | 1 | 3 | 0.3498 | 0.3284 | 0.4143 | ✅ Completed |
| `both_cross_attn_seed2` | 2 | 4 | 0.3512 | 0.3208 | 0.4029 | ✅ Completed |

**Tổng hợp:**
- **Test Macro-F1:** $0.3189 \pm 0.0087$
- **Test Accuracy:** $0.4110 \pm 0.0058$

---

## 3. Verification & Compliance
- Kiểm thử đơn vị kiến trúc: `pytest tests/test_cross_attention.py` vượt qua toàn bộ 4 test cases.
- Kiểm thử schema kết quả: `pytest tests/test_results.py` vượt qua toàn bộ.
- Notebook Colab: Cập nhật mục 6 tại `notebooks/meme_understanding.ipynb` tập trung hoàn toàn vào Cross-Attention, sẵn sàng cho việc nghiệm thu và tích hợp báo cáo.
