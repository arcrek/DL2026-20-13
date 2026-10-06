# Concat & Product Multimodal Fusion Refactor

Tài liệu kỹ thuật mô tả cấu trúc, công thức toán học, điểm giao thoa (fusion seam), thứ tự khởi tạo/nhóm tham số và hành vi freeze/BatchNorm sau đợt refactor `ConcatFusionModel` và `ProductFusionModel`.

## 1. Tổng quan kiến trúc & Lớp cơ sở `_PooledFusionBase`

Trước đợt refactor, cả hai lớp `ConcatFusionModel` và `ProductFusionModel` đều lặp lại logic:
- Khởi tạo `self.text_enc` (BERT) và `self.img_enc` (ResNet-50).
- Xử lý freeze văn bản (`freeze_text=True`).
- Trích xuất đặc trưng pooled: lấy token `[CLS]` từ text và pooling GAP từ image.
- Quản lý tách nhóm tham số backbone (`backbone_params`) với requires_grad check.

Sau refactor, một private base class `_PooledFusionBase(nn.Module)` được tạo ra trong `src/models/fusion.py` để dùng chung giữa hai cấu hình:
- Quản lý khởi tạo text encoder và image encoder theo thứ tự nghiêm ngặt (text trước, image sau) nhằm bảo toàn trạng thái RNG của PyTorch.
- Phương thức `_encode_modalities(pixel_values, input_ids, attention_mask)` trả về tuple `(v_text, v_img)`.
- Phương thức `backbone_params()` duyệt qua các tham số của `text_enc` và `img_enc` (chỉ lấy các tham số có `requires_grad=True`).
- Phương thức `head_params()` duyệt qua các module theo danh sách tên string `_head_module_names` được định nghĩa trên subclass, tránh tạo registered module wrapper/alias mới trên `self`.
- Không tác động hay bao bọc `CrossAttentionFusionModel` để giữ nguyên tính độc lập.

## 2. Công thức toán học & Điểm giao thoa `fuse_features`

Điểm giao thoa `fuse_features(v_text, v_img)` được tách riêng trên mỗi subclass, nhận hai tensor đặc trưng đã pooled và trả về tensor đặc trưng kết hợp trước khi đưa vào MLP classification head:

### 2.1. Late Concatenation Fusion (`ConcatFusionModel`)
- **Đầu vào**:
  - `v_text`: vector `[CLS]` từ BERT, kích thước `(Batch, hidden_size)` (mặc định 768).
  - `v_img`: vector GAP từ ResNet-50, kích thước `(Batch, 2048)`.
- **Phép toán seam**:
  $$\mathbf{z}_{\text{fused}} = [\mathbf{v}_{\text{text}} \,;\, \mathbf{v}_{\text{img}}] \in \mathbb{R}^{B \times (\text{hidden\_size} + 2048)}$$
  (mặc định `(Batch, 2816)`).
- **Classification Head**:
  $$\text{Dropout}(p) \to \text{Linear}(2816, 512) \to \text{ReLU} \to \text{Dropout}(p) \to \text{Linear}(512, \text{num\_classes})$$

### 2.2. Product Fusion (`ProductFusionModel`)
- **Đầu vào**:
  - `v_text`: vector `[CLS]` từ BERT, kích thước `(Batch, hidden_size)`.
  - `v_img`: vector GAP từ ResNet-50, kích thước `(Batch, 2048)`.
- **Tầng chiếu (Projections)**:
  - `proj_text`: $\text{Linear}(\text{hidden\_size}, \text{proj\_dim})$ (mặc định 512).
  - `proj_img`: $\text{Linear}(2048, \text{proj\_dim})$ (mặc định 512).
- **Phép toán seam**:
  $$\mathbf{z}_{\text{fused}} = \text{LayerNorm}\left( \text{proj}_{\text{text}}(\mathbf{v}_{\text{text}}) \odot \text{proj}_{\text{img}}(\mathbf{v}_{\text{img}}) \right) \in \mathbb{R}^{B \times \text{proj\_dim}}$$
- **Classification Head**:
  $$\text{Dropout}(p) \to \text{Linear}(\text{proj\_dim}, 256) \to \text{ReLU} \to \text{Dropout}(p) \to \text{Linear}(256, \text{num\_classes})$$

## 3. Thứ tự khởi tạo, State Dict & Nhóm Optimizer

- **Thứ tự khởi tạo (Initialization Order & RNG)**:
  1. `text_enc`: Khởi tạo `AutoModel.from_pretrained(text_model_name)`
  2. `img_enc`: Khởi tạo `ResNet50Backbone(pretrained=True, freeze=freeze_image)`
  3. Freeze text: Cập nhật `requires_grad = False` nếu `freeze_text=True`
  4. Các module riêng của subclass:
     - `ConcatFusionModel`: khởi tạo `self.head`.
     - `ProductFusionModel`: khởi tạo `self.proj_text`, `self.proj_img`, `self.norm`, rồi `self.head`.
  Việc giữ nguyên thứ tự này đảm bảo cùng một `torch.manual_seed(S)` sinh ra chính xác cùng bộ trọng số khởi tạo ban đầu và trạng thái RNG cuối cùng.
- **State Dict Keys**:
  Không thêm bất kỳ tiền tố mới nào. Các keys nguyên bản `text_enc.*`, `img_enc.*`, `head.*` (và `proj_text.*`, `proj_img.*`, `norm.*` đối với Product) hoàn toàn trùng khớp 100%, cho phép load song phương strict (`strict=True`).
- **Phân hoạch Optimizer**:
  - `backbone_params`: Tham số trainable của `text_enc` theo sau bởi `img_enc`.
  - `head_params`:
    - Concat: `head` parameters.
    - Product: `proj_text` $\to$ `proj_img` $\to$ `norm` $\to$ `head` parameters.
  - Hai nhóm tham số luôn thỏa mãn tính chất rời nhau (`disjoint`), không lặp phần tử, và hợp lại bao phủ toàn bộ tham số trainable (`exhaustive`).

## 4. Ngữ nghĩa Freeze & BatchNorm

- Cờ `freeze_image` và `freeze_text` chỉ thao tác trên `requires_grad = False` của các parameters.
- Mô hình **không** tự ý gọi `.eval()` hoặc override phương thức `train()` trên các encoder bị đóng băng.
- Khi ở chế độ `train()`, các bộ đệm thống kê (`running_mean`, `running_var`, `num_batches_tracked`) của BatchNorm trong `img_enc` vẫn được cập nhật theo đúng chuẩn huấn luyện PyTorch; trọng số mạng không bị thay đổi và gradient tương ứng là `None`.

## 5. Xác minh & Kiểm thử (Verification & Test Commands)

Tất cả các kiểm thử contract được chạy độc lập offline bằng CPU fixtures, không tải trọng số từ Internet:

```bash
# Chạy bộ test fusion và kết quả:
py -3.12 -m pytest tests/test_fusion.py tests/test_results.py -q
# Kết quả: 50 passed, 1 warning (requests dependency warning từ môi trường)

# Biên dịch kiểm tra cú pháp:
py -3.12 -m py_compile src/models/fusion.py tests/test_fusion.py
```

### Các kiểm thử bao phủ:
1. `test_factory_defaults_kwargs_and_shapes`: Kiểm tra `build_fusion_model` cho cả 3 cấu hình.
2. `test_public_api_and_exports`: Kiểm tra signature và exports của runner và module.
3. `test_same_seed_state_rng_strict_loading_and_legacy_logits`: So sánh trực tiếp với mô hình reference độc lập `LegacyConcatFusionModel` và `LegacyProductFusionModel` (load state_dict 2 chiều strict, eval logits trùng khớp tuyệt đối).
4. `test_explicit_math_with_zero_negative_features`: Kiểm tra công thức toán với giá trị âm/zero.
5. `test_fuse_features_exact_math_and_gradients`: Kiểm tra trực tiếp hàm `fuse_features` và gradient lan truyền qua cả 2 modality.
6. `test_freeze_gradients_groups_and_adamw`: Kiểm tra 4 tổ hợp freeze, cập nhật optimizer AdamW.
7. `test_feature_gradients_even_with_frozen_encoders`: Kiểm tra gradient của feature tensor.
8. `test_frozen_image_bn_buffers_remain_mutable_in_train`: Kiểm tra BN buffers trong train mode.
