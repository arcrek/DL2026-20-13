---
title: "Role B: Text Baseline Implementation & Training Plan"
status: pending
priority: P1
effort: "12h"
blockedBy: []
blocks: ["20261001-0830-meme-understanding-modality-study"]
---

# Role B: Text Baseline Implementation & Training Plan

## Executive Summary
Kế hoạch triển khai chi tiết cho **Thành viên B (Text Baseline Specialist)** trong đề tài nghiên cứu *"Multimodal Meme Understanding: Image, Text, or Both?"* trên bộ dữ liệu **Memotion 7k** (SemEval-2020 Task 8, Task A).

Thành viên B chịu trách nhiệm xây dựng, huấn luyện và đánh giá mô hình đơn phương thái văn bản (**Text-only Baseline**) bằng `bert-base-uncased` với kỹ thuật **Differential Learning Rate**, huấn luyện trên 3 seed cố định (0, 1, 2), xử lý mất cân bằng dữ liệu bằng `class_weights`, và xuất kết quả theo hợp đồng dữ liệu chuẩn tại `results/text_seed{0,1,2}.json` để phục vụ trực tiếp cho các kiểm định thống kê của Thành viên E và báo cáo của Thành viên F.

## Context & Constraints
- **Roadmap tham chiếu:** [docs/project-roadmap.md](../../docs/project-roadmap.md) §Phase 2 (Nhánh 1).
- **Phân công nhóm:** [plans/reports/advise-260510-team-split.md](../reports/advise-260510-team-split.md).
- **Cấu hình chung:** [configs/base.yaml](../../configs/base.yaml) (`configs.text: B`, `seeds: [0, 1, 2]`, `class_weighted_loss: true`).
- **Hợp đồng kết quả:** [src/results.py](../../src/results.py) (`save_results()`, schema version 1).
- **Thời gian thực hiện chính:** Giờ 6 - 18 (hoàn thành huấn luyện), Giờ 18 - 34 (hỗ trợ phân tích cùng E), Giờ 34 - 48 (hỗ trợ báo cáo cùng F và tích hợp với A).

## Phase Overview
| Phase | Title | Effort | Priority | Deliverables |
|---|---|---|---|---|
| [Phase 1](phase-01-scaffold-and-data.md) | Module Scaffold & Data Pipeline | 2h | P1 | `src/models/text.py` (Dataset, DataLoader, Collate) |
| [Phase 2](phase-02-model-and-optimizer.md) | Model Architecture & Differential LR | 2h | P1 | `BertMemeClassifier`, AdamW Differential LR, Loss weighting |
| [Phase 3](phase-03-training-and-results.md) | Training Loop & Contract Export (3 Seeds) | 5h | P1 | `results/text_seed{0,1,2}.json`, Checkpoint selection |
| [Phase 4](phase-04-downstream-and-report.md) | Downstream Handoff & Report Contribution | 3h | P2 | Phân tích Sarcasm, Paired Bootstrap test, Mục 4 & 5 báo cáo |

## Success Metrics
- 3 file JSON hợp lệ tại `results/text_seed0.json`, `results/text_seed1.json`, `results/text_seed2.json`.
- `python tests/test_results.py` chạy thành công không có ngoại lệ.
- Bảng tổng hợp thực nghiệm với `mean ± std` Macro-F1 và Accuracy của Text-only sẵn sàng cho E và F.
- Không vi phạm phân tách dữ liệu (không dùng `validation`/`test` để chọn checkpoint, chỉ dùng `holdout`).
