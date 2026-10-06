"""Multimodal Fusion architectures and training pipeline for Memotion 7k.

This module implements the three multimodal fusion architectures required for Role D:
1. ConcatFusionModel (both_concat): Late concatenation of text [CLS] and image GAP.
2. CrossAttentionFusionModel (both_cross_attn): Cross-attention where text queries
   attend to 49 spatial image feature vectors with residual LayerNorm.
3. ProductFusionModel (both_product): Element-wise product of projected text and
   image representations (ablation baseline).
"""
import argparse
import json
import multiprocessing
import os
import random
import sys

try:
    if hasattr(multiprocessing, "set_start_method"):
        multiprocessing.set_start_method("fork", force=True)
except Exception:
    pass

import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models
from PIL import Image, ImageFile
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.models import ResNet50_Weights
from transformers import AutoModel, AutoTokenizer
import transformers

transformers.logging.set_verbosity_error()
ImageFile.LOAD_TRUNCATED_IMAGES = True

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
SRC_DIR = os.path.join(ROOT_DIR, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from data import DATA_DIR, FEATURE_DIR, load_jsonl
from results import RESULTS_DIR, save_results

NUM_CLASSES = 3
TEXT_MODEL_NAME = "bert-base-uncased"
FUSION_CONFIGS = ["both_concat", "both_cross_attn", "both_product"]


try:
    from preprocessing import LABEL2ID, is_valid_image, normalize_sentiment
except ImportError:
    LABEL2ID = {"negative": 0, "neutral": 1, "positive": 2}
    is_valid_image = None
    normalize_sentiment = None


def safe_open_image(path):
    """Safely open and convert an image to RGB, returning fallback if corrupt."""
    try:
        if path and os.path.exists(path):
            with Image.open(path) as img:
                img.verify()
            with Image.open(path) as img:
                return img.convert("RGB")
    except Exception:
        pass
    return Image.new("RGB", (224, 224), (128, 128, 128))


class MultimodalMemeDataset(Dataset):
    """Dataset providing text, image path, label, id, and sarcasm attribute."""

    def __init__(self, rows, data_dir=DATA_DIR):
        self.rows = rows
        self.data_dir = data_dir

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        r = self.rows[idx]
        # Resolve image path supporting raw JSONL ('img') and preprocessed DataFrame ('image_path')
        if "image_path" in r and r["image_path"]:
            img_path = str(r["image_path"])
        else:
            img_rel = str(r.get("img", ""))
            img_path = (
                os.path.join(self.data_dir, img_rel)
                if self.data_dir and not os.path.isabs(img_rel)
                else img_rel
            )

        # Support integer label, string label_name, or normalize raw label
        if "label" in r and r["label"] is not None:
            label = int(r["label"])
        elif "label_name" in r and r["label_name"] in LABEL2ID:
            label = LABEL2ID[r["label_name"]]
        elif "raw_sentiment" in r and normalize_sentiment:
            norm = normalize_sentiment(r["raw_sentiment"])
            label = LABEL2ID.get(norm, 1)
        else:
            label = 1

        text = str(r.get("text", r.get("text_corrected", "")))
        sample_id = str(r.get("id", idx))
        sarcasm = str(r.get("sarcasm", "not_sarcastic"))

        return {
            "id": sample_id,
            "img_path": img_path,
            "text": text,
            "label": label,
            "sarcasm": sarcasm,
        }


class FusionCollate:
    """Picklable collate callable for Multimodal Fusion."""

    def __init__(self, tokenizer, image_transform, max_length=128):
        self.tokenizer = tokenizer
        self.image_transform = image_transform
        self.max_length = max_length

    def __call__(self, batch):
        ids = [item["id"] for item in batch]
        texts = [item["text"] for item in batch]
        labels = [item["label"] for item in batch]
        sarcasms = [item["sarcasm"] for item in batch]

        imgs = [self.image_transform(safe_open_image(item["img_path"])) for item in batch]
        pixel_values = torch.stack(imgs, dim=0)

        enc = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )

        return {
            "ids": ids,
            "pixel_values": pixel_values,
            "input_ids": enc["input_ids"],
            "attention_mask": enc["attention_mask"],
            "labels": torch.tensor(labels, dtype=torch.long),
            "sarcasms": sarcasms,
        }


def make_fusion_collate_fn(tokenizer, image_transform, max_length=128):
    """Create collate function for batching multimodal samples."""
    return FusionCollate(tokenizer, image_transform, max_length=max_length)


class ResNet50Backbone(nn.Module):
    """ResNet50 feature extractor providing pooled features and spatial feature maps."""

    def __init__(self, pretrained=True, freeze=True):
        super().__init__()
        weights = ResNet50_Weights.DEFAULT if pretrained else None
        base = models.resnet50(weights=weights)

        # Truncate after layer4 to keep spatial feature map (B, 2048, 7, 7)
        self.conv1 = base.conv1
        self.bn1 = base.bn1
        self.relu = base.relu
        self.maxpool = base.maxpool
        self.layer1 = base.layer1
        self.layer2 = base.layer2
        self.layer3 = base.layer3
        self.layer4 = base.layer4
        self.avgpool = base.avgpool

        if freeze:
            for p in self.parameters():
                p.requires_grad = False

    def forward(self, x):
        """
        Returns:
            pooled: (B, 2048) global average pooled representation
            spatial: (B, 49, 2048) flattened spatial grid features (7x7=49)
        """
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        feat_map = self.layer4(x)  # (B, 2048, 7, 7)

        # Pooled vector
        pooled = self.avgpool(feat_map)
        pooled = torch.flatten(pooled, 1)  # (B, 2048)

        # Spatial tokens: (B, 2048, 7, 7) -> (B, 2048, 49) -> (B, 49, 2048)
        B, C, H, W = feat_map.shape
        spatial = feat_map.view(B, C, H * W).permute(0, 2, 1)  # (B, 49, 2048)

        return pooled, spatial


class _PooledFusionBase(nn.Module):
    """Lớp cơ sở dùng chung cho các mô hình kết hợp trích xuất đặc trưng pooled.

    Quản lý việc khởi tạo các encoder (BERT text + ResNet image), freeze trọng số,
    trích xuất đặc trưng pooled (CLS và GAP), và phân tách các nhóm tham số
    backbone / head phục vụ tối ưu hóa với learning rate phân tầng.
    """

    _head_module_names = ()

    def __init__(
        self,
        text_model_name=TEXT_MODEL_NAME,
        freeze_image=True,
        freeze_text=False,
    ):
        super().__init__()
        # Backbone văn bản: BERT Base Uncased
        self.text_enc = AutoModel.from_pretrained(text_model_name)
        # Backbone hình ảnh: ResNet50
        self.img_enc = ResNet50Backbone(pretrained=True, freeze=freeze_image)

        if freeze_text:
            for p in self.text_enc.parameters():
                p.requires_grad = False

    def _encode_modalities(self, pixel_values, input_ids, attention_mask):
        """Trích xuất đặc trưng pooled từ hai modality: CLS text và GAP image."""
        text_out = self.text_enc(input_ids=input_ids, attention_mask=attention_mask)
        v_text = text_out.last_hidden_state[:, 0, :]  # (Batch, hidden_size)
        v_img, _ = self.img_enc(pixel_values)         # (Batch, 2048)
        return v_text, v_img

    def backbone_params(self):
        """Lấy danh sách tham số của backbone (BERT + ResNet) để gán learning rate nhỏ."""
        params = []
        for p in self.text_enc.parameters():
            if p.requires_grad:
                params.append(p)
        for p in self.img_enc.parameters():
            if p.requires_grad:
                params.append(p)
        return params

    def head_params(self):
        """Lấy danh sách tham số của head phân loại và các tầng chiếu riêng."""
        params = []
        for name in self._head_module_names:
            module = getattr(self, name)
            for p in module.parameters():
                if p.requires_grad:
                    params.append(p)
        return params


# =============================================================================
# CẤU HÌNH 3A: LATE CONCATENATION FUSION (Mô hình kết hợp ghép nối)
# =============================================================================
class ConcatFusionModel(_PooledFusionBase):
    """Nhánh 3A: Late Concat Fusion.

    Nguyên lý:
    - Text: BERT trích xuất vector [CLS] đại diện toàn câu -> (Batch, 768).
    - Image: ResNet50 trích xuất vector GAP -> (Batch, 2048).
    - Fusion: Nối trực tiếp 2 vector lại với nhau: [v_text, v_img] -> Kích thước: 768 + 2048 = 2816.
    - Phân loại: Đi qua mạng MLP 2 tầng (Linear 2816->512 -> ReLU -> Dropout -> Linear 512->3).
    """

    _head_module_names = ("head",)

    def __init__(
        self,
        num_classes=NUM_CLASSES,
        text_model_name=TEXT_MODEL_NAME,
        dropout=0.2,
        freeze_image=True,
        freeze_text=False,
    ):
        super().__init__(
            text_model_name=text_model_name,
            freeze_image=freeze_image,
            freeze_text=freeze_text,
        )

        # Tổng số chiều sau khi ghép nối: 768 (BERT) + 2048 (ResNet) = 2816
        in_dim = self.text_enc.config.hidden_size + 2048
        # Đầu phân loại MLP (Classification Head)
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(in_dim, 512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, num_classes),
        )

    def fuse_features(self, v_text, v_img):
        """Ghép nối vector Text và Image dọc theo chiều đặc trưng."""
        return torch.cat([v_text, v_img], dim=-1)

    def forward(self, pixel_values, input_ids, attention_mask):
        # 1 & 2. Trích xuất đặc trưng text (CLS) và ảnh (GAP)
        v_text, v_img = self._encode_modalities(pixel_values, input_ids, attention_mask)

        # 3. Kết hợp bằng cách ghép nối (Concatenation) dọc theo chiều features (dim=-1)
        fused = self.fuse_features(v_text, v_img)  # (Batch, 2816)

        # 4. Dự đoán điểm logits của 3 nhãn cảm xúc qua MLP
        logits = self.head(fused)
        return logits


# =============================================================================
# CẤU HÌNH ĐỐI CHỨNG: PRODUCT FUSION (Phép nhân Hadamard từng phần tử)
# =============================================================================
class ProductFusionModel(_PooledFusionBase):
    """Mô hình Ablation: Product Fusion.

    Nguyên lý:
    - Thay vì ghép nối, mô hình chiếu cả vector Text (768d) và vector Image (2048d)
      về cùng một không gian đặc trưng có kích thước `proj_dim = 512`.
    - Sau đó thực hiện nhân từng phần tử (Element-wise / Hadamard Product):
      fused = LayerNorm(proj_text(v_text) * proj_img(v_img)).
    - Kiểm tra giả thuyết: Liệu phép nhân tương quan có mang lại hiệu quả tốt hơn ghép nối thuần túy?
    """

    _head_module_names = ("proj_text", "proj_img", "norm", "head")

    def __init__(
        self,
        num_classes=NUM_CLASSES,
        text_model_name=TEXT_MODEL_NAME,
        proj_dim=512,
        dropout=0.2,
        freeze_image=True,
        freeze_text=False,
    ):
        super().__init__(
            text_model_name=text_model_name,
            freeze_image=freeze_image,
            freeze_text=freeze_text,
        )

        # Các tầng tuyến tính chiếu Text và Ảnh về chung kích thước 512 chiều
        self.proj_text = nn.Linear(self.text_enc.config.hidden_size, proj_dim)
        self.proj_img = nn.Linear(2048, proj_dim)
        # LayerNorm giúp ổn định độ phân tán giá trị sau khi nhân 2 vector
        self.norm = nn.LayerNorm(proj_dim)

        # Đầu phân loại MLP
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(proj_dim, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes),
        )

    def fuse_features(self, v_text, v_img):
        """Chiếu hai vector về cùng chiều, nhân Hadamard và chuẩn hóa LayerNorm."""
        return self.norm(self.proj_text(v_text) * self.proj_img(v_img))

    def forward(self, pixel_values, input_ids, attention_mask):
        # 1 & 2. Trích xuất vector text [CLS] và ảnh pooled
        v_text, v_img = self._encode_modalities(pixel_values, input_ids, attention_mask)

        # 3 & 4. Chiếu vector, nhân từng phần tử và chuẩn hóa qua LayerNorm
        fused = self.fuse_features(v_text, v_img)  # (Batch, 512)

        # 5. Phân loại qua MLP
        logits = self.head(fused)
        return logits


class CrossAttentionFusionModel(nn.Module):
    """Branch 3B: Cross-Attention Fusion.

    Query = text tokens (L x 256)
    Key, Value = image spatial feature map (49 x 256)
    Residual LayerNorm: norm(Query + CrossAttn(Query, Key, Value))
    Pooling: [CLS] token -> MLP -> 3 classes
    """

    def __init__(
        self,
        num_classes=NUM_CLASSES,
        text_model_name=TEXT_MODEL_NAME,
        attn_dim=256,
        num_heads=4,
        dropout=0.2,
        freeze_image=True,
        freeze_text=False,
    ):
        super().__init__()
        self.text_enc = AutoModel.from_pretrained(text_model_name)
        self.img_enc = ResNet50Backbone(pretrained=True, freeze=freeze_image)

        if freeze_text:
            for p in self.text_enc.parameters():
                p.requires_grad = False

        self.proj_text = nn.Linear(self.text_enc.config.hidden_size, attn_dim)
        self.proj_img = nn.Linear(2048, attn_dim)

        self.cross_attn = nn.MultiheadAttention(
            embed_dim=attn_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm = nn.LayerNorm(attn_dim)

        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(attn_dim, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes),
        )

    def backbone_params(self):
        params = []
        for p in self.text_enc.parameters():
            if p.requires_grad:
                params.append(p)
        for p in self.img_enc.parameters():
            if p.requires_grad:
                params.append(p)
        return params

    def head_params(self):
        head_p = (
            list(self.proj_text.parameters())
            + list(self.proj_img.parameters())
            + list(self.cross_attn.parameters())
            + list(self.norm.parameters())
            + list(self.head.parameters())
        )
        return [p for p in head_p if p.requires_grad]

    def forward(self, pixel_values, input_ids, attention_mask):
        # Text sequence tokens: (B, L, 768)
        text_out = self.text_enc(input_ids=input_ids, attention_mask=attention_mask)
        seq_text = text_out.last_hidden_state  # (B, L, 768)

        # Image spatial features: (B, 49, 2048)
        _, spatial_img = self.img_enc(pixel_values)

        # Projections to attn_dim (256)
        query = self.proj_text(seq_text)      # (B, L, 256)
        key_val = self.proj_img(spatial_img)  # (B, 49, 256)

        # Cross attention: text queries attend to image visual regions
        attn_out, _ = self.cross_attn(query=query, key=key_val, value=key_val)

        # Residual connection + LayerNorm
        res = self.norm(query + attn_out)  # (B, L, 256)

        # Pool via [CLS] token representation (index 0)
        v_fused = res[:, 0, :]  # (B, 256)

        logits = self.head(v_fused)
        return logits


def build_fusion_model(config_name, num_classes=NUM_CLASSES, dropout=0.2, freeze_image=True):
    """Factory function for creating fusion models by config name."""
    if config_name == "both_concat":
        return ConcatFusionModel(num_classes=num_classes, dropout=dropout, freeze_image=freeze_image)
    elif config_name == "both_cross_attn":
        return CrossAttentionFusionModel(num_classes=num_classes, dropout=dropout, freeze_image=freeze_image)
    elif config_name == "both_product":
        return ProductFusionModel(num_classes=num_classes, dropout=dropout, freeze_image=freeze_image)
    else:
        raise ValueError(
            f"Unknown fusion config: '{config_name}'. Expected one of {FUSION_CONFIGS}."
        )


@torch.no_grad()
def evaluate_split(model, loader, device):
    """Evaluate model on a DataLoader, computing probabilities, predictions, and metrics."""
    model.eval()
    all_probs, all_ys, all_ids, all_sarcasms = [], [], [], []

    for batch in loader:
        ids = batch["ids"]
        pixel_values = batch["pixel_values"].to(device)
        input_ids = batch["input_ids"].to(device)
        mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        autocast_dtype = (
            torch.bfloat16
            if (torch.cuda.is_available() and hasattr(torch.cuda, "is_bf16_supported") and torch.cuda.is_bf16_supported())
            else torch.float16
        )
        with torch.autocast("cuda", dtype=autocast_dtype, enabled=(str(device).startswith("cuda"))):
            logits = model(pixel_values=pixel_values, input_ids=input_ids, attention_mask=mask)

        probs = torch.softmax(logits.float(), dim=-1).cpu().numpy()

        all_probs.append(probs)
        all_ys.append(labels.cpu().numpy())
        all_ids.extend(ids)
        all_sarcasms.extend(batch["sarcasms"])

    probs_arr = np.concatenate(all_probs, axis=0)
    ys_arr = np.concatenate(all_ys, axis=0)
    preds = np.argmax(probs_arr, axis=-1)

    macro_f1 = float(f1_score(ys_arr, preds, average="macro", zero_division=0))
    acc = float(accuracy_score(ys_arr, preds))
    return {
        "macro_f1": macro_f1,
        "acc": acc,
        "probs": probs_arr,
        "y_true": ys_arr,
        "ids": all_ids,
        "sarcasm": all_sarcasms,
    }


def train_seed(config_name, seed, cfg, device="cuda", data_df=None):
    """Train one fusion configuration with a specified random seed."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    ho_path = os.path.join(cfg.get("feature_dir", FEATURE_DIR), "train_holdout.json")
    if not os.path.exists(ho_path):
        raise FileNotFoundError(f"Missing holdout file: {ho_path}")

    with open(ho_path, "r", encoding="utf-8") as f:
        ho = json.load(f)

    data_dir = cfg.get("data_dir", DATA_DIR)

    if data_df is not None:
        records = data_df.to_dict("records")
        train_rows = {str(r.get("id", i)): r for i, r in enumerate(records)}
    else:
        train_rows = {r["id"]: r for r in load_jsonl("train", data_dir=data_dir)}

    fit_rows = [train_rows[i] for i in ho["fit"] if i in train_rows]
    holdout_rows = [train_rows[i] for i in ho["holdout"] if i in train_rows]
    test_rows = load_jsonl("test", data_dir=data_dir)

    # Preprocessing integrity check: filter corrupted/missing images
    if is_valid_image is not None:
        bad_fit = [
            r["id"]
            for r in fit_rows
            if not is_valid_image(r.get("image_path", os.path.join(data_dir, r.get("img", ""))))
        ]
        if bad_fit:
            print(f"[{config_name} | Seed {seed}] Preprocessing: filtered {len(bad_fit)} corrupted/missing images from fit set.")
            fit_rows = [r for r in fit_rows if r["id"] not in bad_fit]

    class_weights = torch.tensor(ho["class_weights"], dtype=torch.float).to(device)
    loss_fn = nn.CrossEntropyLoss(weight=class_weights)

    tokenizer = AutoTokenizer.from_pretrained(TEXT_MODEL_NAME)
    image_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    collate = make_fusion_collate_fn(tokenizer, image_transform, max_length=cfg.get("max_length", 128))

    g = torch.Generator().manual_seed(seed)
    fit_loader = DataLoader(
        MultimodalMemeDataset(fit_rows, data_dir=data_dir),
        batch_size=cfg["batch"],
        shuffle=True,
        collate_fn=collate,
        generator=g,
        num_workers=cfg.get("workers", 2),
    )
    holdout_loader = DataLoader(
        MultimodalMemeDataset(holdout_rows, data_dir=data_dir),
        batch_size=cfg.get("eval_batch", 32),
        shuffle=False,
        collate_fn=collate,
        num_workers=cfg.get("workers", 2),
    )
    test_loader = DataLoader(
        MultimodalMemeDataset(test_rows, data_dir=data_dir),
        batch_size=cfg.get("eval_batch", 32),
        shuffle=False,
        collate_fn=collate,
        num_workers=cfg.get("workers", 2),
    )

    model = build_fusion_model(
        config_name=config_name,
        num_classes=NUM_CLASSES,
        dropout=cfg.get("dropout", 0.2),
        freeze_image=cfg.get("freeze_image", True),
    ).to(device)

    optimizer = torch.optim.AdamW([
        {"params": model.backbone_params(), "lr": cfg["lr_backbone"]},
        {"params": model.head_params(), "lr": cfg["lr_head"]},
    ], weight_decay=cfg.get("weight_decay", 0.01))

    total_steps = cfg["epochs"] * len(fit_loader)
    warmup_steps = int(0.10 * total_steps)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda s: min(1.0, (s + 1) / max(1, warmup_steps)) * max(0.0, (total_steps - s) / max(1, total_steps)),
    )

    best_f1 = -1.0
    best_epoch = -1
    best_state = None

    total_batches = len(fit_loader)
    print(f"\n=== Training [{config_name}] Seed {seed} on {str(device).upper()} ===")
    for ep in range(cfg["epochs"]):
        model.train()
        total_loss = 0.0
        for step, batch in enumerate(fit_loader, 1):
            pixel_values = batch["pixel_values"].to(device)
            input_ids = batch["input_ids"].to(device)
            mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            optimizer.zero_grad()
            autocast_dtype = (
                torch.bfloat16
                if (torch.cuda.is_available() and hasattr(torch.cuda, "is_bf16_supported") and torch.cuda.is_bf16_supported())
                else torch.float16
            )
            with torch.autocast("cuda", dtype=autocast_dtype, enabled=(str(device).startswith("cuda"))):
                logits = model(pixel_values=pixel_values, input_ids=input_ids, attention_mask=mask)
                loss = loss_fn(logits.float(), labels)

            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            total_loss += loss.item()

            log_interval = 15 if str(device) == "cpu" else 30
            if step % log_interval == 0 or step == total_batches:
                pct = (step / total_batches) * 100
                print(
                    f"[{config_name} | Seed {seed} | Ep {ep+1:02d}/{cfg['epochs']:02d}] "
                    f"Batch {step:03d}/{total_batches} ({pct:5.1f}%) | Loss: {total_loss / step:.4f}",
                    flush=True,
                )

        avg_loss = total_loss / total_batches
        eval_ho = evaluate_split(model, holdout_loader, device)
        ho_f1 = eval_ho["macro_f1"]
        ho_acc = eval_ho["acc"]

        print(
            f"[{config_name} | Seed {seed}] Epoch {ep + 1:02d}/{cfg['epochs']:02d} | "
            f"Train Loss: {avg_loss:.4f} | Holdout F1: {ho_f1:.4f} | Holdout Acc: {ho_acc:.4f}",
            flush=True,
        )

        if ho_f1 > best_f1:
            best_f1 = ho_f1
            best_epoch = ep + 1
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    print(f"[{config_name} | Seed {seed}] Best Epoch: {best_epoch} | Holdout Macro-F1: {best_f1:.4f}")

    model.load_state_dict(best_state)
    test_eval = evaluate_split(model, test_loader, device)

    res_dir = cfg.get("results_dir", RESULTS_DIR)
    out_path = save_results(
        config=config_name,
        seed=seed,
        ids=test_eval["ids"],
        y_true=test_eval["y_true"],
        probs=test_eval["probs"],
        sarcasm=test_eval["sarcasm"],
        split="test",
        extra={
            "model": config_name,
            "best_epoch": best_epoch,
            "holdout_macro_f1": best_f1,
            "lr_backbone": cfg["lr_backbone"],
            "lr_head": cfg["lr_head"],
        },
        results_dir=res_dir,
    )

    print(
        f"[{config_name} | Seed {seed}] Saved result: {out_path} | "
        f"Test Macro-F1: {test_eval['macro_f1']:.4f} | Test Acc: {test_eval['acc']:.4f}"
    )

    del model, optimizer
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return test_eval["macro_f1"], test_eval["acc"]


class FusionArgs:
    """Hardcoded arguments and hyperparameters for Multimodal Fusion models."""

    def __init__(
        self,
        config="both_concat",
        seeds=(0, 1, 2),
        epochs=5,
        batch_size=16,
        eval_batch_size=32,
        lr_backbone=1.5e-5,
        lr_head=5e-4,
        max_length=128,
        workers=2,
        dropout=0.2,
        weight_decay=0.01,
        freeze_image=True,
        data_dir=DATA_DIR,
        holdout_file=None,
        results_dir=None,
    ):
        self.config = config
        self.seeds = list(seeds)
        self.epochs = epochs
        self.batch_size = batch_size
        self.batch = batch_size
        self.eval_batch_size = eval_batch_size
        self.eval_batch = eval_batch_size
        self.lr_backbone = lr_backbone
        self.lr_head = lr_head
        self.max_length = max_length
        self.workers = workers
        self.dropout = dropout
        self.weight_decay = weight_decay
        self.freeze_image = freeze_image
        self.data_dir = data_dir
        self.holdout_file = holdout_file or os.path.join(FEATURE_DIR, "train_holdout.json")
        self.results_dir = results_dir or os.environ.get("RESULTS_DIR", RESULTS_DIR)


class CrossAttentionArgs(FusionArgs):
    """Hardcoded arguments and hyperparameters for Cross-Attention Multimodal Fusion (Role D)."""

    def __init__(
        self,
        seeds=(0, 1, 2),
        epochs=5,
        batch_size=16,
        eval_batch_size=32,
        lr_backbone=1.5e-5,
        lr_head=5e-4,
        max_length=128,
        attn_dim=256,
        num_heads=4,
        dropout=0.2,
        workers=2,
        freeze_image=True,
        weight_decay=0.01,
        data_dir=DATA_DIR,
        holdout_file=None,
        results_dir=None,
    ):
        super().__init__(
            config="both_cross_attn",
            seeds=seeds,
            epochs=epochs,
            batch_size=batch_size,
            eval_batch_size=eval_batch_size,
            lr_backbone=lr_backbone,
            lr_head=lr_head,
            max_length=max_length,
            workers=workers,
            dropout=dropout,
            weight_decay=weight_decay,
            freeze_image=freeze_image,
            data_dir=data_dir,
            holdout_file=holdout_file,
            results_dir=results_dir,
        )
        self.attn_dim = attn_dim
        self.num_heads = num_heads


def run_training(args=None, device=None, data_df=None):
    """Execute Multimodal Fusion training with hardcoded or user-supplied arguments."""
    if args is None:
        args = FusionArgs()
    elif isinstance(args, dict):
        args = FusionArgs(**args)

    cfg = {
        "epochs": getattr(args, "epochs", 5),
        "batch": getattr(args, "batch_size", getattr(args, "batch", 16)),
        "eval_batch": getattr(args, "eval_batch_size", getattr(args, "eval_batch", 32)),
        "lr_backbone": getattr(args, "lr_backbone", 1.5e-5),
        "lr_head": getattr(args, "lr_head", 5e-4),
        "max_length": getattr(args, "max_length", 128),
        "workers": getattr(args, "workers", 2),
        "dropout": getattr(args, "dropout", 0.2),
        "weight_decay": getattr(args, "weight_decay", 0.01),
        "freeze_image": getattr(args, "freeze_image", True),
        "data_dir": getattr(args, "data_dir", DATA_DIR),
        "feature_dir": os.path.dirname(getattr(args, "holdout_file", os.path.join(FEATURE_DIR, "train_holdout.json"))),
        "holdout_file": getattr(args, "holdout_file", os.path.join(FEATURE_DIR, "train_holdout.json")),
        "results_dir": getattr(args, "results_dir", RESULTS_DIR),
    }

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    config_choice = getattr(args, "config", "both_concat")
    if isinstance(config_choice, (list, tuple)):
        configs_to_run = list(config_choice)
    elif config_choice == "all":
        configs_to_run = FUSION_CONFIGS
    else:
        configs_to_run = [config_choice]

    seeds = list(getattr(args, "seeds", [0, 1, 2]))

    print(f"Device: {str(device).upper()}")
    print(f"Configurations to run: {configs_to_run}")
    print(f"Seeds: {seeds}")
    print(f"Hyperparameters: {cfg}")

    if data_df is not None:
        print(f"Preprocessing integration: using preprocessed DataFrame ({len(data_df)} records).")
    elif is_valid_image is not None:
        data_dir = cfg["data_dir"]
        raw_train = load_jsonl("train", data_dir=str(data_dir))
        corrupt = [
            r["id"]
            for r in raw_train
            if not is_valid_image(r.get("image_path", os.path.join(data_dir, r.get("img", ""))))
        ]
        if corrupt:
            print(f"Notice: {len(corrupt)} corrupted/missing images identified via preprocessing filter.")
        else:
            print(f"Data verification: All {len(raw_train)} training images validated successfully.")

    summary = {}
    for cname in configs_to_run:
        f1_list, acc_list = [], []
        for s in seeds:
            f1, acc = train_seed(cname, s, cfg, device=device, data_df=data_df)
            f1_list.append(f1)
            acc_list.append(acc)

        summary[cname] = {
            "f1_mean": float(np.mean(f1_list)),
            "f1_std": float(np.std(f1_list)),
            "acc_mean": float(np.mean(acc_list)),
            "acc_std": float(np.std(acc_list)),
            "f1_per_seed": [round(x, 4) for x in f1_list],
            "acc_per_seed": [round(x, 4) for x in acc_list],
        }

    print("\n" + "=" * 60)
    print("FINAL MULTIMODAL FUSION SUMMARY")
    print("=" * 60)
    for cname, res in summary.items():
        print(f"[{cname}]")
        print(f"  Macro-F1 per seed: {res['f1_per_seed']} => mean: {res['f1_mean']:.4f} ± {res['f1_std']:.4f}")
        print(f"  Accuracy per seed: {res['acc_per_seed']} => mean: {res['acc_mean']:.4f} ± {res['acc_std']:.4f}")

    return summary


def main(args=None):
    if args is not None:
        return run_training(args)

    if len(sys.argv) > 1 and any(arg.startswith("--") for arg in sys.argv[1:]):
        parser = argparse.ArgumentParser(description="Multimodal Fusion Trainer for Memotion 7k")
        parser.add_argument(
            "--config",
            choices=FUSION_CONFIGS + ["all"],
            default="all",
            help="Fusion configuration to train ('both_concat', 'both_cross_attn', 'both_product', or 'all')",
        )
        parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2], help="Random seeds")
        parser.add_argument("--epochs", type=int, default=5, help="Number of training epochs")
        parser.add_argument("--batch", type=int, default=16, help="Batch size for training")
        parser.add_argument("--eval-batch", type=int, default=32, help="Batch size for evaluation")
        parser.add_argument("--lr-backbone", type=float, default=1.5e-5, help="Learning rate for BERT backbone")
        parser.add_argument("--lr-head", type=float, default=5e-4, help="Learning rate for fusion head")
        parser.add_argument("--max-length", type=int, default=128, help="Max token length for text")
        parser.add_argument("--workers", type=int, default=2, help="DataLoader worker processes")
        parser.add_argument("--unfreeze-image", action="store_true", help="Unfreeze ResNet50 backbone")
        parsed_args = parser.parse_args()
        return run_training(parsed_args)
    else:
        return run_training(FusionArgs())


if __name__ == "__main__":
    main()
