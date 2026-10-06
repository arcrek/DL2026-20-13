"""Offline characterization of the working, pre-refactor fusion contracts.

Legacy references deliberately duplicate the old topology/forward arithmetic:
they must not inherit production classes or call their fusion/encoding helpers.
These tests characterize fake-encoder CPU behavior, not model quality.
"""
import inspect
import os
import sys
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
from torch import nn  # noqa: E402

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import models.fusion as fusion  # noqa: E402


class FakeTextEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(hidden_size=7)
        self.embedding = nn.Embedding(31, 7)
        self.linear = nn.Linear(7, 7)

    def forward(self, input_ids, attention_mask):
        self.last_inputs = (input_ids, attention_mask)
        tokens = self.linear(self.embedding(input_ids))
        # Mask influences CLS too, exposing lost/misrouted attention masks.
        context = (tokens * attention_mask.unsqueeze(-1)).mean(dim=1, keepdim=True)
        return SimpleNamespace(last_hidden_state=tokens + context)


class FakeImageEncoder(nn.Module):
    def __init__(self, pretrained=True, freeze=True):
        super().__init__()
        self.pretrained = pretrained
        self.linear = nn.Linear(3, 2048)
        self.bn = nn.BatchNorm1d(2048)
        if freeze:
            for parameter in self.parameters():
                parameter.requires_grad = False

    def forward(self, pixels):
        pooled = self.bn(self.linear(pixels.mean(dim=(-2, -1))))
        spatial = pooled.unsqueeze(1).expand(-1, 49, -1)
        return pooled, spatial


@pytest.fixture(autouse=True)
def offline_encoders(monkeypatch):
    """Patch only the model's lookup namespace; fail on real weight loaders."""
    rng = torch.get_rng_state()
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    torch.manual_seed(73)
    calls = []

    def fake_text(name):
        calls.append(name)
        return FakeTextEncoder()

    def forbidden(*args, **kwargs):
        pytest.fail("A real pretrained weight/tokenizer loader was called")

    monkeypatch.setattr(fusion.AutoModel, "from_pretrained", fake_text)
    monkeypatch.setattr(fusion, "ResNet50Backbone", FakeImageEncoder)
    monkeypatch.setattr(fusion.models, "resnet50", forbidden)
    monkeypatch.setattr(fusion.AutoTokenizer, "from_pretrained", forbidden)
    try:
        yield calls
    finally:
        torch.set_rng_state(rng)
        torch.set_num_threads(threads)


class LegacyConcatFusionModel(nn.Module):
    def __init__(self, num_classes=3, text_model_name="bert-base-uncased",
                 dropout=0.2, freeze_image=True, freeze_text=False):
        super().__init__()
        self.text_enc = FakeTextEncoder()
        self.img_enc = FakeImageEncoder(pretrained=True, freeze=freeze_image)
        if freeze_text:
            for p in self.text_enc.parameters():
                p.requires_grad = False
        self.head = nn.Sequential(
            nn.Dropout(dropout), nn.Linear(self.text_enc.config.hidden_size + 2048, 512),
            nn.ReLU(), nn.Dropout(dropout), nn.Linear(512, num_classes),
        )

    def forward(self, pixel_values, input_ids, attention_mask):
        text = self.text_enc(input_ids=input_ids, attention_mask=attention_mask)
        image, _ = self.img_enc(pixel_values)
        return self.head(torch.cat([text.last_hidden_state[:, 0, :], image], dim=-1))


class LegacyProductFusionModel(nn.Module):
    def __init__(self, num_classes=3, text_model_name="bert-base-uncased",
                 proj_dim=512, dropout=0.2, freeze_image=True, freeze_text=False):
        super().__init__()
        self.text_enc = FakeTextEncoder()
        self.img_enc = FakeImageEncoder(pretrained=True, freeze=freeze_image)
        if freeze_text:
            for p in self.text_enc.parameters():
                p.requires_grad = False
        self.proj_text = nn.Linear(self.text_enc.config.hidden_size, proj_dim)
        self.proj_img = nn.Linear(2048, proj_dim)
        self.norm = nn.LayerNorm(proj_dim)
        self.head = nn.Sequential(
            nn.Dropout(dropout), nn.Linear(proj_dim, 256), nn.ReLU(),
            nn.Dropout(dropout), nn.Linear(256, num_classes),
        )

    def forward(self, pixel_values, input_ids, attention_mask):
        text = self.text_enc(input_ids=input_ids, attention_mask=attention_mask)
        image, _ = self.img_enc(pixel_values)
        product = self.proj_text(text.last_hidden_state[:, 0, :]) * self.proj_img(image)
        return self.head(self.norm(product))


VARIANTS = [
    pytest.param(fusion.ConcatFusionModel, LegacyConcatFusionModel, {}, id="concat"),
    pytest.param(fusion.ProductFusionModel, LegacyProductFusionModel, {}, id="product-default"),
    pytest.param(fusion.ProductFusionModel, LegacyProductFusionModel, {"proj_dim": 11}, id="product-small"),
]
MODELS = [fusion.ConcatFusionModel, fusion.ProductFusionModel]
CONFIGS = [
    ("both_concat", fusion.ConcatFusionModel),
    ("both_cross_attn", fusion.CrossAttentionFusionModel),
    ("both_product", fusion.ProductFusionModel),
]


def inputs(batch=2, seq=5):
    pixels = torch.linspace(-2, 3, batch * 3 * 4 * 4).reshape(batch, 3, 4, 4)
    ids = torch.arange(batch * seq).reshape(batch, seq) % 31
    mask = ((torch.arange(batch * seq).reshape(batch, seq) % 3) != 0).long()
    return pixels, ids, mask


def assert_exact(left, right):
    torch.testing.assert_close(left, right, rtol=0, atol=0)


@pytest.mark.parametrize("config,model_class", CONFIGS)
@pytest.mark.parametrize("custom", [False, True])
def test_factory_defaults_kwargs_and_shapes(config, model_class, custom, offline_encoders):
    kwargs = {"num_classes": 5, "dropout": 0.37, "freeze_image": False} if custom else {}
    model = fusion.build_fusion_model(config, **kwargs).eval()
    assert type(model) is model_class
    assert offline_encoders == ["bert-base-uncased"]
    assert model.img_enc.pretrained is True
    assert all(p.requires_grad == custom for p in model.img_enc.parameters())
    assert all(p.requires_grad for p in model.text_enc.parameters())
    assert model.head[0].p == (0.37 if custom else 0.2)
    assert model.head[3].p == model.head[0].p
    assert model.head[-1].out_features == (5 if custom else 3)
    args = inputs()
    with torch.no_grad():
        positional = model(*args)
        keyword = model(pixel_values=args[0], input_ids=args[1], attention_mask=args[2])
    assert positional.shape == (2, 5 if custom else 3)
    assert torch.isfinite(positional).all()
    assert_exact(positional, keyword)
    assert model.text_enc.last_inputs[0] is args[1]
    assert model.text_enc.last_inputs[1] is args[2]
    assert model.backbone_params() and model.head_params()


def test_factory_invalid_config():
    with pytest.raises(ValueError, match="not-a-config"):
        fusion.build_fusion_model("not-a-config")


def test_public_api_and_exports():
    import fusion as runner
    import models

    common = "num_classes=3, text_model_name='bert-base-uncased', "
    tail = "dropout=0.2, freeze_image=True, freeze_text=False"
    assert str(inspect.signature(fusion.ConcatFusionModel)) == f"({common}{tail})"
    assert str(inspect.signature(fusion.ProductFusionModel)) == f"({common}proj_dim=512, {tail})"
    assert str(inspect.signature(fusion.build_fusion_model)) == (
        "(config_name, num_classes=3, dropout=0.2, freeze_image=True)"
    )
    assert fusion.FUSION_CONFIGS == [name for name, _ in CONFIGS]
    for _, cls in CONFIGS:
        assert str(inspect.signature(cls.forward)) == "(self, pixel_values, input_ids, attention_mask)"
        assert str(inspect.signature(cls.backbone_params)) == "(self)"
        assert str(inspect.signature(cls.head_params)) == "(self)"
        assert getattr(runner, cls.__name__) is cls
        assert getattr(models, cls.__name__) is cls
    assert runner.build_fusion_model is models.build_fusion_model is fusion.build_fusion_model


@pytest.mark.parametrize("cls,legacy,kwargs", VARIANTS)
@pytest.mark.parametrize("batch,seq", [(1, 1), (1, 5), (2, 1), (2, 5)])
def test_same_seed_state_rng_strict_loading_and_legacy_logits(cls, legacy, kwargs, batch, seq):
    options = dict(kwargs, num_classes=5, dropout=0.31, text_model_name="offline-custom")
    torch.manual_seed(913)
    reference = legacy(**options).eval()
    reference_rng = torch.get_rng_state().clone()
    torch.manual_seed(913)
    model = cls(**options).eval()
    assert_exact(torch.get_rng_state(), reference_rng)  # Final RNG, not just weights.
    old, new = reference.state_dict(), model.state_dict()
    assert list(new) == list(old)
    assert [(k, tuple(v.shape)) for k, v in new.items()] == [(k, tuple(v.shape)) for k, v in old.items()]
    for key in old:
        assert_exact(new[key], old[key])
    assert list(model._modules) == list(reference._modules)
    assert repr(model.head) == repr(reference.head)
    # Non-initial weights/buffers prove strict loading works in both directions.
    with torch.no_grad():
        for p in reference.parameters():
            p.add_(0.013)
        reference.img_enc.bn.running_mean.fill_(0.17)
    loaded = model.load_state_dict(reference.state_dict(), strict=True)
    assert loaded.missing_keys == loaded.unexpected_keys == []
    args = inputs(batch, seq)
    with torch.no_grad():
        assert_exact(model(*args), reference(*args))
    with torch.no_grad():
        model.head[-1].bias.add_(0.23)
    loaded = reference.load_state_dict(model.state_dict(), strict=True)
    assert loaded.missing_keys == loaded.unexpected_keys == []
    with torch.no_grad():
        assert_exact(reference(*args), model(*args))
    for key in model.state_dict():
        assert_exact(reference.state_dict()[key], model.state_dict()[key])


@pytest.mark.parametrize("cls,legacy,kwargs", VARIANTS)
@pytest.mark.parametrize("batch,seq", [(1, 1), (2, 5)])
def test_explicit_math_with_zero_negative_features(cls, legacy, kwargs, batch, seq):
    model = cls(num_classes=4, **kwargs).eval()
    text = torch.linspace(-3, 2, batch * seq * 7).reshape(batch, seq, 7)
    image = torch.linspace(2, -4, batch * 2048).reshape(batch, 2048)
    text[..., 0] = 0
    image[:, ::3] = 0
    handles = [
        model.text_enc.register_forward_hook(lambda module, args, out: SimpleNamespace(last_hidden_state=text)),
        model.img_enc.register_forward_hook(lambda module, args, out: (image, image[:, None, :].expand(-1, 49, -1))),
    ]
    try:
        with torch.no_grad():
            if cls is fusion.ConcatFusionModel:
                fused = torch.cat([text[:, 0, :], image], dim=-1)
            else:
                fused = model.norm(model.proj_text(text[:, 0, :]) * model.proj_img(image))
            expected = model.head(fused)
            actual = model(*inputs(batch, seq))
        assert actual.shape == (batch, 4)
        assert_exact(actual, expected)
    finally:
        for handle in handles:
            handle.remove()


@pytest.mark.parametrize("cls", MODELS)
@pytest.mark.parametrize("freeze_text", [False, True])
@pytest.mark.parametrize("freeze_image", [False, True])
def test_freeze_gradients_groups_and_adamw(cls, freeze_text, freeze_image):
    model = cls(freeze_text=freeze_text, freeze_image=freeze_image, dropout=0).train()
    for enc, frozen in [(model.text_enc, freeze_text), (model.img_enc, freeze_image)]:
        assert all(p.requires_grad == (not frozen) for p in enc.parameters())
    head_modules = [model.head] if cls is fusion.ConcatFusionModel else [model.proj_text, model.proj_img, model.norm, model.head]
    expected_backbone = [p for enc in [model.text_enc, model.img_enc] for p in enc.parameters() if p.requires_grad]
    expected_head = [p for module in head_modules for p in module.parameters() if p.requires_grad]
    backbone, head = model.backbone_params(), model.head_params()
    assert isinstance(backbone, list) and isinstance(head, list)
    assert [id(p) for p in backbone] == [id(p) for p in expected_backbone]
    assert [id(p) for p in head] == [id(p) for p in expected_head]
    b_ids, h_ids = {id(p) for p in backbone}, {id(p) for p in head}
    assert len(b_ids) == len(backbone) and len(h_ids) == len(head)
    assert b_ids.isdisjoint(h_ids)
    assert b_ids | h_ids == {id(p) for p in model.parameters() if p.requires_grad}
    if freeze_text and freeze_image:
        assert backbone == []
    optimizer = torch.optim.AdamW([
        {"params": backbone, "lr": 1.5e-5}, {"params": head, "lr": 5e-4},
    ], weight_decay=0.01)
    assert [g["lr"] for g in optimizer.param_groups] == [1.5e-5, 5e-4]
    before = {name: p.detach().clone() for name, p in model.named_parameters()}
    optimizer.zero_grad(set_to_none=True)
    nn.functional.cross_entropy(model(*inputs()), torch.tensor([0, 2])).backward()
    for name, p in model.named_parameters():
        if p.requires_grad:
            assert p.grad is not None, name
            assert torch.isfinite(p.grad).all(), name
        else:
            assert p.grad is None, name
    optimizer.step()
    for name, p in model.named_parameters():
        if p.requires_grad:
            assert torch.isfinite(p).all(), name
            assert optimizer.state[p]["step"].item() == 1
        else:
            assert_exact(p, before[name])
    assert any(not torch.equal(p, before[name]) for name, p in model.named_parameters() if p.requires_grad)


@pytest.mark.parametrize("cls", MODELS)
@pytest.mark.parametrize("freeze_text", [False, True])
@pytest.mark.parametrize("freeze_image", [False, True])
def test_feature_gradients_even_with_frozen_encoders(cls, freeze_text, freeze_image):
    model = cls(freeze_text=freeze_text, freeze_image=freeze_image, dropout=0).eval()
    text = torch.randn(2, 5, 7, requires_grad=True)
    image = torch.randn(2, 2048, requires_grad=True)
    handles = [
        model.text_enc.register_forward_hook(lambda module, args, out: SimpleNamespace(last_hidden_state=text)),
        model.img_enc.register_forward_hook(lambda module, args, out: (image, image[:, None, :].expand(-1, 49, -1))),
    ]
    try:
        nn.functional.cross_entropy(model(*inputs()), torch.tensor([0, 2])).backward()
        for features in [text, image]:
            assert features.grad is not None
            assert torch.isfinite(features.grad).all()
            assert features.grad.abs().sum().item() > 0
        assert torch.count_nonzero(text.grad[:, 1:, :]).item() == 0  # CLS only.
    finally:
        for handle in handles:
            handle.remove()


@pytest.mark.parametrize("cls", MODELS)
def test_frozen_image_bn_buffers_remain_mutable_in_train(cls):
    model = cls(freeze_image=True, freeze_text=True).train()
    bn = model.img_enc.bn
    assert model.img_enc.training and bn.training and model.text_enc.training
    weights = {name: p.detach().clone() for name, p in model.img_enc.named_parameters()}
    mean, variance, batches = bn.running_mean.clone(), bn.running_var.clone(), bn.num_batches_tracked.clone()
    model(*inputs())
    assert not torch.equal(bn.running_mean, mean)
    assert not torch.equal(bn.running_var, variance)
    assert_exact(bn.num_batches_tracked, batches + 1)
    for name, p in model.img_enc.named_parameters():
        assert not p.requires_grad
        assert_exact(p, weights[name])
    model.eval()
    assert not model.img_enc.training and not bn.training
    buffers = {name: value.clone() for name, value in bn.named_buffers()}
    model(*inputs())
    for name, value in bn.named_buffers():
        assert_exact(value, buffers[name])
    model.train()
    assert bn.training  # No forced eval override for frozen encoders.


@pytest.mark.parametrize("cls,kwargs", [
    (fusion.ConcatFusionModel, {}),
    (fusion.ProductFusionModel, {}),
    (fusion.ProductFusionModel, {"proj_dim": 13}),
])
def test_fuse_features_exact_math_and_gradients(cls, kwargs):
    model = cls(dropout=0, **kwargs).eval()
    batch = 3
    hidden_size = model.text_enc.config.hidden_size
    v_text = torch.randn(batch, hidden_size, requires_grad=True)
    v_img = torch.randn(batch, 2048, requires_grad=True)

    fused = model.fuse_features(v_text, v_img)

    if cls is fusion.ConcatFusionModel:
        expected = torch.cat([v_text, v_img], dim=-1)
        assert fused.shape == (batch, hidden_size + 2048)
    else:
        expected = model.norm(model.proj_text(v_text) * model.proj_img(v_img))
        assert fused.shape == (batch, kwargs.get("proj_dim", 512))

    assert_exact(fused, expected)

    loss = fused.sum()
    loss.backward()
    assert v_text.grad is not None and torch.isfinite(v_text.grad).all()
    assert v_img.grad is not None and torch.isfinite(v_img.grad).all()
    assert v_text.grad.abs().sum().item() > 0
    assert v_img.grad.abs().sum().item() > 0
    if cls is fusion.ProductFusionModel:
        for module in [model.proj_text, model.proj_img, model.norm]:
            for p in module.parameters():
                if p.requires_grad:
                    assert p.grad is not None and torch.isfinite(p.grad).all()


def test_fusion_args():
    args = fusion.FusionArgs()
    assert args.config == "both_concat"
    assert args.seeds == [0, 1, 2]
    assert args.epochs == 5
    assert args.batch_size == 16
    assert args.workers == 2

    args_all = fusion.FusionArgs(config="all", seeds=[0], epochs=1, batch_size=8)
    assert args_all.config == "all"
    assert args_all.seeds == [0]
    assert args_all.epochs == 1
    assert args_all.batch_size == 8


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
