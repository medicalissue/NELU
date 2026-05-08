"""Schema-level sanity checks for every shipped config.

These tests do not train anything; they verify that each YAML parses
cleanly and that the handful of fields the trainers rely on are present
with sensible types and values.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml


REPO_ROOT = Path(__file__).resolve().parent.parent
IMAGENET_CONFIGS = sorted((REPO_ROOT / "configs" / "imagenet").glob("*.yaml"))
CIFAR_DIR = REPO_ROOT / "configs" / "cifar100"
CIFAR_BASE = CIFAR_DIR / "_base.yaml"
# Model-specific stubs under configs/cifar100/ inherit from _base.yaml via
# the ``include:`` directive resolved by train/cifar.py::_load_config_with_includes.
CIFAR_STUBS = sorted(p for p in CIFAR_DIR.glob("*.yaml") if p.name != "_base.yaml")
OTHER_CONFIGS = [CIFAR_BASE]

_REQUIRED_IMAGENET_KEYS = {
    "model", "num_classes", "data_dir",
    "batch_size", "opt", "weight_decay",
    "sched", "epochs", "warmup_epochs",
    "activation",
    "gamma_init",
    "seed",
}


def _norm_axes_token(cfg: dict) -> object:
    """Return a hashable view of norm_axes for schema checks.

    Returns ``None`` when the config omits ``norm_axes`` (the new
    default policy: ``train.swap.default_norm_axes`` returns
    ``"position"``, rank-dispatched at runtime). YAML lists (e.g.
    ``[-1]``) are unhashable so we convert them to tuples.
    """
    axes = cfg.get("norm_axes")
    if axes is None:
        return None
    if isinstance(axes, list):
        return tuple(axes)
    return axes


@pytest.mark.parametrize("path", IMAGENET_CONFIGS, ids=lambda p: p.name)
def test_imagenet_config_has_required_fields(path: Path) -> None:
    cfg = yaml.safe_load(path.read_text())
    missing = _REQUIRED_IMAGENET_KEYS - cfg.keys()
    assert not missing, f"{path.name} missing keys: {missing}"


@pytest.mark.parametrize("path", IMAGENET_CONFIGS, ids=lambda p: p.name)
def test_imagenet_config_values_are_sane(path: Path) -> None:
    cfg = yaml.safe_load(path.read_text())
    assert cfg["num_classes"] == 1000
    assert cfg["activation"] in {"relu", "gelu", "silu", "nelu", "nilu"}
    axes = _norm_axes_token(cfg)
    valid_aliases = {"position", "channel", "sample"}
    # ``norm_axes`` is now optional — when omitted, swap.default_norm_axes
    # picks the rank-dispatched ``"position"`` alias. When present it
    # must be either a known alias or an explicit axis tuple.
    assert axes is None or axes in valid_aliases or isinstance(axes, tuple), (
        f"{path.name}: norm_axes={axes!r} is neither an alias nor an axis tuple"
    )
    assert cfg["seed"] == 42, "All shipped configs pin seed=42 for comparability."
    assert cfg["epochs"] > 0
    assert 0 <= cfg["warmup_epochs"] < cfg["epochs"]
    # torch.compile knobs are opt-in; when present they must be null by default
    # so paper-fidelity runs match MMPretrain's un-compiled baseline exactly.
    assert cfg.get("torchcompile") in (None, "inductor", "eager", "aot_eager")
    assert cfg.get("torchcompile_mode") in (
        None, "default", "reduce-overhead",
        "max-autotune", "max-autotune-no-cudagraphs",
    )


def test_every_family_has_both_scales() -> None:
    """Paper's experiment matrix requires two scales per architecture."""
    names = {p.stem for p in IMAGENET_CONFIGS}
    expected = {
        "convnext_tiny", "convnext_small",
        "deit_small", "deit_base",
        "swin_tiny", "swin_small",
        "efficientnet_b0", "efficientnet_b2",
    }
    assert names == expected, f"Config set changed: {names ^ expected}"


def test_axis_policy_matches_architecture_family() -> None:
    """Configs use the unified ``"position"`` default, or override explicitly.

    Under the new policy, ``norm_axes`` is rank-dispatched at runtime
    (CNN→spatial, Tx→token), so configs typically just omit it. Legacy
    explicit values are still tolerated for backward-compatible schemas.
    """
    for path in IMAGENET_CONFIGS:
        cfg = yaml.safe_load(path.read_text())
        axes = _norm_axes_token(cfg)
        model = cfg["model"]
        if axes is None or axes == "position":
            # New default — rank-dispatched at runtime.
            continue
        if model.startswith(("deit_", "swin_", "vit_", "convnext_")):
            # Legacy explicit channel-axis spec.
            assert axes in {"channel", (-1,)}, (
                f"{path.name}: expected position/channel-axis gate stats, got {axes!r}"
            )
        elif model.startswith("efficientnet_"):
            # Legacy explicit sample-axis spec.
            assert axes == "sample", (
                f"{path.name}: EfficientNet legacy spec should be 'sample'; got {axes!r}"
            )


def test_convnext_deit_swin_share_mmpretrain_defaults() -> None:
    """Every config ported from MMPretrain has the same warmup, label smoothing,
    mixup, cutmix, rand-aug policy, and erase prob — regressions here mean
    the port drifted from the reproduced MMPretrain recipe."""
    for path in IMAGENET_CONFIGS:
        cfg = yaml.safe_load(path.read_text())
        if cfg["model"].startswith("efficientnet_"):
            continue  # timm-derived recipe, different numbers
        assert cfg["warmup_epochs"] == 20, f"{path.name}: warmup_epochs != 20"
        assert cfg["smoothing"] == 0.1
        assert cfg["mixup"] == 0.8
        assert cfg["cutmix"] == 1.0
        assert cfg["aa"] == "rand-m9-mstd0.5-inc1"
        assert cfg["reprob"] == 0.25
        assert cfg["remode"] == "rand", f"{path.name}: remode should be 'rand'"
        assert cfg["color_jitter"] == 0.0, (
            f"{path.name}: MMPretrain pipeline has no ColorJitter"
        )


def test_transformer_configs_set_layer_scale() -> None:
    """All ImageNet vision-transformer recipes (DeiT, Swin, ConvNeXt)
    initialize LayerScale at 1e-6 — the standard CaiT setting that
    paper-default NELU runs adopt."""
    for path in IMAGENET_CONFIGS:
        cfg = yaml.safe_load(path.read_text())
        model = cfg["model"]
        if model.startswith("efficientnet_"):
            continue
        kwargs = cfg.get("model_kwargs", {})
        if model.startswith("convnext_"):
            assert kwargs.get("ls_init_value") == 1.0e-6, (
                f"{path.name}: ConvNeXt expects ls_init_value=1e-6"
            )
        else:
            assert kwargs.get("init_values") == 1.0e-6, (
                f"{path.name}: transformer expects init_values=1e-6"
            )


def test_imagenet_configs_share_unified_ema() -> None:
    """All ImageNet runs use the same EMA recipe (decay 0.9999) so that
    eval_top1 in W&B is directly comparable across the whole sweep."""
    for path in IMAGENET_CONFIGS:
        cfg = yaml.safe_load(path.read_text())
        assert cfg.get("model_ema") is True, f"{path.name}: model_ema must be on"
        assert cfg.get("model_ema_decay") == 0.9999, (
            f"{path.name}: unified decay 0.9999, got {cfg.get('model_ema_decay')!r}"
        )


def test_imagenet_configs_set_validation_batch_size() -> None:
    """Eval has no optimizer state / activations-for-grad to hold, so we
    run validation at 2× the train batch — finishes ~2× faster."""
    for path in IMAGENET_CONFIGS:
        cfg = yaml.safe_load(path.read_text())
        bs = cfg["batch_size"]
        vbs = cfg.get("validation_batch_size")
        assert vbs == 2 * bs, (
            f"{path.name}: validation_batch_size should be 2×batch_size "
            f"({2 * bs}); got {vbs!r}"
        )


def test_efficientnet_configs_match_timm_training_script() -> None:
    """timm training_script.mdx RMSProp-TF recipe hallmarks."""
    for path in IMAGENET_CONFIGS:
        cfg = yaml.safe_load(path.read_text())
        if not cfg["model"].startswith("efficientnet_"):
            continue
        assert cfg["opt"] == "rmsproptf"
        assert cfg["opt_eps"] == 1.0e-3
        assert cfg["sched"] == "step"
        assert cfg["epochs"] == 450
        assert cfg["decay_epochs"] == 2.4
        assert cfg["decay_rate"] == 0.97
        assert cfg["model_ema"] is True
        assert cfg["model_ema_decay"] == 0.9999
        assert cfg["aa"] == "rand-m9-mstd0.5"


@pytest.mark.parametrize("path", OTHER_CONFIGS, ids=lambda p: p.name)
def test_other_configs_parse(path: Path) -> None:
    cfg = yaml.safe_load(path.read_text())
    assert "seed" in cfg
    assert cfg["seed"] == 42


@pytest.mark.parametrize("path", CIFAR_STUBS, ids=lambda p: p.name)
def test_cifar_stubs_inherit_from_base(path: Path) -> None:
    """Every model stub declares include: _base.yaml and picks a model."""
    cfg = yaml.safe_load(path.read_text())
    assert cfg.get("include") == "_base.yaml", (
        f"{path.name}: expected 'include: _base.yaml' to inherit the unified recipe"
    )
    assert "model" in cfg, f"{path.name}: missing 'model' field"


def test_cifar_base_has_unified_recipe() -> None:
    """The CIFAR-100 base config fixes the shared recipe fields we care about."""
    cfg = yaml.safe_load(CIFAR_BASE.read_text())
    # chenyaofo official recipe hallmarks (protocol, not execution backend)
    assert cfg["optimizer"] == "sgd"
    assert cfg["lr"] == 0.1
    assert cfg["momentum"] == 0.9
    assert cfg["weight_decay"] == 5.0e-4
    assert cfg["nesterov"] is True
    assert cfg["scheduler"] == "cosine"
    assert cfg["epochs"] == 200
    assert cfg["warmup_epochs"] == 0
    assert cfg["batch_size"] == 256
    # Execution backend: bf16 AMP + inductor. Applied uniformly across
    # activations so the comparison stays controlled; not part of the
    # protocol per se, but must be the same for every run.
    assert cfg["amp"] is True
    assert cfg["amp_dtype"] == "bfloat16"
    assert cfg["compile"] is True
    assert cfg["compile_backend"] == "inductor"
