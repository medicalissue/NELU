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
OTHER_CONFIGS = [
    REPO_ROOT / "configs" / "cifar100.yaml",
    REPO_ROOT / "configs" / "ablation" / "gamma_init.yaml",
]

_REQUIRED_IMAGENET_KEYS = {
    "model", "num_classes", "data_dir",
    "batch_size", "opt", "weight_decay",
    "sched", "epochs", "warmup_epochs",
    "activation", "gamma_init", "rms_mode",
    "seed",
}


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
    assert cfg["rms_mode"] in {"per_token", "per_sample"}
    assert cfg["seed"] == 42, "All shipped configs pin seed=42 for comparability."
    assert 0 <= cfg["gamma_init"] <= 1e-3
    assert cfg["epochs"] > 0
    assert 0 <= cfg["warmup_epochs"] < cfg["epochs"]


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


def test_transformer_configs_use_per_token() -> None:
    for path in IMAGENET_CONFIGS:
        cfg = yaml.safe_load(path.read_text())
        is_transformer = cfg["model"].startswith(("deit_", "swin_", "vit_"))
        assert cfg["rms_mode"] == ("per_token" if is_transformer else "per_sample"), (
            f"{path.name}: rms_mode does not match architecture family"
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
