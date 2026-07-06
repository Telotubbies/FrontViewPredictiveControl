"""
Config package — loads YAML configuration as module-level constants.

Source of truth: ``config/default.yaml`` at the repository root.
The loader flattens nested sections (e.g. ``mpc.MPC_DT`` → ``MPC_DT``)
so that downstream code can ``from config import MPC_DT`` unchanged.
"""
from pathlib import Path

import yaml

# ── Paths ────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_YAML = PROJECT_ROOT / "config" / "default.yaml"

# Keys whose YAML list value must become a tuple (kernel sizes, etc.)
_TUPLE_KEYS = frozenset({
    "RAW_MASK_DILATE_KERNEL", "RAW_MASK_CLOSE_KERNEL", "RAW_MASK_CLOSE_KERNEL2",
    "MASK_DILATE_VERTICAL", "MASK_CLOSE_KERNEL_LARGE",
    "BEV_ALONG_ROAD_DILATE_KERNEL",
})


def _load_yaml_config(path: Path) -> dict:
    """Load and flatten a YAML config file. Raise if missing or empty."""
    if not path.exists():
        raise FileNotFoundError(
            f"config.yaml not found at {path} — please ensure config/default.yaml exists"
        )
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not data:
        raise ValueError(f"config file {path} is empty or invalid")
    flat: dict = {}
    for section, values in data.items():
        if isinstance(values, dict):
            for k, v in values.items():
                flat[k] = v
        else:
            flat[section] = values
    return flat


def _apply_config(flat: dict) -> None:
    """Set module-level constants from *flat*; convert lists to tuples per _TUPLE_KEYS."""
    import sys
    mod = sys.modules[__name__]
    for k, v in flat.items():
        if k in _TUPLE_KEYS and isinstance(v, list):
            v = tuple(v)
        setattr(mod, k, v)


# Load and apply
_flat = _load_yaml_config(DEFAULT_YAML)
_apply_config(_flat)

# ── Derived / legacy aliases ─────────────────────────────────────────────────
UNET_WEIGHT_BASE = UNET_CENTER_WEIGHT  # type: ignore[name-defined]
DEFAULT_MODEL = PROJECT_ROOT / "model" / "lane_unet_final.pth"

# ── Performance flags (not user-tunable parameters) ──────────────────────────
PERFORMANCE_USE_CPU_BEV_TRANSFORM = True
PERFORMANCE_USE_CPU_SLIDING_WINDOW = True
PERFORMANCE_USE_CPU_REFERENCE_GEN = True
PERFORMANCE_USE_GPU_MPC_EVAL = False
PERFORMANCE_GPU_READY = True
PERFORMANCE_BATCH_SIZE = 1
PERFORMANCE_ENABLE_LOGGING = False


class UXColors:
    """UI colors (BGR for cv2, RGB for pygame)."""
    GREEN_VALID_BGR = (100, 255, 120)
    ORANGE_PATH_BGR = (0, 200, 255)
    RED_INVALID_BGR = (0, 150, 255)
    GREEN_VALID = (120, 255, 100)
    RED_INVALID = (255, 150, 150)
    TEXT_NORMAL = (200, 200, 200)
    TEXT_DIM = (120, 125, 140)
    SECTION_LABEL = (180, 180, 180)
