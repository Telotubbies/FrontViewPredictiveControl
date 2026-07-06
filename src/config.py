"""
LKA Pipeline — ค่าคงที่โหลดจาก config.yaml (ตั้งค่าที่เดียว)
config.yaml เป็น source of truth เดียว — ไม่มี hardcoded fallback
"""
from pathlib import Path

import yaml

# ── Paths (ยังคงใน Python) ───────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent

# Keys ที่ค่าใน YAML เป็น list → แปลงเป็น tuple (kernel sizes ฯลฯ)
_TUPLE_KEYS = frozenset({
    "RAW_MASK_DILATE_KERNEL", "RAW_MASK_CLOSE_KERNEL", "RAW_MASK_CLOSE_KERNEL2",
    "MASK_DILATE_VERTICAL", "MASK_CLOSE_KERNEL_LARGE",
    "BEV_ALONG_ROAD_DILATE_KERNEL",
})


def _load_yaml_config() -> dict:
    """โหลด config.yaml จากโฟลเดอร์เดียวกับ config.py; raise ถ้าไม่มีไฟล์"""
    path = PROJECT_ROOT / "config.yaml"
    if not path.exists():
        raise FileNotFoundError(
            "config.yaml not found — please ensure config.yaml exists"
        )
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not data:
        raise ValueError("config.yaml is empty — please add configuration values")
    # Flatten nested: camera.CAM_W -> CAM_W
    flat = {}
    for section, values in data.items():
        if isinstance(values, dict):
            for k, v in values.items():
                flat[k] = v
    return flat


def _apply_config(flat: dict) -> None:
    """ตั้งค่า module-level จาก dict; แปลง list เป็น tuple ตาม _TUPLE_KEYS"""
    import sys
    mod = sys.modules[__name__]
    for k, v in flat.items():
        if k in _TUPLE_KEYS and isinstance(v, list):
            v = tuple(v)
        setattr(mod, k, v)


# โหลดค่าจาก YAML (source of truth เดียว — ไม่มี fallback)
_flat = _load_yaml_config()
_apply_config(_flat)

# Legacy alias
UNET_WEIGHT_BASE = UNET_CENTER_WEIGHT

DEFAULT_MODEL = PROJECT_ROOT / "model" / "lane_unet_final.pth"


# ── Performance Optimization ───────────────────────────────────────────────────
# Use CPU for operations where it's faster than GPU (current data sizes)
PERFORMANCE_USE_CPU_BEV_TRANSFORM = True      # CPU faster: 0.47ms vs 3.33ms
PERFORMANCE_USE_CPU_SLIDING_WINDOW = True     # CPU faster: 0.38ms vs 14.82ms
PERFORMANCE_USE_CPU_REFERENCE_GEN = True      # CPU faster: 0.03ms vs 0.03ms
PERFORMANCE_USE_GPU_MPC_EVAL = False          # CPU faster: 0.01ms vs 0.00ms (but CPU is fine)

# Enable GPU acceleration for future larger-scale processing
PERFORMANCE_GPU_READY = True                   # Infrastructure ready
PERFORMANCE_BATCH_SIZE = 1                     # Single frame processing
PERFORMANCE_ENABLE_LOGGING = False             # Performance debug logging


class UXColors:
    """สีสำหรับ UI (BGR สำหรับ cv2, RGB สำหรับ pygame)"""
    GREEN_VALID_BGR = (100, 255, 120)
    ORANGE_PATH_BGR = (0, 200, 255)
    RED_INVALID_BGR = (0, 150, 255)
    GREEN_VALID = (120, 255, 100)
    RED_INVALID = (255, 150, 150)
    TEXT_NORMAL = (200, 200, 200)
    TEXT_DIM = (120, 125, 140)
    SECTION_LABEL = (180, 180, 180)
