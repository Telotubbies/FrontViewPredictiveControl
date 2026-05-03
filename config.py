"""
LKA Pipeline — ค่าคงที่โหลดจาก config.yaml (ตั้งค่าที่เดียว)
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
    """โหลด config.yaml จากโฟลเดอร์เดียวกับ config.py"""
    path = PROJECT_ROOT / "config.yaml"
    if not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not data:
        return {}
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


# โหลดค่าจาก YAML
_flat = _load_yaml_config()
if _flat:
    _apply_config(_flat)
    import sys
    _mod = sys.modules[__name__]
    if not hasattr(_mod, "LANE_OVERLAY_ROAD_FORM"):
        setattr(_mod, "LANE_OVERLAY_ROAD_FORM", False)
    if not hasattr(_mod, "LANE_OVERLAY_STRIP_HALF_WIDTH_M"):
        setattr(_mod, "LANE_OVERLAY_STRIP_HALF_WIDTH_M", 0.9)
    if not hasattr(_mod, "LANE_OVERLAY_STRAIGHT_NEAR_M"):
        setattr(_mod, "LANE_OVERLAY_STRAIGHT_NEAR_M", 6.0)
    if not hasattr(_mod, "PLANNING_VIEW_RADIUS_M"):
        setattr(_mod, "PLANNING_VIEW_RADIUS_M", 500.0)
    for _k in ("LANE_FRAME_WIDTH_MIN_M", "LANE_FRAME_WIDTH_MAX_M", "LANE_FRAME_WIDTH_NOMINAL_M",
               "LANE_FRAME_WIDTH_EMA_ALPHA", "LANE_OVERLAY_USE_FRAME_WIDTH", "LANE_OVERLAY_TRAPEZOID"):
        if not hasattr(_mod, _k):
            _fw = {"LANE_FRAME_WIDTH_MIN_M": 2.0, "LANE_FRAME_WIDTH_MAX_M": 5.5,
                   "LANE_FRAME_WIDTH_NOMINAL_M": 3.5, "LANE_FRAME_WIDTH_EMA_ALPHA": 0.6,
                   "LANE_OVERLAY_USE_FRAME_WIDTH": True, "LANE_OVERLAY_TRAPEZOID": False}
            setattr(_mod, _k, _fw.get(_k))
    for _k in ("MPC_N", "MPC_DT", "MPC_L", "MPC_MAX_STEER", "MPC_MAX_STEER_RATE",
               "MPC_MAX_ACCEL", "MPC_MIN_ACCEL", "MPC_W_CTE", "MPC_W_HEADING",
               "MPC_W_VEL", "MPC_W_STEER", "MPC_W_ACCEL", "MPC_W_STEER_RATE",
               "MPC_W_ACCEL_RATE", "MPC_W_STEER_JERK", "MPC_USE_ADAPTIVE_WEIGHTS"):
        if not hasattr(_mod, _k):
            _defaults = {
                "MPC_N": 10, "MPC_DT": 0.1, "MPC_L": 2.875,
                "MPC_MAX_STEER": 0.7, "MPC_MAX_STEER_RATE": 0.2,
                "MPC_MAX_ACCEL": 3.0, "MPC_MIN_ACCEL": -5.0,
                "MPC_W_CTE": 200.0, "MPC_W_HEADING": 85.0, "MPC_W_VEL": 5.0,
                "MPC_W_STEER": 80.0, "MPC_W_ACCEL": 10.0,
                "MPC_W_STEER_RATE": 500.0, "MPC_W_ACCEL_RATE": 10.0,
                "MPC_W_STEER_JERK": 80.0, "MPC_USE_ADAPTIVE_WEIGHTS": True,
            }
            setattr(_mod, _k, _defaults.get(_k))
    for _k in ("STEER_MAX_DELTA_PER_FRAME", "STEER_RATE_LIMIT_SPEED_KMH", "STEER_MAX_DELTA_HIGH_SPEED"):
        if not hasattr(_mod, _k):
            _steer = {"STEER_MAX_DELTA_PER_FRAME": 0.12, "STEER_RATE_LIMIT_SPEED_KMH": 40.0,
                      "STEER_MAX_DELTA_HIGH_SPEED": 0.06}
            setattr(_mod, _k, _steer.get(_k))
    if not hasattr(_mod, "UNET_INPUT_W_INFER"):
        setattr(_mod, "UNET_INPUT_W_INFER", getattr(_mod, "UNET_INPUT_W", 640))
    if not hasattr(_mod, "UNET_INPUT_H_INFER"):
        setattr(_mod, "UNET_INPUT_H_INFER", getattr(_mod, "UNET_INPUT_H", 480))
    if not hasattr(_mod, "PERCEPTION_SKIP_FRAME"):
        setattr(_mod, "PERCEPTION_SKIP_FRAME", 1)
    if not hasattr(_mod, "PERCEPTION_LIGHTWEIGHT_VIS"):
        setattr(_mod, "PERCEPTION_LIGHTWEIGHT_VIS", False)
    if not hasattr(_mod, "MAIN_LOOP_SLEEP_S"):
        setattr(_mod, "MAIN_LOOP_SLEEP_S", 0.01)
    if not hasattr(_mod, "DASHBOARD_DRAW_EVERY_N"):
        setattr(_mod, "DASHBOARD_DRAW_EVERY_N", 1)
    if not hasattr(_mod, "DISPLAY_PATH_CACHE_INTERVAL"):
        setattr(_mod, "DISPLAY_PATH_CACHE_INTERVAL", 5)
else:
    # Fallback ถ้าไม่มี config.yaml — ใช้ค่าเดิม (ให้ import ยังทำงาน)
    CAM_FOV_DEG = 90.0
    CAM_W = 640
    CAM_H = 480
    CTE_TO_METERS = 3.5
    UNET_INPUT_W = 640
    UNET_INPUT_H = 480
    TARGET_SPEED_KMH = 25.0
    NUM_WP = 40
    WP_STEP = 2.0
    WP_LOOKAHEAD = 8.0
    WP_CTE_WEIGHT = 0.40
    UNET_CENTER_WEIGHT = 0.60
    LANE_CONF_THRESHOLD = 0.45
    WP_DIRECTION_WEIGHT = 1.0
    ROI_BOTTOM_RATIO = 0.55
    ROI_TOP_PENALTY_RATIO = 0.35
    ROAD_TOP_RATIO = 0.58
    BEV_LANE_SEARCH_TOP_RATIO = 0.25
    BEV_HISTOGRAM_WEIGHT_FAR = 1.5
    BEV_HISTOGRAM_WEIGHT_NEAR = 0.5
    BEV_WARP_INTERPOLATION = "nearest"
    BEV_RIGHTMOST_PEAK = True
    BEV_RIGHT_PEAK_MIN_RATIO = 0.25
    BEV_ALONG_ROAD_DILATE_KERNEL = (3, 5)
    LANE_CONF_ROI_TOP_RATIO = 0.30
    LANE_CONF_CONTINUITY_ALPHA = 0.4
    LANE_CONTINUITY_MIN_AREA_PX = 50
    RAW_MASK_DILATE_KERNEL = (3, 3)
    RAW_MASK_CLOSE_KERNEL = (5, 5)
    RAW_MASK_CLOSE_KERNEL2 = (7, 7)
    MASK_DILATE_VERTICAL = (3, 14)
    MASK_CLOSE_KERNEL_LARGE = (15, 8)
    DEBUG_MASK_DIR = ""
    DEBUG_MASK_SAVE_EVERY_N = 30
    ROAD_MARGIN_PX = 8
    MIN_LANE_WIDTH_PX = 12
    MAX_LANE_WIDTH_PX = 280
    CENTER_LANE_WIDTH_RATIO = 0.65
    LANE_MEMORY_MARGIN_PX = 55
    LANE_EXTEND_ROWS_UP = 72
    LANE_FILL_STEP = 1
    DASHED_LANE_MIN_PIXELS = 5
    MIN_LANE_WIDTH_M = 2.0
    MAX_LANE_WIDTH_M = 5.5
    LANE_OVERLAY_ROAD_FORM = False
    LANE_OVERLAY_STRIP_HALF_WIDTH_M = 0.9
    LANE_OVERLAY_STRAIGHT_NEAR_M = 6.0
    LANE_FRAME_WIDTH_MIN_M = 2.0
    LANE_FRAME_WIDTH_MAX_M = 5.5
    LANE_FRAME_WIDTH_NOMINAL_M = 3.5
    LANE_FRAME_WIDTH_EMA_ALPHA = 0.6
    LANE_OVERLAY_USE_FRAME_WIDTH = True
    LANE_OVERLAY_TRAPEZOID = False
    MPC_N = 10
    MPC_DT = 0.1
    MPC_L = 2.875
    MPC_MAX_STEER = 0.7
    MPC_MAX_STEER_RATE = 0.2
    MPC_MAX_ACCEL = 3.0
    MPC_MIN_ACCEL = -5.0
    MPC_W_CTE = 200.0
    MPC_W_HEADING = 85.0
    MPC_W_VEL = 5.0
    MPC_W_STEER = 80.0
    MPC_W_ACCEL = 10.0
    MPC_W_STEER_RATE = 500.0
    MPC_W_ACCEL_RATE = 10.0
    MPC_W_STEER_JERK = 80.0
    MPC_USE_ADAPTIVE_WEIGHTS = True
    CTE_LANE_HALF_WIDTH_M = 0.55
    MAX_CURVATURE_REF = 0.035
    STEER_SMOOTH_ALPHA = 0.38
    STEER_MAX_DELTA_PER_FRAME = 0.12
    STEER_RATE_LIMIT_SPEED_KMH = 40.0
    STEER_MAX_DELTA_HIGH_SPEED = 0.06
    USE_TRAJECTORY_PIPELINE = True
    LANE_EMA_ALPHA = 0.38
    STATE_EMA_ALPHA = 0.35
    REF_PATH_LOOKAHEAD_M = 35.0
    REF_PATH_NUM_PTS = 70
    REF_PATH_DISPLAY_LOOKAHEAD_M = 200.0
    REF_PATH_DISPLAY_NUM_PTS = 200
    PLANNING_VIEW_RADIUS_M = 500.0
    TURN_LOOKAHEAD_MIN_M = 12.0
    TURN_LOOKAHEAD_CURV_THRESHOLD = 0.02
    PERCEPTION_PATH_MIN_CONF = 0.5
    REF_PATH_LAT_SANITY_M = 5.0
    PERCEPTION_PATH_SMOOTH_ALPHA = 0.6
    CURVATURE_WP_EMA_ALPHA = 0.5
    LANE_HALF_WIDTH_DRAW_M = 0.9
    LANE_OVERLAY_DISPLAY_MARGIN_M = 0.2
    LANE_OVERLAY_MIN_DISPLAY_WIDTH_M = 2.0
    LANE_OVERLAY_MAX_WIDTH_DRAW_M = 6.0
    LOOKAHEAD_MIN_M = 8.0
    LOOKAHEAD_TIME_S = 2.2
    FUSION_CONF_HIGH = 0.85
    FUSION_CONF_LOW = 0.55
    FUSION_CURVATURE_UNET_REDUCE_THRESHOLD = 0.025
    FUSION_CURVATURE_UNET_REDUCE_SCALE = 0.02
    REF_PATH_DISPLAY_EMA_ALPHA = 0.82
    BEV_PX_PER_M = 35
    BEV_W = 450
    BEV_H = 320
    VIEW3D_W = 350
    VIEW3D_H = 200
    BEV_WIN_W = 350
    BEV_WIN_H = 200
    PANEL_W = 450
    PANEL_H = 320
    USE_PERCEPTION_THREAD = True
    CONTROL_HZ = 10
    STALE_PERCEPTION_S = 0.2
    UNET_INPUT_W_INFER = 640
    UNET_INPUT_H_INFER = 480
    PERCEPTION_SKIP_FRAME = 1
    PERCEPTION_LIGHTWEIGHT_VIS = False
    MAIN_LOOP_SLEEP_S = 0.01
    DASHBOARD_DRAW_EVERY_N = 1
    SAFETY_MAX_STEER_RAD = 0.45

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
