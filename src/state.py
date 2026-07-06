"""
State types ต่อ frame — หลักเหตุผล: ข้อมูลที่ไหลจาก Pipeline → Dashboard มี type ชัด
"""
from dataclasses import dataclass
from typing import Optional, List, Tuple, Any

import numpy as np


@dataclass
class FrameState:
    """
    State เดียวต่อ frame สำหรับทั้ง Control และ View.
    Pipeline สร้างตัวนี้แล้วส่งให้ Dashboard วาดอย่างเดียว (ไม่ยิง control จาก state โดยตรง)
    """
    rgb: np.ndarray
    speed_kmh: float
    steer: float
    throttle: float
    brake: float
    cte_m: float
    heading_rad: float
    curvature: float
    mode: str
    lane_conf: float = 0.0
    reference_path: Optional[List[Tuple[float, float]]] = None  # [(s_m, lat_m), ...]
    geometry_valid: bool = False
    lane_overlay: Optional[np.ndarray] = None
    mask: Optional[np.ndarray] = None
    bev_window_vis: Optional[np.ndarray] = None
    # Image-space left/right boundary points (N, 2) row,col for Raw mask segment display
    left_px_img: Optional[np.ndarray] = None
    right_px_img: Optional[np.ndarray] = None
    # Extended for dashboard (alg layer)
    road_symbol_result: Optional[Any] = None
    tracked: bool = False
    used_completion: bool = False
    # Back-to-basic phase report (P1..P5 ครบ = จับเลนและตีเส้นแบ่ง phase สำเร็จ)
    phase_p1_ok: bool = False  # Lane mask + confidence
    phase_p2_ok: bool = False  # Left/right boundary
    phase_p3_ok: bool = False  # Centerline
    phase_p4_ok: bool = False  # Lane state (cte, heading, curvature)
    phase_p5_ok: bool = False  # Lane phase/mode
    phase_p2_case: str = "none"   # both | mirror_left | mirror_right | completion | none
    phase_p3_case: str = "none"   # from_both | completion | none
    solver_status: str = "Solve_Succeeded"  # for eval: Solve_Succeeded | Fallback
    # Peter Moran visualization data
    bev_binary: Optional[np.ndarray] = None
    raw_windows_left: Optional[list] = None
    raw_windows_right: Optional[list] = None
    filt_windows_left: Optional[list] = None
    filt_windows_right: Optional[list] = None
    left_px_bev: Optional[np.ndarray] = None   # BEV-space lane points (N,2) row,col
    right_px_bev: Optional[np.ndarray] = None
    left_curvature_m: float = 9999.0
    right_curvature_m: float = 9999.0
    # ── ADAS status (full ADAS suite) ──────────────────────────────────────
    # LDW + LKA Pro
    ldw_state: str = "in_lane"          # in_lane | departing | departed
    ldw_warning_active: bool = False
    ldw_side: str = "none"              # left | right | none
    lka_pro_assist: float = 0.0         # steering assist (rad)
    # BSW + LCA
    bsw_left_alert: str = "clear"       # clear | blind_spot | approaching
    bsw_right_alert: str = "clear"
    bsw_safe_left: bool = True
    bsw_safe_right: bool = True
    # TSR + Traffic Light
    tsr_speed_limit_kmh: Optional[float] = None
    tsr_traffic_light: str = "unknown"  # red | yellow | green | unknown
    tsr_traffic_light_distance: float = -1.0
    tsr_action: str = "none"            # none | stop | slow | proceed
    # TJA + Stop & Go
    tja_state: str = "inactive"         # inactive | active | stopped | creeping | no_lead
    tja_active: bool = False
    stop_and_go_stopped: bool = False
    # AEB + ACC (already in safety/emergency_braking_adaptive_cruise_control.py, expose status here)
    aeb_active: bool = False
    aeb_ttc: float = -1.0
    aeb_warning_level: str = "none"     # none | caution | warning | critical
    acc_active: bool = False
    acc_target_speed_ms: float = -1.0
    acc_distance_m: float = -1.0
