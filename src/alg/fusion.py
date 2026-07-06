"""
Fusion: WP เป็นหลัก; UNet ช่วยจัดกลาง (CTE) และเช็คเลน (lane_conf).

- Heading/curvature: จาก waypoint เสมอเมื่อมี wp_state
- CTE: จาก WP เป็นหลัก; เมื่อ lane_conf สูง (เช็คเลนผ่าน) ใช้ UNet ช่วยจัดกลาง
- Mode: WP_PRIMARY (ไม่ใช้ UNet), WP+UNET_CENTER (ใช้ UNet ช่วยจัดกลาง)
"""
from typing import Optional, Tuple

import numpy as np

from config import (
    CTE_LANE_HALF_WIDTH_M,
    MAX_CURVATURE_REF,
    FUSION_CONF_LOW,
    FUSION_CONF_HIGH,
    FUSION_CURVATURE_UNET_REDUCE_THRESHOLD,
    FUSION_CURVATURE_UNET_REDUCE_SCALE,
    UNET_CENTER_WEIGHT,
)


def apply_fusion(
    wp_state: Optional[Tuple[float, float, float]],
    cte_m_lane: float,
    head_s: float,
    curv_s: float,
    lane_conf: float,
    geometry_valid: Optional[bool] = None,
) -> Tuple[float, float, float, str]:
    """
    Returns: (cte_m, head_s, curv_s, mode).

    WP primary: heading/curvature จาก WP เสมอเมื่อมี wp_state.
    UNet assist: ใช้ UNet ช่วย CTE (จัดกลาง) เฉพาะเมื่อ lane_conf สูงพอ (เช็คเลนผ่าน).
    โค้ง: ลด w_unet แบบ adaptive ตาม |curv|; เมื่อ geometry_valid=False และโค้งแรง → WP only.
    """
    if wp_state is None:
        cte_m = cte_m_lane
        mode = "LANE+MPC" if lane_conf >= FUSION_CONF_LOW else "LANE+MPC"
    else:
        cte_m_wp, head_wp, curv_wp = wp_state
        # CTE/Heading/Curvature: blend WP กับ perception ตาม lane_conf
        if lane_conf < FUSION_CONF_LOW:
            w_unet = 0.0
            mode = "WP_PRIMARY"
        else:
            # Blend perception เมื่อ lane_conf สูง
            blend = (lane_conf - FUSION_CONF_LOW) / max(FUSION_CONF_HIGH - FUSION_CONF_LOW, 1e-6)
            blend = min(1.0, blend)
            w_unet = UNET_CENTER_WEIGHT * blend
            # โค้ง: ลดน้ำหนัก UNet (adaptive); ในโค้งแรง + geometry invalid → WP only
            curv_abs = max(abs(curv_s), abs(curv_wp))
            if curv_abs > FUSION_CURVATURE_UNET_REDUCE_THRESHOLD:
                scale = max(1e-6, FUSION_CURVATURE_UNET_REDUCE_SCALE)
                w_unet *= max(0.0, 1.0 - (curv_abs - FUSION_CURVATURE_UNET_REDUCE_THRESHOLD) / scale)
            if geometry_valid is False and curv_abs > FUSION_CURVATURE_UNET_REDUCE_THRESHOLD:
                w_unet = 0.0
            mode = "WP+UNET_CENTER" if w_unet > 0.05 else "WP_PRIMARY"
        # Blend all: CTE, heading, curvature
        cte_m = (1.0 - w_unet) * cte_m_wp + w_unet * cte_m_lane
        head_s = (1.0 - w_unet) * head_wp + w_unet * head_s
        curv_s = (1.0 - w_unet) * curv_wp + w_unet * curv_s

    cte_m = float(np.clip(cte_m, -CTE_LANE_HALF_WIDTH_M, CTE_LANE_HALF_WIDTH_M))
    head_s = float(np.clip(head_s, -0.25, 0.25))
    curv_s = float(np.clip(curv_s, -MAX_CURVATURE_REF, MAX_CURVATURE_REF))
    return cte_m, head_s, curv_s, mode
