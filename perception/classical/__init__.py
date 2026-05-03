"""
Classical lane detection — ไม่ใช้ ML.

Edge → perspective warp → sliding window → polynomial fit.
Output: cte_m, heading_rad, curvature for MPC.
ใช้กับ: ./scripts/run.sh classical หรือ ./scripts/run_lka.sh classical
"""

from perception.classical.detector import ClassicalLane

__all__ = ["ClassicalLane"]
