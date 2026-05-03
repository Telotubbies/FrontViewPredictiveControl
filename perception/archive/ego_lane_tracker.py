"""
Ego-lane selection + temporal lane ID tracking.

Why: ไม่มี lane ID tracking → frame นี้เลือก left/right อีกคู่ → ข้ามเลน
→ เลือกคู่ที่ "อยู่กลาง (ego)" และ match กับ frame ก่อน (temporal consistency)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np

from .lane_clustering import LaneCandidate, cluster_lane_candidates

logger = logging.getLogger(__name__)

# Ego lane: prefer pair whose center at x=0 is closest to 0
LANE_WIDTH_NOMINAL_M = 3.6
MAX_PAIR_WIDTH_M = 5.5
MIN_PAIR_WIDTH_M = 2.2
# จับเดิมดีขึ้น: ยอม lateral ต่างจาก frame ก่อนได้มากขึ้น (ไม่สลับเลนง่าย)
LATERAL_MATCH_THRESH_M = 1.8
# คะแนนขั้นต่ำของคู่ที่ match กับของเดิม (ถ้า match ได้ ให้ยอม score ต่ำกว่าได้)
MIN_SCORE_TRACKED_PAIR = 0.15
COMPLETION_CONFIDENCE = 0.5


@dataclass
class EgoLaneResult:
    """Stable ego lane from selection + temporal tracking."""
    left_coeffs: Optional[np.ndarray]
    right_coeffs: Optional[np.ndarray]
    center_coeffs: Optional[np.ndarray]
    confidence: float
    tracked: bool  # True if matched to previous frame
    used_completion: bool  # True if used temporal completion (no detection)


class EgoLaneTracker:
    """
    Select ego lane from candidates and track across frames by matching lateral position.
    When detection fails, complete from previous frame (temporal completion).
    """

    def __init__(
        self,
        lane_width_nominal_m: float = LANE_WIDTH_NOMINAL_M,
        max_pair_width_m: float = MAX_PAIR_WIDTH_M,
        min_pair_width_m: float = MIN_PAIR_WIDTH_M,
        lateral_match_thresh_m: float = LATERAL_MATCH_THRESH_M,
    ):
        self.lane_width_nominal_m = lane_width_nominal_m
        self.max_pair_width_m = max_pair_width_m
        self.min_pair_width_m = min_pair_width_m
        self.lateral_match_thresh_m = lateral_match_thresh_m
        self._prev_left: Optional[np.ndarray] = None
        self._prev_right: Optional[np.ndarray] = None
        self._prev_center: Optional[np.ndarray] = None

    def reset(self) -> None:
        self._prev_left = None
        self._prev_right = None
        self._prev_center = None

    def _select_ego_pair(
        self,
        candidates: List[LaneCandidate],
    ) -> Optional[Tuple[LaneCandidate, LaneCandidate]]:
        """Select best left+right pair. จับเดิมดีขึ้น: ถ้ามีคู่ที่ match frame ก่อน → เลือกคู่นั้นก่อน."""
        left_cands = [c for c in candidates if c.is_left]
        right_cands = [c for c in candidates if not c.is_left]
        if not left_cands or not right_cands:
            return None

        pairs_tracked: List[Tuple[LaneCandidate, LaneCandidate, float]] = []
        pairs_new: List[Tuple[LaneCandidate, LaneCandidate, float]] = []

        for lc in left_cands:
            for rc in right_cands:
                width = rc.lateral_at_ego_m - lc.lateral_at_ego_m
                if width < self.min_pair_width_m or width > self.max_pair_width_m:
                    continue
                center = (lc.lateral_at_ego_m + rc.lateral_at_ego_m) / 2.0
                center_err = abs(center)
                combined_score = lc.score + rc.score
                left_coeffs = lc.coeffs
                right_coeffs = rc.coeffs
                if self._match_to_previous(left_coeffs, right_coeffs):
                    if combined_score >= MIN_SCORE_TRACKED_PAIR:
                        pairs_tracked.append((lc, rc, center_err))
                else:
                    if combined_score > 0.3:
                        pairs_new.append((lc, rc, center_err))

        # เลือกคู่ที่ match กับของเดิมก่อน (จับเดิมดีขึ้น)
        if pairs_tracked:
            pairs_tracked.sort(key=lambda x: x[2])
            return (pairs_tracked[0][0], pairs_tracked[0][1])
        if pairs_new:
            pairs_new.sort(key=lambda x: x[2])
            return (pairs_new[0][0], pairs_new[0][1])
        return None

    def _match_to_previous(
        self,
        left_coeffs: np.ndarray,
        right_coeffs: np.ndarray,
    ) -> bool:
        """Check if (left, right) match previous frame (lateral distance at x=0)."""
        if self._prev_left is None or self._prev_right is None:
            return False
        prev_left_y = float(np.polyval(self._prev_left, 0.0))
        prev_right_y = float(np.polyval(self._prev_right, 0.0))
        curr_left_y = float(np.polyval(left_coeffs, 0.0))
        curr_right_y = float(np.polyval(right_coeffs, 0.0))
        if abs(curr_left_y - prev_left_y) <= self.lateral_match_thresh_m and \
           abs(curr_right_y - prev_right_y) <= self.lateral_match_thresh_m:
            return True
        return False

    def update(
        self,
        bev_binary: np.ndarray,
        lookahead_m: float = 20.0,
        half_width_m: float = 10.0,
    ) -> EgoLaneResult:
        """
        Run clustering, select ego pair, track IDs. If no detection, return previous (completion).
        """
        candidates = cluster_lane_candidates(
            bev_binary,
            lookahead_m=lookahead_m,
            half_width_m=half_width_m,
        )
        pair = self._select_ego_pair(candidates)

        if pair is not None:
            lc, rc = pair
            left_coeffs = lc.coeffs
            right_coeffs = rc.coeffs
            center_coeffs = (left_coeffs + right_coeffs) / 2.0
            tracked = self._match_to_previous(left_coeffs, right_coeffs)
            conf = (lc.score + rc.score) / 2.0
            self._prev_left = left_coeffs.copy()
            self._prev_right = right_coeffs.copy()
            self._prev_center = center_coeffs.copy()
            return EgoLaneResult(
                left_coeffs=left_coeffs,
                right_coeffs=right_coeffs,
                center_coeffs=center_coeffs,
                confidence=conf,
                tracked=tracked,
                used_completion=False,
            )

        if self._prev_center is not None:
            return EgoLaneResult(
                left_coeffs=self._prev_left,
                right_coeffs=self._prev_right,
                center_coeffs=self._prev_center,
                confidence=COMPLETION_CONFIDENCE,
                tracked=True,
                used_completion=True,
            )

        return EgoLaneResult(
            left_coeffs=None,
            right_coeffs=None,
            center_coeffs=None,
            confidence=0.0,
            tracked=False,
            used_completion=False,
        )
