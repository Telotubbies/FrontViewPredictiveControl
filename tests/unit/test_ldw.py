"""Comprehensive tests for Lane Departure Warning + LKA Pro (safety/ldw.py)."""
import math
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
for _p in (SRC, ROOT):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from safety.ldw import (
    LDWConfig,
    LDWState,
    LaneDepartureWarning,
    LKAPro,
)


# ── Helpers ──────────────────────────────────────────────────────────────────


def make_config(**overrides) -> LDWConfig:
    """Create an LDWConfig with optional overrides."""
    defaults = dict(
        departure_threshold_m=0.3,
        departure_hysteresis_m=0.1,
        warning_duration_s=2.0,
        lka_pro_assist_gain=0.5,
        blink_rate_hz=1.5,
    )
    defaults.update(overrides)
    return LDWConfig(**defaults)


def step(ldw: LaneDepartureWarning, cte, t, speed=10.0, heading_err=0.0):
    """Convenience: call update with sane defaults."""
    return ldw.update(cte, heading_err, speed, t)


# ── LDWConfig ────────────────────────────────────────────────────────────────


class TestLDWConfig:
    def test_defaults(self):
        cfg = LDWConfig()
        assert cfg.departure_threshold_m == 0.3
        assert cfg.departure_hysteresis_m == 0.1
        assert cfg.warning_duration_s == 2.0
        assert cfg.lka_pro_assist_gain == 0.5
        assert cfg.blink_rate_hz == 1.5

    def test_override(self):
        cfg = make_config(departure_threshold_m=0.5)
        assert cfg.departure_threshold_m == 0.5


# ── LDWState ─────────────────────────────────────────────────────────────────


class TestLDWState:
    def test_values(self):
        assert LDWState.IN_LANE == 0
        assert LDWState.DEPARTING == 1
        assert LDWState.DEPARTED == 2

    def test_is_intenum(self):
        assert int(LDWState.IN_LANE) == 0
        assert isinstance(LDWState.DEPARTING, int)


# ── State transitions ────────────────────────────────────────────────────────


class TestStateTransitions:
    """Test IN_LANE → DEPARTING → DEPARTED → IN_LANE with hysteresis."""

    def test_initial_state_is_in_lane(self):
        ldw = LaneDepartureWarning(make_config())
        assert ldw._state == LDWState.IN_LANE

    def test_in_lane_to_departing(self):
        ldw = LaneDepartureWarning(make_config())
        out = step(ldw, 0.0, 0.0)
        assert out["state"] == "IN_LANE"
        # cross threshold → DEPARTING
        out = step(ldw, 0.35, 0.1)
        assert out["state"] == "DEPARTING"
        assert out["warning_active"] is True

    def test_departing_to_departed(self):
        ldw = LaneDepartureWarning(make_config())
        step(ldw, 0.35, 0.0)  # DEPARTING
        out = step(ldw, 0.45, 0.1)  # > 0.3 + 0.1 = 0.4 → DEPARTED
        assert out["state"] == "DEPARTED"
        assert out["warning_active"] is True

    def test_departed_to_in_lane_hysteresis(self):
        """ต้องลดลงต่ำกว่า threshold - hysteresis = 0.2 ถึงจะกลับ IN_LANE."""
        ldw = LaneDepartureWarning(make_config())
        step(ldw, 0.35, 0.0)  # DEPARTING
        step(ldw, 0.45, 0.1)  # DEPARTED
        # ลดลงเหลือ 0.35 (ยัง > 0.2) → ยัง DEPARTED
        out = step(ldw, 0.35, 0.2)
        assert out["state"] == "DEPARTED"
        # ลดลงเหลือ 0.15 (< 0.2) → IN_LANE
        out = step(ldw, 0.15, 0.3)
        assert out["state"] == "IN_LANE"

    def test_departing_back_to_in_lane_hysteresis(self):
        """DEPARTING กลับ IN_LANE เมื่อ |cte| < threshold - hysteresis."""
        ldw = LaneDepartureWarning(make_config())
        step(ldw, 0.35, 0.0)  # DEPARTING
        # ยัง > 0.2 → ยัง DEPARTING
        out = step(ldw, 0.25, 0.1)
        assert out["state"] == "DEPARTING"
        # < 0.2 → IN_LANE
        out = step(ldw, 0.15, 0.2)
        assert out["state"] == "IN_LANE"

    def test_full_cycle(self):
        """Full cycle: IN_LANE → DEPARTING → DEPARTED → IN_LANE."""
        ldw = LaneDepartureWarning(make_config())
        t = 0.0
        step(ldw, 0.0, t)
        t += 0.1
        assert ldw._state == LDWState.IN_LANE
        step(ldw, 0.35, t)
        t += 0.1
        assert ldw._state == LDWState.DEPARTING
        step(ldw, 0.45, t)
        t += 0.1
        assert ldw._state == LDWState.DEPARTED
        step(ldw, 0.15, t)
        t += 0.1
        assert ldw._state == LDWState.IN_LANE


# ── Low speed ────────────────────────────────────────────────────────────────


class TestLowSpeed:
    def test_no_warning_below_5_ms(self):
        ldw = LaneDepartureWarning(make_config())
        out = step(ldw, 0.5, 0.0, speed=4.0)
        assert out["state"] == "IN_LANE"
        assert out["warning_active"] is False
        assert out["steering_assist"] == 0.0

    def test_no_warning_at_zero_speed(self):
        ldw = LaneDepartureWarning(make_config())
        out = step(ldw, 1.0, 0.0, speed=0.0)
        assert out["state"] == "IN_LANE"
        assert out["warning_active"] is False

    def test_warning_at_just_above_5_ms(self):
        ldw = LaneDepartureWarning(make_config())
        out = step(ldw, 0.4, 0.0, speed=5.1)
        assert out["state"] == "DEPARTING"
        assert out["warning_active"] is True

    def test_low_speed_resets_state(self):
        """ถ้าเคย DEPARTED แล้วความเร็วต่ำ → reset กลับ IN_LANE."""
        ldw = LaneDepartureWarning(make_config())
        step(ldw, 0.35, 0.0, speed=10.0)  # DEPARTING
        step(ldw, 0.45, 0.1, speed=10.0)  # DEPARTED
        assert ldw._state == LDWState.DEPARTED
        out = step(ldw, 0.45, 0.2, speed=3.0)
        assert out["state"] == "IN_LANE"


# ── Side detection ───────────────────────────────────────────────────────────


class TestSideDetection:
    def test_left_departure_positive_cte(self):
        ldw = LaneDepartureWarning(make_config())
        out = step(ldw, 0.35, 0.0)
        assert out["side"] == "left"

    def test_right_departure_negative_cte(self):
        ldw = LaneDepartureWarning(make_config())
        out = step(ldw, -0.35, 0.0)
        assert out["side"] == "right"

    def test_in_lane_side_none(self):
        ldw = LaneDepartureWarning(make_config())
        out = step(ldw, 0.1, 0.0)
        assert out["side"] == "none"

    def test_departed_left(self):
        ldw = LaneDepartureWarning(make_config())
        step(ldw, 0.35, 0.0)
        out = step(ldw, 0.45, 0.1)
        assert out["side"] == "left"

    def test_departed_right(self):
        ldw = LaneDepartureWarning(make_config())
        step(ldw, -0.35, 0.0)
        out = step(ldw, -0.45, 0.1)
        assert out["side"] == "right"


# ── Steering assist ──────────────────────────────────────────────────────────


class TestSteeringAssist:
    def test_assist_zero_in_lane(self):
        ldw = LaneDepartureWarning(make_config())
        out = step(ldw, 0.1, 0.0)
        assert out["steering_assist"] == 0.0

    def test_assist_corrective_direction_positive_cte(self):
        """cte บวก (ออกซ้าย) → assist ลบ (steer ขวากลับเข้าเลน)."""
        ldw = LaneDepartureWarning(make_config())
        out = step(ldw, 0.35, 0.0)
        assert out["steering_assist"] < 0.0

    def test_assist_corrective_direction_negative_cte(self):
        """cte ลบ (ออกขวา) → assist บวก (steer ซ้ายกลับเข้าเลน)."""
        ldw = LaneDepartureWarning(make_config())
        out = step(ldw, -0.35, 0.0)
        assert out["steering_assist"] > 0.0

    def test_assist_magnitude_clamped_at_threshold(self):
        """min(|cte|/threshold, 1.0) → ที่ cte=0.6 ratio=1.0 → assist = -gain."""
        ldw = LaneDepartureWarning(make_config())
        out = step(ldw, 0.6, 0.0)  # DEPARTING, ratio = min(0.6/0.3,1)=1.0
        expected = -math.copysign(0.5 * 1.0, 0.6)
        assert abs(out["steering_assist"] - expected) < 1e-9

    def test_assist_scales_with_cte(self):
        ldw = LaneDepartureWarning(make_config())
        out1 = step(ldw, 0.31, 0.0)  # ratio ~ 1.033 → clamped 1.0
        # reset and try smaller
        ldw2 = LaneDepartureWarning(make_config())
        out2 = step(ldw2, 0.305, 0.0)  # ratio ~ 1.016 → clamped 1.0
        # both clamped at 1.0 so equal
        assert abs(out1["steering_assist"] - out2["steering_assist"]) < 1e-9

    def test_assist_zero_at_low_speed(self):
        ldw = LaneDepartureWarning(make_config())
        out = step(ldw, 0.5, 0.0, speed=3.0)
        assert out["steering_assist"] == 0.0


# ── Blink toggle ─────────────────────────────────────────────────────────────


class TestBlinkToggle:
    def test_blink_off_in_lane(self):
        ldw = LaneDepartureWarning(make_config(blink_rate_hz=1.5))
        out = step(ldw, 0.1, 0.0)
        assert out["blink_on"] is False

    def test_blink_toggles_when_warning_active(self):
        """เมื่อ warning active, blink_on ควร toggle ตาม blink_rate."""
        cfg = make_config(blink_rate_hz=2.0)  # period = 0.5s, half = 0.25s
        ldw = LaneDepartureWarning(cfg)
        out = step(ldw, 0.4, 0.0)  # DEPARTING → warning active
        # first update dt=0 → blink stays False
        assert out["warning_active"] is True
        assert out["blink_on"] is False
        # advance time past half period (dt=0.3 > 0.25) → toggle to True
        out2 = step(ldw, 0.4, 0.3)
        assert out2["blink_on"] is True
        # advance again (dt=0.3 > 0.25) → toggle back to False
        out3 = step(ldw, 0.4, 0.6)
        assert out3["blink_on"] is False

    def test_blink_off_when_warning_clears(self):
        cfg = make_config()
        ldw = LaneDepartureWarning(cfg)
        step(ldw, 0.4, 0.0)  # DEPARTING
        # back to in lane
        out = step(ldw, 0.1, 0.1)
        # warning timer may still be active but blink should eventually be off
        # when warning_active becomes False
        # force back to in lane and let timer expire
        for i in range(30):
            out = step(ldw, 0.1, 0.1 + (i + 1) * 0.2)
        assert out["warning_active"] is False
        assert out["blink_on"] is False


# ── get_status ───────────────────────────────────────────────────────────────


class TestGetStatus:
    def test_status_in_lane(self):
        ldw = LaneDepartureWarning(make_config())
        step(ldw, 0.1, 0.0)
        s = ldw.get_status()
        assert s["state"] == "IN_LANE"
        assert s["side"] == "none"
        assert s["warning_active"] is False
        assert s["steering_assist"] == 0.0

    def test_status_departing(self):
        ldw = LaneDepartureWarning(make_config())
        step(ldw, 0.35, 0.0)
        s = ldw.get_status()
        assert s["state"] == "DEPARTING"
        assert s["side"] == "left"
        assert s["warning_active"] is True
        assert s["steering_assist"] != 0.0


# ── reset ────────────────────────────────────────────────────────────────────


class TestReset:
    def test_reset_returns_to_in_lane(self):
        ldw = LaneDepartureWarning(make_config())
        step(ldw, 0.35, 0.0)  # DEPARTING
        step(ldw, 0.45, 0.1)  # DEPARTED
        assert ldw._state == LDWState.DEPARTED
        ldw.reset()
        assert ldw._state == LDWState.IN_LANE
        assert ldw._warning_active is False
        assert ldw._side == "none"
        assert ldw._steering_assist == 0.0

    def test_reset_clears_blink(self):
        ldw = LaneDepartureWarning(make_config())
        step(ldw, 0.4, 0.0)
        step(ldw, 0.4, 0.3)
        ldw.reset()
        assert ldw._blink_on is False

    def test_reset_then_works_normally(self):
        ldw = LaneDepartureWarning(make_config())
        step(ldw, 0.45, 0.0)
        ldw.reset()
        out = step(ldw, 0.35, 0.0)
        assert out["state"] == "DEPARTING"


# ── LKAPro ───────────────────────────────────────────────────────────────────


class TestLKAPro:
    def test_zero_assist_in_lane(self):
        lka = LKAPro(make_config())
        a = lka.compute_assist(0.1, 0.0, 10.0, 0.0)
        # EMA from 0: a = 0.3*0 + 0.7*0 = 0
        assert a == 0.0

    def test_assist_when_departing(self):
        lka = LKAPro(make_config())
        a = lka.compute_assist(0.5, 0.0, 10.0, 0.0)
        # raw = -sign(0.5)*0.5*min(0.5/0.3,2) = -0.5*1.6667 = -0.8333
        # EMA: 0.3*(-0.8333) + 0.7*0 = -0.25
        expected = 0.3 * (-0.5 * min(0.5 / 0.3, 2.0))
        assert abs(a - expected) < 1e-9

    def test_assist_corrective_direction_positive_cte(self):
        lka = LKAPro(make_config())
        a = lka.compute_assist(0.5, 0.0, 10.0, 0.0)
        assert a < 0.0  # steer back right

    def test_assist_corrective_direction_negative_cte(self):
        lka = LKAPro(make_config())
        a = lka.compute_assist(-0.5, 0.0, 10.0, 0.0)
        assert a > 0.0  # steer back left

    def test_assist_zero_low_speed(self):
        lka = LKAPro(make_config())
        a = lka.compute_assist(0.5, 0.0, 3.0, 0.0)
        assert a == 0.0

    def test_assist_clamped_at_2x(self):
        """min(|cte|/threshold, 2.0) → ที่ cte=1.0 ratio=2.0 (clamped)."""
        lka = LKAPro(make_config())
        a_big = lka.compute_assist(1.0, 0.0, 10.0, 0.0)
        lka2 = LKAPro(make_config())
        a_huge = lka2.compute_assist(5.0, 0.0, 10.0, 0.0)
        # both should have same raw magnitude (clamped at 2.0)
        expected_raw = -0.5 * 2.0
        assert abs(a_big - 0.3 * expected_raw) < 1e-9
        assert abs(a_huge - 0.3 * expected_raw) < 1e-9

    def test_ema_smoothing(self):
        """Assist should smooth over multiple steps."""
        lka = LKAPro(make_config())
        a1 = lka.compute_assist(0.5, 0.0, 10.0, 0.0)
        a2 = lka.compute_assist(0.5, 0.0, 10.0, 0.1)
        # a2 should be closer to raw than a1 (less extreme)
        raw = -0.5 * min(0.5 / 0.3, 2.0)
        assert abs(a2 - raw) < abs(a1 - raw)
        # a2 = 0.3*raw + 0.7*a1
        expected = 0.3 * raw + 0.7 * a1
        assert abs(a2 - expected) < 1e-9

    def test_last_assist_stored(self):
        lka = LKAPro(make_config())
        a = lka.compute_assist(0.5, 0.0, 10.0, 0.0)
        assert lka._last_assist == a

    def test_get_debug_info(self):
        lka = LKAPro(make_config())
        lka.compute_assist(0.4, 0.0, 10.0, 0.0)
        info = lka.get_debug_info()
        assert info["last_cte"] == 0.4
        assert "last_assist" in info
        assert "last_state" in info

    def test_debug_info_state_reflects_ldw(self):
        lka = LKAPro(make_config())
        lka.compute_assist(0.35, 0.0, 10.0, 0.0)  # DEPARTING
        lka.compute_assist(0.45, 0.0, 10.0, 0.1)  # DEPARTED
        info = lka.get_debug_info()
        assert info["last_state"] == "DEPARTED"

    def test_zero_assist_at_exact_threshold(self):
        """|cte| == threshold → not > threshold → no assist."""
        lka = LKAPro(make_config())
        a = lka.compute_assist(0.3, 0.0, 10.0, 0.0)
        assert a == 0.0


# ── Config validation ──────────────────────────────────────────────────────────


class TestConfigValidation:
    def test_threshold_zero_raises(self):
        with pytest.raises(ValueError, match="departure_threshold_m must be > 0"):
            make_config(departure_threshold_m=0.0)

    def test_threshold_negative_raises(self):
        with pytest.raises(ValueError, match="departure_threshold_m must be > 0"):
            make_config(departure_threshold_m=-0.1)

    def test_hysteresis_ge_threshold_raises(self):
        """hysteresis >= threshold should raise."""
        with pytest.raises(ValueError, match="departure_hysteresis_m must be <"):
            make_config(departure_threshold_m=0.3, departure_hysteresis_m=0.3)

    def test_hysteresis_gt_threshold_raises(self):
        with pytest.raises(ValueError, match="departure_hysteresis_m must be <"):
            make_config(departure_threshold_m=0.3, departure_hysteresis_m=0.5)

    def test_negative_blink_rate_raises(self):
        with pytest.raises(ValueError, match="blink_rate_hz must be >= 0"):
            make_config(blink_rate_hz=-1.0)

    def test_zero_blink_rate_ok(self):
        """blink_rate_hz == 0 is valid (handled via max(…, 1e-6) in update)."""
        cfg = make_config(blink_rate_hz=0.0)
        assert cfg.blink_rate_hz == 0.0

    def test_valid_config_no_raise(self):
        """Standard config should not raise."""
        cfg = make_config()
        assert cfg.departure_threshold_m == 0.3


# ── cte=0 side detection ───────────────────────────────────────────────────────


class TestCteZeroSideDetection:
    def test_cte_zero_in_lane_returns_none(self):
        """cte=0 → stays IN_LANE → side 'none'."""
        ldw = LaneDepartureWarning(make_config())
        out = step(ldw, 0.0, 0.0)
        assert out["state"] == "IN_LANE"
        assert out["side"] == "none"

    def test_cte_zero_departing_returns_none(self):
        """If already DEPARTING and cte drops to exactly 0, side should be 'none'."""
        ldw = LaneDepartureWarning(make_config())
        step(ldw, 0.35, 0.0)  # DEPARTING
        assert ldw._state == LDWState.DEPARTING
        # cte=0 → |cte|=0 < threshold - hysteresis (0.2) → back to IN_LANE
        out = step(ldw, 0.0, 0.1)
        assert out["side"] == "none"

    def test_cte_zero_departed_returns_none(self):
        """If DEPARTED and cte becomes 0, side should be 'none' not 'right'."""
        ldw = LaneDepartureWarning(make_config())
        step(ldw, 0.35, 0.0)   # DEPARTING
        step(ldw, 0.45, 0.1)   # DEPARTED
        assert ldw._state == LDWState.DEPARTED
        # cte=0 → |cte|=0 < 0.2 → back to IN_LANE
        out = step(ldw, 0.0, 0.2)
        assert out["side"] == "none"


# ── Timestamp going backwards ─────────────────────────────────────────────────


class TestTimestampBackwards:
    def test_dt_clamped_to_zero_when_timestamp_goes_backwards(self):
        """When timestamp decreases, dt should be clamped to 0 (no negative dt)."""
        ldw = LaneDepartureWarning(make_config())
        step(ldw, 0.4, 10.0)   # DEPARTING, warning active
        # timestamp goes backwards from 10.0 to 5.0
        out = step(ldw, 0.4, 5.0)
        # dt clamped to 0 → blink should not have toggled
        assert out["state"] == "DEPARTING"
        assert out["warning_active"] is True
        # blink_phase should not have advanced (dt=0)
        assert ldw._blink_phase_s == 0.0

    def test_backwards_timestamp_no_negative_timer(self):
        """Warning timer should never go negative from backwards timestamp."""
        ldw = LaneDepartureWarning(make_config())
        step(ldw, 0.35, 0.0)   # DEPARTING
        step(ldw, 0.45, 0.1)   # DEPARTED, timer set to 2.0 then decremented by dt=0.1 → 1.9
        timer_after_departed = ldw._warning_timer_s
        assert timer_after_departed == pytest.approx(1.9)
        # timestamp goes backwards → dt clamped to 0 → timer unchanged
        step(ldw, 0.45, 0.05)
        assert ldw._warning_timer_s >= 0.0
        assert ldw._warning_timer_s == pytest.approx(timer_after_departed)


# ── Exact boundary conditions ─────────────────────────────────────────────────


class TestBoundaryConditions:
    def test_cte_equals_threshold_stays_in_lane(self):
        """|cte| == threshold (not > threshold) → stays IN_LANE."""
        ldw = LaneDepartureWarning(make_config())
        out = step(ldw, 0.3, 0.0)  # threshold = 0.3, not > 0.3
        assert out["state"] == "IN_LANE"
        assert out["warning_active"] is False

    def test_cte_equals_threshold_plus_hysteresis_stays_departing(self):
        """|cte| == threshold + hysteresis (not >) → stays DEPARTING."""
        ldw = LaneDepartureWarning(make_config())
        step(ldw, 0.35, 0.0)  # DEPARTING
        # threshold + hysteresis = 0.3 + 0.1 = 0.4, not > 0.4
        out = step(ldw, 0.4, 0.1)
        assert out["state"] == "DEPARTING"

    def test_cte_equals_threshold_minus_hysteresis_stays_departing(self):
        """|cte| == threshold - hysteresis (not <) → stays DEPARTING."""
        ldw = LaneDepartureWarning(make_config())
        step(ldw, 0.35, 0.0)  # DEPARTING
        # threshold - hysteresis = 0.3 - 0.1 = 0.2, not < 0.2
        out = step(ldw, 0.2, 0.1)
        assert out["state"] == "DEPARTING"

    def test_cte_equals_threshold_minus_hysteresis_stays_departed(self):
        """|cte| == threshold - hysteresis (not <) → stays DEPARTED."""
        ldw = LaneDepartureWarning(make_config())
        step(ldw, 0.35, 0.0)  # DEPARTING
        step(ldw, 0.45, 0.1)  # DEPARTED
        # 0.2 is not < 0.2 → stays DEPARTED
        out = step(ldw, 0.2, 0.2)
        assert out["state"] == "DEPARTED"

    def test_cte_just_above_threshold_transitions(self):
        """|cte| slightly > threshold → DEPARTING."""
        ldw = LaneDepartureWarning(make_config())
        out = step(ldw, 0.3001, 0.0)
        assert out["state"] == "DEPARTING"

    def test_cte_just_above_threshold_plus_hysteresis_transitions(self):
        """|cte| slightly > threshold + hysteresis → DEPARTED."""
        ldw = LaneDepartureWarning(make_config())
        step(ldw, 0.35, 0.0)  # DEPARTING
        out = step(ldw, 0.4001, 0.1)
        assert out["state"] == "DEPARTED"

    def test_cte_just_below_threshold_minus_hysteresis_returns_in_lane(self):
        """|cte| slightly < threshold - hysteresis → IN_LANE from DEPARTING."""
        ldw = LaneDepartureWarning(make_config())
        step(ldw, 0.35, 0.0)  # DEPARTING
        out = step(ldw, 0.199, 0.1)  # < 0.2
        assert out["state"] == "IN_LANE"
