"""
Stuck vehicle detection and recovery (brake → reverse → forward).

Used when the vehicle is stationary with throttle applied for an extended period.
"""
import logging
from math import sin
from typing import Optional, Tuple

logger = logging.getLogger(__name__)


class StuckRecovery:
    """Detect stuck condition and execute brake → reverse → forward recovery."""

    STUCK_THRESHOLD = 280
    STUCK_CONFIRM_FRAMES = 5
    STUCK_SPEED_MS = 0.18
    STUCK_THROTTLE_MIN = 0.35
    COOLDOWN_AFTER_RECOVERY = 100
    BRAKE_FRAMES = 12
    REVERSE_FRAMES = 40
    MAX_RECOVERY_ATTEMPTS = 3
    PROGRESS_SPEED_MS = 0.5

    def __init__(self) -> None:
        self._n = 0
        self._confirm = 0
        self._phase = "idle"
        self._counter = 0
        self.just_recovered = False
        self._cooldown = 0
        self._recovery_count = 0

    def update(
        self, speed: float, throttle: float
    ) -> Optional[Tuple[float, float, float, bool]]:
        """
        Returns (steer, throttle, brake, reverse) when in recovery, else None.
        """
        self.just_recovered = False
        if self._cooldown > 0:
            self._cooldown -= 1

        if self._phase == "idle":
            stuck_cond = (
                speed < self.STUCK_SPEED_MS and throttle > self.STUCK_THROTTLE_MIN
            )
            if self._cooldown == 0 and stuck_cond:
                self._n += 1
                if self._n >= self.STUCK_THRESHOLD:
                    self._confirm += 1
            else:
                self._n = max(0, self._n - 1)
                self._confirm = 0
            # Reset recovery counter when meaningful progress is made
            if speed > self.PROGRESS_SPEED_MS:
                self._recovery_count = 0
            if self._confirm >= self.STUCK_CONFIRM_FRAMES:
                if self._recovery_count >= self.MAX_RECOVERY_ATTEMPTS:
                    self._phase = "safe_stop"
                    self._counter = 0
                    self._n = 0
                    self._confirm = 0
                    logger.error(
                        "STUCK -> SAFE_STOP after %d recovery attempts "
                        "without progress",
                        self._recovery_count,
                    )
                    return 0.0, 0.0, 1.0, False
                self._recovery_count += 1
                self._phase = "brake"
                self._counter = 0
                self._n = 0
                self._confirm = 0
                self._cooldown = self.COOLDOWN_AFTER_RECOVERY
                logger.warning(
                    "STUCK -> recovering (spd=%.2f thr=%.2f attempt=%d)",
                    speed,
                    throttle,
                    self._recovery_count,
                )
            else:
                return None

        self._counter += 1

        if self._phase == "safe_stop":
            if self._counter > 300:  # 5 seconds at 60fps
                self._phase = "idle"
                self._counter = 0
                self._recovery_count = 0
                logger.info("SAFE_STOP timeout - returning to idle")
                return None
            return 0.0, 0.0, 1.0, False

        if self._phase == "brake":
            if self._counter < self.BRAKE_FRAMES:
                return 0.0, 0.0, 1.0, False
            self._phase = "reverse"
            self._counter = 0

        if self._phase == "reverse":
            if self._counter < self.REVERSE_FRAMES:
                steer = 0.3 * sin(self._counter * 0.3)
                return steer, 0.4, 0.0, True
            self._phase = "forward"
            self._counter = 0

        if self._phase == "forward":
            if self._counter < 25:
                return 0.0, 0.6, 0.0, False
            self._phase = "idle"
            self._counter = 0
            self.just_recovered = True
            self._cooldown = self.COOLDOWN_AFTER_RECOVERY
            return None

    def should_recover(self, speed_ms: float, transform, control_state) -> bool:
        """Check if stuck recovery should be triggered."""
        # Simple implementation based on speed and current phase
        return self._phase != "idle" or (
            speed_ms < self.STUCK_SPEED_MS and
            self._n >= self.STUCK_THRESHOLD
        )

    def get_status(self) -> dict:
        """Return current recovery status as a dict."""
        return {
            "phase": self._phase,
            "recovery_count": self._recovery_count,
            "stuck_frames": self._n,
            "just_recovered": self.just_recovered,
        }
