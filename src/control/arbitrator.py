"""
ControlArbitrator — consolidates the override chain into a single priority-ordered
arbitration step.

Priority (lowest → highest):
    1. MPC planner output          (baseline)
    2. ADAS (ACC, LKA Pro, ...)    via adas_manager.apply_to_control()
    3. Safety override             via safety_override.apply_safety_override()
    4. Stuck recovery              via stuck_recovery.update()
    5. AEB emergency brake         (force brake=1.0, throttle=0.0)

The arbitrator DELEGATES to the existing classes (composition). It does NOT
reimplement any override logic — it only sequences the existing calls and logs
what each step changed.
"""
import logging
from dataclasses import dataclass, field
from typing import List, Optional, Any

logger = logging.getLogger(__name__)


@dataclass
class ArbitrationLogEntry:
    """A single field change recorded during arbitration."""
    source: str   # "mpc", "adas", "safety", "stuck", "aeb"
    field: str    # "steer", "throttle", "brake", "reverse"
    old_value: float
    new_value: float
    reason: str


@dataclass
class ControlArbitratorResult:
    """Final arbitrated control plus a trace of what modified it."""
    steer: float
    throttle: float
    brake: float
    reverse: bool
    winning_source: str
    log: List[ArbitrationLogEntry] = field(default_factory=list)

    @property
    def any_override_active(self) -> bool:
        return len(self.log) > 0


# Priority ranking used to decide which source "won" the arbitration.
# Higher number = higher priority.
_PRIORITY = {
    "mpc": 1,
    "adas": 2,
    "safety": 3,
    "stuck": 4,
    "aeb": 5,
}


class ControlArbitrator:
    """
    Arbitrates between MPC, ADAS, safety, stuck-recovery and AEB overrides.

    The arbitrator holds references (composition) to the existing subsystems and
    delegates the actual override computation to them. It only sequences the
    calls in priority order and records what each step changed.
    """

    def __init__(self, safety_override, adas_manager, stuck_recovery):
        """
        Args:
            safety_override: SafetyOverride instance (apply_safety_override).
            adas_manager:    ADASManager instance (apply_to_control).
            stuck_recovery:  StuckRecovery instance (update).
        """
        self._safety = safety_override
        self._adas = adas_manager
        self._stuck = stuck_recovery

    # ------------------------------------------------------------------ #
    #  Public API                                                         #
    # ------------------------------------------------------------------ #
    def arbitrate(
        self,
        mpc_steer: float,
        mpc_throttle: float,
        mpc_brake: float,
        speed_ms: float,
        frame_state: Optional[Any],
        adas_out: Optional[Any],
    ) -> ControlArbitratorResult:
        """
        Run the full override chain and return the arbitrated control.

        Args:
            mpc_steer:    Raw MPC steering command (already smoothed).
            mpc_throttle: Raw MPC throttle command.
            mpc_brake:    Raw MPC brake command.
            speed_ms:     Current ego speed (m/s).
            frame_state:  FrameState (used for CTE / heading context).
            adas_out:     ADASControlOutput from ADASManager.update().

        Returns:
            ControlArbitratorResult with final steer/throttle/brake/reverse,
            the highest-priority source that modified something, and a log.
        """
        log: List[ArbitrationLogEntry] = []

        # ── 1. Baseline: MPC planner output ────────────────────────────
        steer = float(mpc_steer)
        throttle = float(mpc_throttle)
        brake = float(mpc_brake)
        reverse = False

        # ── 2. ADAS (ACC, LKA Pro, traffic light, ...) ─────────────────
        if adas_out is not None:
            steer, throttle, brake = self._apply_adas(
                steer, throttle, brake, adas_out, log
            )

        # ── 3. Safety override (steering clamp, CTE throttle reduction) ─
        cte = float(getattr(frame_state, "cte_m", 0.0)) if frame_state else 0.0
        steer, throttle, brake = self._apply_safety(
            steer, throttle, brake, speed_ms, cte, log
        )

        # ── 4. Stuck recovery (brake → reverse → forward) ──────────────
        reverse = self._apply_stuck(steer, throttle, brake, speed_ms,
                                    log, reverse)

        # ── 5. AEB emergency brake (highest priority) ──────────────────
        if adas_out is not None:
            steer, throttle, brake = self._apply_aeb(
                steer, throttle, brake, adas_out, log
            )

        # ── Determine the winning source ───────────────────────────────
        winning_source = self._compute_winning_source(log)

        return ControlArbitratorResult(
            steer=steer,
            throttle=throttle,
            brake=brake,
            reverse=reverse,
            winning_source=winning_source,
            log=log,
        )

    # ------------------------------------------------------------------ #
    #  Override steps (delegate to existing classes)                      #
    # ------------------------------------------------------------------ #
    def _apply_adas(self, steer, throttle, brake, adas_out, log):
        """Priority 2 — ADAS via adas_manager.apply_to_control()."""
        try:
            new_steer, new_throttle, new_brake, _target = self._adas.apply_to_control(
                steer, throttle, brake, adas_out
            )
        except Exception as e:
            logger.debug(f"ADAS apply_to_control skipped: {e}")
            return steer, throttle, brake

        self._record_diff("adas", steer, throttle, brake,
                          new_steer, new_throttle, new_brake,
                          "ADAS override (ACC/LKA Pro/traffic light)", log)
        return new_steer, new_throttle, new_brake

    def _apply_safety(self, steer, throttle, brake, speed_ms, cte, log):
        """Priority 3 — safety override via safety_override.apply_safety_override()."""
        vehicle_state = {"velocity": speed_ms, "cte": cte}
        try:
            new_steer, new_throttle, new_brake = self._safety.apply_safety_override(
                vehicle_state, steer, throttle, brake
            )
        except Exception as e:
            logger.debug(f"Safety override skipped: {e}")
            return steer, throttle, brake

        self._record_diff("safety", steer, throttle, brake,
                          new_steer, new_throttle, new_brake,
                          "Safety clamp / CTE throttle reduction", log)
        return new_steer, new_throttle, new_brake

    def _apply_stuck(self, steer, throttle, brake, speed_ms, log, reverse):
        """
        Priority 4 — stuck recovery via stuck_recovery.update().

        Returns the (possibly updated) reverse flag. When recovery is active it
        fully overrides steer/throttle/brake/reverse.
        """
        try:
            recovery = self._stuck.update(speed_ms, throttle)
        except Exception as e:
            logger.debug(f"Stuck recovery skipped: {e}")
            return reverse

        if recovery is None:
            return reverse

        r_steer, r_throttle, r_brake, r_reverse = recovery
        self._record_diff("stuck", steer, throttle, brake,
                          r_steer, r_throttle, r_brake,
                          "Stuck recovery active", log)
        # reverse is a bool — log it as a field change too
        if bool(r_reverse) != bool(reverse):
            log.append(ArbitrationLogEntry(
                source="stuck",
                field="reverse",
                old_value=float(reverse),
                new_value=float(r_reverse),
                reason="Stuck recovery reverse gear",
            ))
        return bool(r_reverse)

    def _apply_aeb(self, steer, throttle, brake, adas_out, log):
        """
        Priority 5 — AEB emergency brake.

        If AEB is active, force brake=1.0 and throttle=0.0 regardless of any
        lower-priority decision.
        """
        aeb_active = bool(getattr(adas_out, "aeb_active", False))
        if not aeb_active:
            return steer, throttle, brake

        new_brake = 1.0
        new_throttle = 0.0
        self._record_diff("aeb", steer, throttle, brake,
                          steer, new_throttle, new_brake,
                          "AEB emergency brake", log)
        return steer, new_throttle, new_brake

    # ------------------------------------------------------------------ #
    #  Helpers                                                            #
    # ------------------------------------------------------------------ #
    @staticmethod
    def _record_diff(source, old_steer, old_throttle, old_brake,
                     new_steer, new_throttle, new_brake, reason, log):
        """Append a log entry for each control field that actually changed."""
        if _changed(old_steer, new_steer):
            log.append(ArbitrationLogEntry(
                source=source, field="steer",
                old_value=float(old_steer), new_value=float(new_steer),
                reason=reason,
            ))
        if _changed(old_throttle, new_throttle):
            log.append(ArbitrationLogEntry(
                source=source, field="throttle",
                old_value=float(old_throttle), new_value=float(new_throttle),
                reason=reason,
            ))
        if _changed(old_brake, new_brake):
            log.append(ArbitrationLogEntry(
                source=source, field="brake",
                old_value=float(old_brake), new_value=float(new_brake),
                reason=reason,
            ))

    @staticmethod
    def _compute_winning_source(log: List[ArbitrationLogEntry]) -> str:
        """
        The winning source is the highest-priority source that modified at
        least one field. If nothing was modified, the MPC baseline wins.
        """
        if not log:
            return "mpc"
        sources = {entry.source for entry in log}
        return max(sources, key=lambda s: _PRIORITY.get(s, 0))


def _changed(old: float, new: float, tol: float = 1e-9) -> bool:
    """True if two control values differ beyond a small tolerance."""
    return abs(float(new) - float(old)) > tol
