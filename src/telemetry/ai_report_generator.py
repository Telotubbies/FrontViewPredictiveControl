"""
AiReportGenerator — สร้างรายงานแบบ AI-readable จากข้อมูล telemetry ทั้งหมด

เป้าหมาย: สร้างรายงาน natural language ฉบับเดียวที่ AI (เช่น LLM) อ่านแล้ว
เข้าใจได้ทันทีว่าเกิดอะไรขึ้นระหว่าง run — โดยเฉพาะเมื่อ/ทำไมรถถึงออกจากเลน
และมีพฤติกรรมอะไรเกิดขึ้นบ้าง

รายงานแบ่งเป็น 6 sections:
  1. Run Overview
  2. Lane Detection Health
  3. Lane Loss Incidents (detailed)
  4. Behavior Patterns
  5. Failure Analysis
  6. Critical Events Timeline

ใช้คู่กับ BehaviorLogger + RunLogger:
  - BehaviorSummary → สรุปเชิงพฤติกรรม
  - events (List[BehaviorEvent]) → timeline ของ critical events
  - lane_loss_incidents (List[dict]) → รายละเอียดแต่ละครั้งที่เสียเลน
  - run_meta (dict) → map, vehicle, target_speed, duration, total_frames
"""
from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from telemetry.behavior_logger import BehaviorEvent, BehaviorSummary, BehaviorType, Severity

logger = logging.getLogger(__name__)


# ── Expected keys for a lane loss incident dict ──────────────────────────────
# {
#   "frame_idx": int,
#   "timestamp": float,
#   "root_cause": str,            # เช่น "lane_conf_drop", "geometry_invalid", "phase_failure"
#   "conf_pre": float,            # confidence ก่อนหาย
#   "conf_post": float,           # confidence ตอนหาย
#   "speed_kmh": float,           # ความเร็วตอนเริ่มหาย
#   "curve_type": str,            # straight | gentle_curve | sharp_curve | unknown
#   "duration_frames": int,       # นานกี่ frame
#   "cte_peak": float,            # CTE สูงสุดระหว่างหาย
#   "cte_direction": str,         # left | right | center
#   "recovery_method": str,       # auto | fallback | manual | not_recovered
# }


class AiReportGenerator:
    """
    สร้างรายงาน AI-readable จาก telemetry data.

    ใช้งาน:
        gen = AiReportGenerator()
        report = gen.generate_report(summary, events, incidents, run_meta)
        gen.save_report(report, "run_report.txt")
        j = gen.generate_json_report(summary, events, incidents, run_meta)
        prompt = gen.generate_prompt_for_llm(report, "Why did the car lose the lane?")
    """

    # Severity levels ที่ถือว่าเป็น critical events (นำไปแสดงใน timeline)
    CRITICAL_SEVERITIES = (Severity.WARNING, Severity.ERROR, Severity.CRITICAL)

    def __init__(self) -> None:
        """สร้าง AiReportGenerator instance."""
        logger.debug("AiReportGenerator initialized")

    # ─────────────────────────────────────────────────────────────────────────
    #  Public API
    # ─────────────────────────────────────────────────────────────────────────

    def generate_report(
        self,
        behavior_summary: BehaviorSummary,
        events: Sequence[BehaviorEvent],
        lane_loss_incidents: Sequence[Dict[str, Any]],
        run_meta: Dict[str, Any],
    ) -> str:
        """
        สร้างรายงานแบบ multi-section เต็มรูปแบบ (markdown-style).

        Args:
            behavior_summary: BehaviorSummary จาก BehaviorLogger.get_summary()
            events: list ของ BehaviorEvent ทั้งหมดใน run
            lane_loss_incidents: list ของ dict อธิบายแต่ละครั้งที่เสียเลน
            run_meta: dict ที่มี keys: map, vehicle, target_speed, duration, total_frames

        Returns:
            รายงานเป็น string พร้อม paste ลง LLM prompt ได้เลย
        """
        logger.info(
            "Generating AI report: %d events, %d lane-loss incidents",
            len(events), len(lane_loss_incidents),
        )
        sections: List[str] = []
        sections.append(self._section_run_overview(behavior_summary, run_meta))
        sections.append(self._section_lane_detection_health(behavior_summary, run_meta))
        sections.append(self._section_lane_loss_incidents(lane_loss_incidents))
        sections.append(self._section_behavior_patterns(behavior_summary))
        sections.append(self._section_failure_analysis(behavior_summary, events, lane_loss_incidents))
        sections.append(self._section_critical_events_timeline(events))
        return "\n\n".join(sections)

    def generate_summary_card(
        self,
        behavior_summary: BehaviorSummary,
        run_meta: Dict[str, Any],
    ) -> str:
        """
        สร้างสรุปแบบกระชับ 3-4 บรรทัด สำหรับ AI อ่านรวดเร็ว.

        Args:
            behavior_summary: BehaviorSummary
            run_meta: run metadata dict

        Returns:
            สรุปสั้น 3-4 บรรทัด
        """
        map_name = run_meta.get("map", "unknown")
        vehicle = run_meta.get("vehicle", "unknown")
        duration = run_meta.get("duration", 0.0)
        total_frames = run_meta.get("total_frames", 0)
        target_speed = run_meta.get("target_speed", 0.0)
        quality = behavior_summary.overall_lane_keeping_quality

        lines = [
            f"Run on {map_name} ({vehicle}), {duration:.1f}s / {total_frames} frames, target {target_speed:.0f} kmh.",
            f"Lane keeping quality: {quality}. Confidence mean={behavior_summary.lane_conf_mean:.2f}, min={behavior_summary.lane_conf_min:.2f}.",
            f"Lane lost {behavior_summary.lane_lost_count}x ({behavior_summary.lane_lost_total_frames} frames), "
            f"off-track {behavior_summary.off_track_count}x, dominant failure: {behavior_summary.dominant_failure_mode}.",
        ]
        return " ".join(lines)

    def generate_failure_analysis(
        self,
        behavior_summary: BehaviorSummary,
        events: Sequence[BehaviorEvent],
        lane_loss_incidents: Sequence[Dict[str, Any]],
    ) -> str:
        """
        สร้างเฉพาะ Section 5 — Failure Analysis.

        Args:
            behavior_summary: BehaviorSummary
            events: list ของ BehaviorEvent
            lane_loss_incidents: list ของ lane loss incident dicts

        Returns:
            ข้อความ failure analysis
        """
        return self._section_failure_analysis(behavior_summary, events, lane_loss_incidents)

    def save_report(self, report_text: str, filepath: str) -> str:
        """
        บันทึกรายงานลงไฟล์ .txt.

        Args:
            report_text: ข้อความรายงาน
            filepath: path ของไฟล์ที่จะบันทึก

        Returns:
            path ของไฟล์ที่บันทึกจริง (absolute path)
        """
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(report_text, encoding="utf-8")
        logger.info("AI report saved to %s (%d chars)", path, len(report_text))
        return str(path.resolve())

    def generate_json_report(
        self,
        behavior_summary: BehaviorSummary,
        events: Sequence[BehaviorEvent],
        lane_loss_incidents: Sequence[Dict[str, Any]],
        run_meta: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        สร้างรายงานเป็น structured JSON สำหรับ AI ที่รับ JSON input.

        Args:
            behavior_summary: BehaviorSummary
            events: list ของ BehaviorEvent
            lane_loss_incidents: list ของ lane loss incident dicts
            run_meta: run metadata dict

        Returns:
            dict ที่ serialize เป็น JSON ได้
        """
        return {
            "generated_at": datetime.now().isoformat(),
            "run_overview": {
                "map": run_meta.get("map", "unknown"),
                "vehicle": run_meta.get("vehicle", "unknown"),
                "duration_s": run_meta.get("duration", 0.0),
                "total_frames": run_meta.get("total_frames", 0),
                "target_speed_kmh": run_meta.get("target_speed", 0.0),
                "overall_lane_keeping_quality": behavior_summary.overall_lane_keeping_quality,
            },
            "lane_detection_health": {
                "lane_conf_mean": round(behavior_summary.lane_conf_mean, 4),
                "lane_conf_min": round(behavior_summary.lane_conf_min, 4),
                "lane_lost_count": behavior_summary.lane_lost_count,
                "lane_lost_total_frames": behavior_summary.lane_lost_total_frames,
                "lane_lost_max_duration_frames": behavior_summary.lane_lost_max_duration_frames,
                "geometry_invalid_count": behavior_summary.geometry_invalid_count,
                "dominant_failure_mode": behavior_summary.dominant_failure_mode,
            },
            "lane_loss_incidents": [self._incident_to_dict(i) for i in lane_loss_incidents],
            "behavior_patterns": {
                "drift_left_count": behavior_summary.drift_left_count,
                "drift_right_count": behavior_summary.drift_right_count,
                "off_track_count": behavior_summary.off_track_count,
                "off_track_total_frames": behavior_summary.off_track_total_frames,
                "oscillation_count": behavior_summary.oscillation_count,
                "sudden_brake_count": behavior_summary.sudden_brake_count,
                "excessive_steering_count": behavior_summary.excessive_steering_count,
                "mpc_unstable_count": behavior_summary.mpc_unstable_count,
                "mode_change_count": behavior_summary.mode_change_count,
            },
            "failure_analysis": self._build_failure_analysis_dict(
                behavior_summary, events, lane_loss_incidents,
            ),
            "critical_events": [
                self._event_to_dict(ev)
                for ev in events
                if ev.severity in self.CRITICAL_SEVERITIES
            ],
            "summary_card": self.generate_summary_card(behavior_summary, run_meta),
        }

    def generate_prompt_for_llm(self, report_text: str, question: str) -> str:
        """
        ห่อรายงานใน prompt template สำหรับถาม LLM.

        Args:
            report_text: รายงานเต็มรูปแบบ (จาก generate_report)
            question: คำถามที่ต้องการให้ LLM ตอบ

        Returns:
            prompt string พร้อมส่งให้ LLM
        """
        return (
            "You are an autonomous driving systems analyst. Below is a structured "
            "telemetry report from a CARLA simulator run of an MPC-based lane-keeping "
            "ADAS system. Read the report carefully and answer the question at the end.\n\n"
            "=== TELEMETRY REPORT ===\n"
            f"{report_text}\n\n"
            "=== END REPORT ===\n\n"
            f"Question: {question}\n\n"
            "Answer with specific references to frames, timestamps, and events from the report. "
            "If the report does not contain enough information, state what is missing."
        )

    # ─────────────────────────────────────────────────────────────────────────
    #  Section builders (text)
    # ─────────────────────────────────────────────────────────────────────────

    def _section_run_overview(
        self,
        summary: BehaviorSummary,
        run_meta: Dict[str, Any],
    ) -> str:
        """Section 1 — Run Overview."""
        map_name = run_meta.get("map", "unknown")
        vehicle = run_meta.get("vehicle", "unknown")
        duration = run_meta.get("duration", 0.0)
        total_frames = run_meta.get("total_frames", 0)
        target_speed = run_meta.get("target_speed", 0.0)
        quality = summary.overall_lane_keeping_quality

        return (
            "## SECTION 1 — RUN OVERVIEW\n\n"
            f"Run completed on **{map_name}** with **{vehicle}**. "
            f"Duration: {duration:.1f}s ({total_frames} frames). "
            f"Target speed: {target_speed:.0f} kmh. "
            f"Overall lane keeping quality: **{quality}**."
        )

    def _section_lane_detection_health(
        self,
        summary: BehaviorSummary,
        run_meta: Dict[str, Any],
    ) -> str:
        """Section 2 — Lane Detection Health."""
        total_frames = run_meta.get("total_frames", 0) or 1
        lost_pct = (summary.lane_lost_total_frames / total_frames) * 100.0

        return (
            "## SECTION 2 — LANE DETECTION HEALTH\n\n"
            f"Lane confidence: mean={summary.lane_conf_mean:.3f}, "
            f"min={summary.lane_conf_min:.3f}. "
            f"Lane was lost **{summary.lane_lost_count}** times for total "
            f"{summary.lane_lost_total_frames} frames ({lost_pct:.1f}% of run). "
            f"Max single loss duration: {summary.lane_lost_max_duration_frames} frames. "
            f"Geometry was invalid {summary.geometry_invalid_count} times. "
            f"Dominant failure mode: **{summary.dominant_failure_mode}**."
        )

    def _section_lane_loss_incidents(
        self,
        incidents: Sequence[Dict[str, Any]],
    ) -> str:
        """Section 3 — Lane Loss Incidents (detailed)."""
        lines = ["## SECTION 3 — LANE LOSS INCIDENTS"]
        if not incidents:
            lines.append("\nNo lane loss incidents recorded during this run.")
            return "\n".join(lines)

        lines.append("")
        for idx, inc in enumerate(incidents, start=1):
            frame = inc.get("frame_idx", -1)
            t = inc.get("timestamp", 0.0)
            root = inc.get("root_cause", "unknown")
            conf_pre = inc.get("conf_pre", 0.0)
            conf_post = inc.get("conf_post", 0.0)
            speed = inc.get("speed_kmh", 0.0)
            curve = inc.get("curve_type", "unknown")
            dur = inc.get("duration_frames", 0)
            cte_peak = inc.get("cte_peak", 0.0)
            cte_dir = inc.get("cte_direction", "unknown")
            recovery = inc.get("recovery_method", "unknown")

            lines.append(
                f"- Loss #{idx} at frame {frame} (t={t:.1f}s): {root}. "
                f"Confidence dropped from {conf_pre:.2f} to {conf_post:.2f}. "
                f"Speed was {speed:.0f} kmh on {curve}. "
                f"Lasted {dur} frames. "
                f"CTE peaked at {cte_peak:.2f}m ({cte_dir}). "
                f"Recovered via {recovery}."
            )
        return "\n".join(lines)

    def _section_behavior_patterns(self, summary: BehaviorSummary) -> str:
        """Section 4 — Behavior Patterns."""
        return (
            "## SECTION 4 — BEHAVIOR PATTERNS\n\n"
            f"Detected behaviors: "
            f"{summary.drift_left_count} drifting left, "
            f"{summary.drift_right_count} drifting right, "
            f"{summary.off_track_count} off-track events "
            f"({summary.off_track_total_frames} frames), "
            f"{summary.oscillation_count} oscillation events, "
            f"{summary.sudden_brake_count} sudden brakes, "
            f"{summary.excessive_steering_count} excessive steering, "
            f"{summary.mpc_unstable_count} MPC unstable events, "
            f"{summary.mode_change_count} perception mode changes."
        )

    def _section_failure_analysis(
        self,
        summary: BehaviorSummary,
        events: Sequence[BehaviorEvent],
        incidents: Sequence[Dict[str, Any]],
    ) -> str:
        """Section 5 — Failure Analysis."""
        dominant = summary.dominant_failure_mode
        conditions = self._infer_failure_conditions(summary, events, incidents)
        pattern_desc = self._describe_failure_pattern(summary, events, incidents)
        recommendation = self._suggest_improvement(summary, events, incidents)

        return (
            "## SECTION 5 — FAILURE ANALYSIS\n\n"
            f"The car's primary issue was **{dominant}**. "
            f"This occurred most often when {conditions}. "
            f"Pattern: {pattern_desc}. "
            f"Recommendation: {recommendation}"
        )

    def _section_critical_events_timeline(
        self,
        events: Sequence[BehaviorEvent],
    ) -> str:
        """Section 6 — Critical Events Timeline."""
        lines = ["## SECTION 6 — CRITICAL EVENTS TIMELINE"]
        critical = [
            ev for ev in events
            if ev.severity in self.CRITICAL_SEVERITIES
        ]
        if not critical:
            lines.append("\nNo WARNING/ERROR/CRITICAL events recorded.")
            return "\n".join(lines)

        lines.append("")
        for ev in critical:
            lines.append(
                f"- [frame {ev.frame_idx} | t={ev.timestamp:.1f}s | {ev.severity.value.upper()}] "
                f"{ev.event_type.value}: {ev.message}"
            )
        return "\n".join(lines)

    # ─────────────────────────────────────────────────────────────────────────
    #  Failure analysis helpers
    # ─────────────────────────────────────────────────────────────────────────

    def _infer_failure_conditions(
        self,
        summary: BehaviorSummary,
        events: Sequence[BehaviorEvent],
        incidents: Sequence[Dict[str, Any]],
    ) -> str:
        """อนุมานเงื่อนไขที่ failure เกิดบ่อยที่สุด."""
        dominant = summary.dominant_failure_mode

        # วิเคราะห์จาก lane loss incidents
        if dominant in ("lane_lost", "lane_conf_drop", "geometry_invalid") and incidents:
            curve_types: Dict[str, int] = {}
            speed_sum = 0.0
            for inc in incidents:
                ct = inc.get("curve_type", "unknown")
                curve_types[ct] = curve_types.get(ct, 0) + 1
                speed_sum += inc.get("speed_kmh", 0.0)
            avg_speed = speed_sum / len(incidents) if incidents else 0.0
            top_curve = max(curve_types, key=curve_types.get) if curve_types else "unknown"
            return (
                f"driving at ~{avg_speed:.0f} kmh on {top_curve} sections "
                f"(curve distribution: {curve_types})"
            )

        # วิเคราะห์จาก events ที่เกี่ยวข้อง
        related = [ev for ev in events if dominant.replace("drifting", "drifting") in ev.event_type.value]
        if related:
            cte_values = [ev.context.get("cte_m", 0.0) for ev in related if "cte_m" in ev.context]
            if cte_values:
                avg_cte = sum(abs(c) for c in cte_values) / len(cte_values)
                return f"CTE averaged {avg_cte:.2f}m during {dominant} events"

        if dominant == "oscillation":
            return "steering input was rapidly varying (high steer std)"
        if dominant == "mpc_unstable":
            return "the MPC solver repeatedly fell back to a fallback controller"
        if dominant == "off_track":
            return "CTE exceeded the off-track threshold and the vehicle left the lane"
        if dominant == "none":
            return "no significant failures were detected"
        return f"the {dominant} condition was triggered"

    def _describe_failure_pattern(
        self,
        summary: BehaviorSummary,
        events: Sequence[BehaviorEvent],
        incidents: Sequence[Dict[str, Any]],
    ) -> str:
        """อธิบายรูปแบบการเกิด failure (clustering)."""
        dominant = summary.dominant_failure_mode
        if dominant == "none":
            return "no failure clustering observed — the run was stable"

        # ดูว่า events ของ dominant type กระจุกตอนไหน
        related = [ev for ev in events if dominant in ev.event_type.value or ev.event_type.value in dominant]
        if not related:
            return f"{dominant} events were sparse and did not cluster"

        frames = [ev.frame_idx for ev in related]
        f_min, f_max = min(frames), max(frames)
        span = f_max - f_min

        # ถ้ากระจุกในช่วงสั้น → cluster; กระจาย → spread
        if span < 100 and len(related) > 2:
            location = f"frames {f_min}-{f_max}"
            return f"{dominant} events clustered around {location} ({len(related)} events in {span} frames)"
        if len(related) > 1:
            return f"{dominant} events were spread across frames {f_min}-{f_max} ({len(related)} total)"
        return f"a single {dominant} event occurred at frame {f_min}"

    def _suggest_improvement(
        self,
        summary: BehaviorSummary,
        events: Sequence[BehaviorEvent],
        incidents: Sequence[Dict[str, Any]],
    ) -> str:
        """แนะนำการปรับปรุงตามข้อมูล."""
        dominant = summary.dominant_failure_mode

        if dominant == "lane_conf_drop" or dominant == "lane_lost":
            if incidents:
                curve_types = [inc.get("curve_type", "unknown") for inc in incidents]
                if "sharp_curve" in curve_types:
                    return "improve lane detection robustness on sharp curves — consider lowering target speed on curved sections or enhancing the phase-2 boundary completion logic."
                if "gentle_curve" in curve_types:
                    return "tune the confidence recovery threshold and improve mask smoothing to reduce false lane-loss on gentle curves."
            return "increase lane detection confidence thresholds or add temporal smoothing to avoid spurious lane-loss detection."
        if dominant == "geometry_invalid":
            return "review the geometry validation pipeline — the centerline/CTE computation may reject valid lanes too aggressively."
        if dominant == "off_track":
            return "tighten the MPC lateral tracking weights and engage LKA Pro earlier when CTE exceeds the drift threshold."
        if dominant == "oscillation":
            return "add steering rate limiting or low-pass filtering on the MPC steer output to dampen oscillation."
        if dominant == "mpc_unstable":
            return "review MPC solver settings (horizon, weights) and improve the fallback controller so consecutive fallbacks do not destabilize tracking."
        if dominant == "drifting":
            return "increase the lateral error penalty in the MPC cost function to reduce steady-state drift."
        if dominant == "none":
            return "no changes needed — the system maintained lane keeping effectively."
        return f"investigate {dominant} events in more detail to determine root cause."

    def _build_failure_analysis_dict(
        self,
        summary: BehaviorSummary,
        events: Sequence[BehaviorEvent],
        incidents: Sequence[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """สร้าง failure analysis เป็น dict (สำหรับ JSON report)."""
        return {
            "dominant_failure_mode": summary.dominant_failure_mode,
            "conditions": self._infer_failure_conditions(summary, events, incidents),
            "pattern": self._describe_failure_pattern(summary, events, incidents),
            "recommendation": self._suggest_improvement(summary, events, incidents),
        }

    # ─────────────────────────────────────────────────────────────────────────
    #  Conversion helpers
    # ─────────────────────────────────────────────────────────────────────────

    def _incident_to_dict(self, inc: Dict[str, Any]) -> Dict[str, Any]:
        """แปลง lane loss incident dict ให้ JSON-safe."""
        return {
            "frame_idx": inc.get("frame_idx", -1),
            "timestamp": round(float(inc.get("timestamp", 0.0)), 3),
            "root_cause": inc.get("root_cause", "unknown"),
            "conf_pre": round(float(inc.get("conf_pre", 0.0)), 4),
            "conf_post": round(float(inc.get("conf_post", 0.0)), 4),
            "speed_kmh": round(float(inc.get("speed_kmh", 0.0)), 2),
            "curve_type": inc.get("curve_type", "unknown"),
            "duration_frames": int(inc.get("duration_frames", 0)),
            "cte_peak": round(float(inc.get("cte_peak", 0.0)), 4),
            "cte_direction": inc.get("cte_direction", "unknown"),
            "recovery_method": inc.get("recovery_method", "unknown"),
        }

    def _event_to_dict(self, ev: BehaviorEvent) -> Dict[str, Any]:
        """แปลง BehaviorEvent ให้ JSON-safe dict."""
        return {
            "frame_idx": ev.frame_idx,
            "timestamp": round(ev.timestamp, 3),
            "event_type": ev.event_type.value,
            "severity": ev.severity.value,
            "message": ev.message,
            "context": ev.context,
        }
