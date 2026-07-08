"""
Metrics package — lane compliance, waypoint following, telemetry, run logging, and analysis.

Real-time:
- RunLogger: per-frame CSV + event log (background thread)
- RealTimeStats: live console stats every N frames
- ComprehensiveTelemetryCollector: ring buffer telemetry

Post-run:
- RunAnalyzer: KPIs + plots + report.md + report.json

Domain metrics:
- LaneComplianceMetrics: CTE, heading, departure events
- WaypointFollowingMetrics: CTE vs waypoints, speed tracking
"""
from .lane_compliance import LaneComplianceMetrics, LaneComplianceSummary
from .waypoint_following import WaypointFollowingMetrics, WaypointFollowingSummary
from .telemetry_collector import ComprehensiveTelemetryCollector, TelemetryFrame, AsyncTelemetrySink
from .run_logger import RunLogger, LogEvent
from .realtime_stats import RealTimeStats
from .run_analyzer import RunAnalyzer

__all__ = [
    "LaneComplianceMetrics",
    "LaneComplianceSummary",
    "WaypointFollowingMetrics",
    "WaypointFollowingSummary",
    "ComprehensiveTelemetryCollector",
    "TelemetryFrame",
    "AsyncTelemetrySink",
    "RunLogger",
    "LogEvent",
    "RealTimeStats",
    "RunAnalyzer",
]
