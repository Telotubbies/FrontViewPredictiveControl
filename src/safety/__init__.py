"""
Safety module for autonomous driving system.

Provides independent safety override logic that must not be bypassed.
"""

from .override import SafetyOverride
from .stuck_recovery import StuckRecovery

__all__ = ["SafetyOverride", "StuckRecovery"]

