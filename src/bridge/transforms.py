"""
CARLA ↔ right-handed (ENU-style) coordinate transforms.
Adapted from carla_apollo_bridge; no Apollo/Cyber dependency.

CARLA (Unreal): X forward, Y right, Z up (left-handed).
Bridge output:  X forward, Y left (-y_carla), Z up (right-handed).
"""

from __future__ import annotations

import math
from typing import Tuple

import numpy as np


def carla_location_to_enu(x: float, y: float, z: float) -> Tuple[float, float, float]:
    """CARLA location → (x_enu, y_enu, z_enu). y_enu = -y_carla."""
    return (x, -y, z)


def enu_to_carla_location(x_enu: float, y_enu: float, z_enu: float) -> Tuple[float, float, float]:
    """(x_enu, y_enu, z_enu) → CARLA (x, y, z)."""
    return (x_enu, -y_enu, z_enu)


def carla_rotation_to_heading_rad(roll_deg: float, pitch_deg: float, yaw_deg: float) -> float:
    """
    CARLA rotation (degrees) → heading theta in radians (right-handed, e.g. Apollo-style).
    theta = -radians(yaw + 90) from original bridge.
    """
    return -math.radians(yaw_deg + 90)


def carla_velocity_to_enu(vx: float, vy: float, vz: float) -> Tuple[float, float, float]:
    """CARLA velocity → ENU velocity. vy_enu = -vy_carla."""
    return (vx, -vy, vz)


def carla_bbox_to_length_width_height(extent_x: float, extent_y: float, extent_z: float) -> Tuple[float, float, float]:
    """Bounding box extent (half) → length, width, height (full)."""
    return (extent_x * 2.0, extent_y * 2.0, extent_z * 2.0)
