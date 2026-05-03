"""
Bridge adapter from carla_apollo_bridge: CARLA ↔ obstacle (no Apollo/Cyber dependency).
Use for traffic participants and coordinate transforms in carla_mpc_classical.
"""

from .transforms import (
    carla_location_to_enu,
    enu_to_carla_location,
    carla_rotation_to_heading_rad,
    carla_velocity_to_enu,
    carla_bbox_to_length_width_height,
)
from .obstacles import (
    Obstacle,
    ObstacleType,
    carla_actor_to_obstacle,
    get_traffic_obstacles,
)

__all__ = [
    "carla_location_to_enu",
    "enu_to_carla_location",
    "carla_rotation_to_heading_rad",
    "carla_velocity_to_enu",
    "carla_bbox_to_length_width_height",
    "Obstacle",
    "ObstacleType",
    "carla_actor_to_obstacle",
    "get_traffic_obstacles",
]
