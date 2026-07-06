"""
Traffic participants: CARLA actors → Obstacle (Apollo-style, no protobuf).
Adapted from carla_apollo_bridge actor/traffic_participant.py.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import IntEnum
from typing import List, Optional

from . import transforms as trans

logger = logging.getLogger(__name__)


class ObstacleType(IntEnum):
    UNKNOWN = 0
    VEHICLE = 1
    BICYCLE = 2
    PEDESTRIAN = 3
    UNKNOWN_MOVABLE = 4
    UNKNOWN_UNMOVABLE = 5


@dataclass
class Obstacle:
    """Apollo-style perception obstacle (no protobuf)."""
    id: int
    position: tuple  # (x, y, z) in ENU
    velocity: tuple  # (vx, vy, vz) in ENU
    theta: float     # heading rad
    length: float
    width: float
    height: float
    type: ObstacleType = ObstacleType.UNKNOWN


def _get_carla_actor_type(carla_actor) -> ObstacleType:
    """Map CARLA actor type to ObstacleType."""
    type_id = getattr(carla_actor, "type_id", "") or ""
    if "vehicle" in type_id:
        if "bike" in type_id or "motorcycle" in type_id or "bicycle" in type_id:
            return ObstacleType.BICYCLE
        return ObstacleType.VEHICLE
    if "walker" in type_id or "pedestrian" in type_id:
        return ObstacleType.PEDESTRIAN
    return ObstacleType.UNKNOWN


def carla_actor_to_obstacle(carla_actor) -> Optional[Obstacle]:
    """
    Convert one CARLA actor (vehicle or walker) to Obstacle.
    Returns None if actor has no bounding_box or invalid.
    """
    try:
        transform = carla_actor.get_transform()
        location = transform.location
        rotation = transform.rotation
        velocity = carla_actor.get_velocity()
    except Exception as e:
        logger.debug("carla_actor_to_obstacle failed for actor %s: %s", getattr(carla_actor, "id", "?"), e)
        return None

    pos_enu = trans.carla_location_to_enu(location.x, location.y, location.z)
    vel_enu = trans.carla_velocity_to_enu(velocity.x, velocity.y, velocity.z)
    theta = trans.carla_rotation_to_heading_rad(
        rotation.roll, rotation.pitch, rotation.yaw
    )

    length, width, height = 4.0, 2.0, 1.5
    try:
        bb = carla_actor.bounding_box
        length, width, height = trans.carla_bbox_to_length_width_height(
            bb.extent.x, bb.extent.y, bb.extent.z
        )
    except Exception as e:
        logger.debug("bounding_box for actor %s: %s", getattr(carla_actor, "id", "?"), e)

    return Obstacle(
        id=carla_actor.id,
        position=pos_enu,
        velocity=vel_enu,
        theta=theta,
        length=length,
        width=width,
        height=height,
        type=_get_carla_actor_type(carla_actor),
    )


def get_traffic_obstacles(world, exclude_actor_id: Optional[int] = None) -> List[Obstacle]:
    """
    Get all traffic participants (vehicles + walkers) from CARLA world as Obstacles.
    Optionally exclude one actor id (e.g. ego vehicle).
    """
    obstacles: List[Obstacle] = []
    try:
        actors = world.get_actors()
    except Exception as e:
        logger.warning("get_traffic_obstacles: world.get_actors failed: %s", e)
        return obstacles

    for actor in actors:
        if exclude_actor_id is not None and actor.id == exclude_actor_id:
            continue
        type_id = getattr(actor, "type_id", "") or ""
        if "vehicle" not in type_id and "walker" not in type_id:
            continue
        obj = carla_actor_to_obstacle(actor)
        if obj is not None:
            obstacles.append(obj)

    return obstacles
