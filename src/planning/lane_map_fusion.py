"""
Lane + Map Fusion — route-aware lane masking.

Combines DSUNet lane detection with CARLA map topology to produce a mask
that highlights the correct lane for the planned route. At intersections,
instead of just showing the ego lane, it shows the lane that leads to the
destination.

Usage:
    from planning.lane_map_fusion import LaneMapFusion

    fusion = LaneMapFusion(world.get_map())
    target_mask = fusion.generate_route_aware_mask(
        rgb, dsunet_mask, vehicle_transform, route,
    )
"""
import logging
import math
from typing import List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


class LaneMapFusion:
    """Fuses DSUNet lane mask with CARLA map topology for route-aware masking.

    This module answers: "which lane should the car be in to reach its
    destination, and what does that lane look like in the camera image?"

    Three modes:
    1. LANE_KEEP: DSUNet mask is sufficient — ego lane is the correct lane
    2. LANE_CHANGE: Mask the target lane (adjacent lane to change into)
    3. INTERSECTION: Mask the lane that connects to the route's next segment
    """

    def __init__(self, carla_map, cam_w: int = 640, cam_h: int = 480,
                 fov_deg: float = 90.0, camera_height: float = 1.8,
                 camera_pitch: float = -8.0):
        """Initialize the lane-map fusion module.

        Args:
            carla_map: carla.Map object
            cam_w, cam_h: Camera resolution
            fov_deg: Camera horizontal FOV
            camera_height: Camera height above ground (m)
            camera_pitch: Camera pitch angle (degrees)
        """
        self.map = carla_map
        self.cam_w = cam_w
        self.cam_h = cam_h
        self.fov_deg = fov_deg
        self.camera_height = camera_height
        self.camera_pitch = math.radians(camera_pitch)
        self.focal_length = (cam_w / 2.0) / math.tan(math.radians(fov_deg) / 2.0)

    def generate_route_aware_mask(
        self,
        rgb: np.ndarray,
        dsunet_mask: np.ndarray,
        vehicle_transform,
        route: List,  # List[RouteEntry]
        lookahead_m: float = 30.0,
    ) -> Tuple[np.ndarray, str]:
        """Generate a route-aware lane mask.

        Args:
            rgb: RGB image (H, W, 3)
            dsunet_mask: DSUNet lane mask (H, W) uint8
            vehicle_transform: carla.Transform of the vehicle
            route: Planned route from GlobalRoutePlanner
            lookahead_m: How far ahead to consider

        Returns:
            (target_mask, mode) where:
            - target_mask: (H, W) uint8 mask of the lane to follow
            - mode: "LANE_KEEP" | "LANE_CHANGE" | "INTERSECTION"
        """
        from planning.route_planner import (
            LANE_KEEP, LANE_CHANGE_LEFT, LANE_CHANGE_RIGHT,
            TURN_LEFT, TURN_RIGHT, ARRIVE,
        )

        # Determine current driving mode from route
        mode = self._determine_mode(vehicle_transform, route, lookahead_m)

        if mode == "LANE_KEEP":
            # DSUNet ego lane mask is correct — just return it
            return dsunet_mask, mode

        elif mode == "LANE_CHANGE":
            # Need to mask the target lane (left or right neighbor)
            target_lane_wp = self._get_target_lane_waypoint(
                vehicle_transform, route, lookahead_m,
            )
            if target_lane_wp is not None:
                map_mask = self._waypoints_to_mask(
                    [target_lane_wp], vehicle_transform, rgb.shape[:2],
                )
                # Blend: use map mask as guide, DSUNet as detail
                fused = self._blend_masks(dsunet_mask, map_mask)
                return fused, mode
            return dsunet_mask, "LANE_KEEP"

        elif mode == "INTERSECTION":
            # At intersection — mask the lane that connects to route
            target_wps = self._get_intersection_waypoints(
                vehicle_transform, route, lookahead_m,
            )
            if target_wps:
                map_mask = self._waypoints_to_mask(
                    target_wps, vehicle_transform, rgb.shape[:2],
                )
                fused = self._blend_masks(dsunet_mask, map_mask)
                return fused, mode
            return dsunet_mask, "LANE_KEEP"

        return dsunet_mask, "LANE_KEEP"

    def _determine_mode(self, vehicle_transform, route, lookahead_m) -> str:
        """Determine current driving mode from route."""
        from planning.route_planner import (
            LANE_KEEP, LANE_CHANGE_LEFT, LANE_CHANGE_RIGHT,
            TURN_LEFT, TURN_RIGHT, ARRIVE,
        )

        vehicle_loc = vehicle_transform.location
        for entry in route:
            dist = entry.waypoint.transform.location.distance(vehicle_loc)
            if dist > lookahead_m:
                break
            if entry.action in (LANE_CHANGE_LEFT, LANE_CHANGE_RIGHT):
                return "LANE_CHANGE"
            if entry.action in (TURN_LEFT, TURN_RIGHT):
                return "INTERSECTION"
            if entry.action == ARRIVE and dist < 10:
                return "LANE_KEEP"
        return "LANE_KEEP"

    def _get_target_lane_waypoint(self, vehicle_transform, route, lookahead_m):
        """Get the waypoint of the target lane for lane change."""
        from planning.route_planner import LANE_CHANGE_LEFT, LANE_CHANGE_RIGHT

        vehicle_loc = vehicle_transform.location
        for entry in route:
            dist = entry.waypoint.transform.location.distance(vehicle_loc)
            if dist > lookahead_m:
                break
            if entry.action == LANE_CHANGE_LEFT:
                # Get left lane waypoint
                wp = self.map.get_waypoint(vehicle_loc)
                try:
                    left = wp.get_left_lane()
                    if left is not None:
                        return left
                except Exception:
                    pass
            elif entry.action == LANE_CHANGE_RIGHT:
                wp = self.map.get_waypoint(vehicle_loc)
                try:
                    right = wp.get_right_lane()
                    if right is not None:
                        return right
                except Exception:
                    pass
        return None

    def _get_intersection_waypoints(self, vehicle_transform, route, lookahead_m):
        """Get waypoints for the lane through an intersection."""
        from planning.route_planner import TURN_LEFT, TURN_RIGHT, STRAIGHT

        vehicle_loc = vehicle_transform.location
        target_wps = []
        for entry in route:
            dist = entry.waypoint.transform.location.distance(vehicle_loc)
            if dist > lookahead_m:
                break
            if entry.action in (TURN_LEFT, TURN_RIGHT, STRAIGHT):
                # Collect waypoints along this turn
                try:
                    wps = entry.waypoint.next(2.0)
                    if wps:
                        target_wps.extend(wps[:10])
                except Exception:
                    pass
        return target_wps

    def _waypoints_to_mask(self, waypoints, vehicle_transform, img_shape) -> np.ndarray:
        """Project waypoints to image space and create a lane mask.

        Args:
            waypoints: List of carla.Waypoint
            vehicle_transform: carla.Transform of vehicle
            img_shape: (H, W) of the output mask

        Returns:
            Binary mask (H, W) uint8 with 255 at lane positions
        """
        h, w = img_shape
        mask = np.zeros((h, w), dtype=np.uint8)

        # Get vehicle location and rotation
        veh_loc = vehicle_transform.location
        veh_rot = vehicle_transform.rotation

        # Vehicle forward direction
        veh_yaw = math.radians(veh_rot.yaw)
        cos_yaw = math.cos(veh_yaw)
        sin_yaw = math.sin(veh_yaw)

        # Camera offset from vehicle center (approximate)
        cam_x_offset = 2.0  # camera is ~2m forward of vehicle center
        cam_z_offset = self.camera_height

        for wp in waypoints:
            wp_loc = wp.transform.location

            # Transform to vehicle coordinate frame
            dx = wp_loc.x - veh_loc.x
            dy = wp_loc.y - veh_loc.y
            # Rotate to vehicle frame (x = forward, y = left)
            x_veh = dx * cos_yaw + dy * sin_yaw - cam_x_offset
            y_veh = -dx * sin_yaw + dy * cos_yaw

            # Skip points behind camera
            if x_veh < 1.0:
                continue

            # Project to image using pinhole model
            # Camera looks forward, pitched down
            pitch = self.camera_pitch
            # Adjust y for pitch
            y_cam = y_veh * math.cos(pitch) - x_veh * math.sin(pitch)
            x_cam = x_veh * math.cos(pitch) + y_veh * math.sin(pitch)

            if x_cam < 1.0:
                continue

            # Project to pixel
            u = int(self.focal_length * y_cam / x_cam + w / 2.0)
            v = int(self.focal_length * cam_z_offset / x_cam + h / 2.0)

            # Draw a thick point on the mask
            if 0 <= u < w and 0 <= v < h:
                thickness = max(3, int(20.0 / x_cam))  # thicker when close
                cv2_circle(mask, u, v, thickness)

        # Close gaps
        import cv2
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        return mask

    def _blend_masks(self, dsunet_mask: np.ndarray, map_mask: np.ndarray) -> np.ndarray:
        """Blend DSUNet mask with map-generated mask.

        The map mask provides the lane location (which lane to follow),
        while DSUNet provides the precise lane boundary shape.
        """
        # Union: take pixels from either mask
        blended = np.maximum(dsunet_mask, map_mask)
        return blended

    def get_lane_selection_info(self, vehicle_transform, route, lookahead_m=50.0) -> dict:
        """Get information about which lane the vehicle should be in.

        Returns:
            dict with:
            - current_lane: (road_id, lane_id)
            - target_lane: (road_id, lane_id) or None
            - mode: "LANE_KEEP" | "LANE_CHANGE" | "INTERSECTION"
            - next_maneuver: action string or None
            - distance_to_maneuver: float (m)
        """
        from planning.route_planner import (
            LANE_KEEP, LANE_CHANGE_LEFT, LANE_CHANGE_RIGHT,
            TURN_LEFT, TURN_RIGHT, ARRIVE, STRAIGHT,
        )

        wp = self.map.get_waypoint(vehicle_transform.location)
        current_lane = (wp.road_id, wp.lane_id)

        mode = self._determine_mode(vehicle_transform, route, lookahead_m)
        next_maneuver = None
        dist_to_maneuver = float('inf')
        target_lane = None

        vehicle_loc = vehicle_transform.location
        for entry in route:
            dist = entry.waypoint.transform.location.distance(vehicle_loc)
            if dist > lookahead_m:
                break
            if entry.action not in (LANE_KEEP, STRAIGHT):
                next_maneuver = entry.action
                dist_to_maneuver = dist
                target_lane = (entry.road_id, entry.lane_id)
                break

        return {
            "current_lane": current_lane,
            "target_lane": target_lane,
            "mode": mode,
            "next_maneuver": next_maneuver,
            "distance_to_maneuver": dist_to_maneuver if dist_to_maneuver < float('inf') else -1.0,
        }


def cv2_circle(mask: np.ndarray, cx: int, cy: int, radius: int):
    """Draw a filled circle on the mask without importing cv2 at module level."""
    import cv2
    cv2.circle(mask, (cx, cy), radius, 255, -1)
