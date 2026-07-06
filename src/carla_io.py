"""
CARLA I/O — หลักเหตุผล: ดึง waypoints และ geometry จาก CARLA แยกจาก pipeline
"""
import math
from typing import List, Optional, Tuple

import carla
import numpy as np

from config import NUM_WP, WP_STEP


def get_waypoints(vehicle: carla.Actor, cmap: carla.Map) -> List[carla.Waypoint]:
    """Waypoints ข้างหน้าตามถนน (สำหรับ fusion กับ lane detection)."""
    vt = vehicle.get_transform()
    wp = cmap.get_waypoint(vt.location, project_to_road=True,
                           lane_type=carla.LaneType.Driving)
    if wp is None:
        return []
    pts: List[carla.Waypoint] = []
    cur = wp
    for _ in range(NUM_WP):
        pts.append(cur)
        nxt = cur.next(WP_STEP)
        if not nxt:
            break
        cur = nxt[0]
    return pts


def get_waypoints_all_lanes(
    vehicle: carla.Actor,
    cmap: carla.Map,
    max_s_m: float = 500.0,
    step_m: float = 2.0,
) -> Tuple[List[Tuple[float, float]], List[Tuple[float, float]], List[Tuple[float, float]]]:
    """
    Waypoints รอบตัวรถ ในรัศมี max_s_m (m) — ใช้ vector ถนนทั้งหมดรวมทุกเลน.
    คืน (ego_path, left_lane_path, right_lane_path) แต่ละอันเป็น list of (s, lat) ใน vehicle frame.
    """
    vt = vehicle.get_transform()
    wp = cmap.get_waypoint(vt.location, project_to_road=True,
                           lane_type=carla.LaneType.Driving)
    if wp is None:
        return [], [], []

    loc = vt.location
    rot = vt.rotation
    xv, yv = loc.x, loc.y
    psi_v = math.radians(rot.yaw)
    cos_p = math.cos(psi_v)
    sin_p = math.sin(psi_v)

    def to_s_lat(xw: float, yw: float) -> Tuple[float, float]:
        dx, dy = xw - xv, yw - yv
        forward = dx * cos_p + dy * sin_p
        lat = -dx * sin_p + dy * cos_p
        return (forward, lat)

    n_steps = max(10, int(max_s_m / step_m))
    ego_path: List[Tuple[float, float]] = []
    left_path: List[Tuple[float, float]] = []
    right_path: List[Tuple[float, float]] = []

    cur = wp
    for _ in range(n_steps):
        xw, yw = cur.transform.location.x, cur.transform.location.y
        s, lat = to_s_lat(xw, yw)
        if s >= -0.5:
            ego_path.append((s, lat))

        left_wp = cur.get_left_lane() if hasattr(cur, "get_left_lane") else None
        if left_wp is not None and left_wp.lane_type == carla.LaneType.Driving:
            xl, yl = left_wp.transform.location.x, left_wp.transform.location.y
            sl, latl = to_s_lat(xl, yl)
            if sl >= -0.5:
                left_path.append((sl, latl))
        else:
            left_path.append((s, lat - 3.5))  # fallback offset

        right_wp = cur.get_right_lane() if hasattr(cur, "get_right_lane") else None
        if right_wp is not None and right_wp.lane_type == carla.LaneType.Driving:
            xr, yr = right_wp.transform.location.x, right_wp.transform.location.y
            sr, latr = to_s_lat(xr, yr)
            if sr >= -0.5:
                right_path.append((sr, latr))
        else:
            right_path.append((s, lat + 3.5))  # fallback offset

        nxt = cur.next(step_m)
        if not nxt:
            break
        cur = nxt[0]
        if ego_path and s >= max_s_m:
            break

    return ego_path, left_path, right_path


def waypoints_to_cte_heading(
    vehicle_transform: carla.Transform,
    waypoints: List[carla.Waypoint],
) -> Optional[Tuple[float, float, float]]:
    """
    คืน (cte_m, heading_err_rad, curvature) ใน vehicle frame.
    คืน None ถ้า waypoints ไม่พอ.
    """
    if not waypoints or len(waypoints) < 2:
        return None
    loc = vehicle_transform.location
    rot = vehicle_transform.rotation
    xv, yv = loc.x, loc.y
    psi_v = math.radians(rot.yaw)
    wp_pts = np.array([
        (w.transform.location.x, w.transform.location.y,
         math.radians(w.transform.rotation.yaw))
        for w in waypoints
    ])
    xy = wp_pts[:, :2]
    pv = np.array([xv, yv])
    seg_vec = np.diff(xy, axis=0)
    seg_len = np.sqrt(np.sum(seg_vec ** 2, axis=1)) + 1e-6
    best_cte = 1e6
    best_head = 0.0
    best_curv = 0.0

    for i in range(len(xy) - 1):
        p0, p1 = xy[i], xy[i + 1]
        seg = p1 - p0
        t_par = np.dot(pv - p0, seg) / (seg_len[i] ** 2)
        if t_par > 1.2:
            continue
        proj = p0 + np.clip(t_par, 0, 1) * seg
        d = pv - proj
        cte_m = math.sqrt(d[0] ** 2 + d[1] ** 2)
        seg_a = math.atan2(seg[1], seg[0])
        cross = seg[0] * d[1] - seg[1] * d[0]
        if cross < 0:
            cte_m = -cte_m
        if abs(cte_m) < abs(best_cte) or best_cte > 1e5:
            best_cte = cte_m
            best_head = math.atan2(
                math.sin(psi_v - seg_a), math.cos(psi_v - seg_a)
            )
            if i + 2 < len(wp_pts):
                seg2_a = math.atan2(
                    wp_pts[i + 2, 1] - wp_pts[i + 1, 1],
                    wp_pts[i + 2, 0] - wp_pts[i + 1, 0],
                )
                best_curv = math.atan2(
                    math.sin(seg2_a - seg_a),
                    math.cos(seg2_a - seg_a),
                ) / max(seg_len[i], 0.5)

    if best_cte > 1e5 and len(xy) >= 2:
        seg_a = math.atan2(xy[1, 1] - xy[0, 1], xy[1, 0] - xy[0, 0])
        best_cte = 0.0
        best_head = math.atan2(
            math.sin(psi_v - seg_a), math.cos(psi_v - seg_a)
        )
        best_curv = 0.0

    return (
        float(np.clip(best_cte, -5.0, 5.0)),
        float(np.clip(best_head, -0.4, 0.4)),
        float(np.clip(best_curv, -0.08, 0.08)),
    )


def waypoint_lookahead_curvature(
    waypoints: List[carla.Waypoint],
    lookahead_m: float = 30.0,
    step_m: float = 2.0,
) -> float:
    """
    Compute max absolute curvature from waypoints within lookahead distance.
    This allows anticipating sharp turns before UNet can see them.
    Returns max |curvature| in 1/m.
    """
    if not waypoints or len(waypoints) < 3:
        return 0.0

    # Extract yaw angles and cumulative distance
    yaws = []
    dists = []
    cum_dist = 0.0
    prev_loc = None
    for w in waypoints:
        loc = w.transform.location
        yaw = math.radians(w.transform.rotation.yaw)
        yaws.append(yaw)
        if prev_loc is not None:
            d = math.sqrt((loc.x - prev_loc.x)**2 + (loc.y - prev_loc.y)**2)
            cum_dist += d
        dists.append(cum_dist)
        prev_loc = loc
        if cum_dist > lookahead_m:
            break

    max_curv = 0.0
    for i in range(len(yaws) - 1):
        ds = dists[i + 1] - dists[i]
        if ds < 0.1:
            continue
        dyaw = math.atan2(math.sin(yaws[i + 1] - yaws[i]),
                          math.cos(yaws[i + 1] - yaws[i]))
        curv = abs(dyaw / ds)
        if curv > max_curv:
            max_curv = curv

    return float(min(max_curv, 0.5))  # cap at 0.5 (2m radius, extreme)


def waypoints_to_path(
    vehicle_transform: carla.Transform,
    waypoints: List[carla.Waypoint],
) -> List[Tuple[float, float]]:
    """
    แปลง waypoints เป็น path (s, lat) ใน vehicle frame สำหรับวาดใน BEV.
    s = ระยะไปข้างหน้า, lat = เบี่ยงซ้าย/ขวาจากศูนย์.
    """
    if not waypoints:
        return []
    loc = vehicle_transform.location
    rot = vehicle_transform.rotation
    xv, yv = loc.x, loc.y
    psi_v = math.radians(rot.yaw)
    cos_p = math.cos(psi_v)
    sin_p = math.sin(psi_v)
    out: List[Tuple[float, float]] = []
    for w in waypoints:
        xw, yw = w.transform.location.x, w.transform.location.y
        dx, dy = xw - xv, yw - yv
        forward = dx * cos_p + dy * sin_p
        lat = -dx * sin_p + dy * cos_p
        if forward >= -0.5:
            out.append((forward, lat))
    return out


def cleanup(world: carla.World) -> None:
    """ลบ vehicles และ sensors ทั้งหมด (ก่อน spawn ใหม่)."""
    for a in world.get_actors().filter("vehicle.*"):
        a.destroy()
    for a in world.get_actors().filter("sensor.*"):
        a.destroy()
