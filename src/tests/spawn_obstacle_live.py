"""
Spawn obstacle during driving — ทดสอบ obstacle avoidance แบบ live

รัน main.py แล้ว spawn obstacle 30m ข้างหน้ารถหลังจาก 5 วินาที
เก็บ metrics การหลบและสร้าง report
"""
import carla
import math
import time
import threading
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger(__name__)


def spawn_obstacle_ahead(client, delay_s=5.0, distance_m=30.0):
    """Spawn obstacle ข้างหน้า ego vehicle หลัง delay วินาที
    Returns (obstacle, ego_id) tuple"""
    time.sleep(delay_s)

    world = client.get_world()
    actors = world.get_actors()

    # Find ego vehicle — look for lincoln.mkz that's moving fastest
    ego = None
    best_speed = 0
    for a in actors:
        if a.type_id.startswith('vehicle.lincoln'):
            try:
                v = a.get_velocity()
                speed = math.sqrt(v.x**2 + v.y**2)
                if speed > best_speed:
                    best_speed = speed
                    ego = a
            except Exception:
                pass

    if not ego:
        # Fallback: any moving vehicle
        for a in actors:
            if a.type_id.startswith('vehicle'):
                try:
                    v = a.get_velocity()
                    if math.sqrt(v.x**2 + v.y**2) > 1.0:
                        ego = a
                        break
                except Exception:
                    pass

    if not ego:
        logger.error("No ego vehicle found")
        return None, None

    ego_id = ego.id
    logger.info(f"Ego vehicle found: ID={ego_id} type={ego.type_id}")

    # Use CARLA waypoint to find road ahead
    tf = ego.get_transform()
    map_ = world.get_map()
    waypoint = map_.get_waypoint(tf.location, project_to_road=True)

    next_waypoints = waypoint.next(distance_m)
    if next_waypoints:
        target_wp = next_waypoints[0]
        obs_loc = target_wp.transform.location
        obs_rot = target_wp.transform.rotation
    else:
        fwd = tf.rotation.get_forward_vector()
        obs_loc = tf.location
        obs_loc.x += fwd.x * distance_m
        obs_loc.y += fwd.y * distance_m
        obs_loc.z += 0.5
        obs_rot = tf.rotation

    obs_loc.z += 0.5

    bp = world.get_blueprint_library().find('vehicle.tesla.model3')
    obs_tf = carla.Transform(obs_loc, obs_rot)

    try:
        obs = world.spawn_actor(bp, obs_tf)
        # Wait for obstacle to settle in world
        time.sleep(1.0)
        # Verify obstacle exists and has valid location
        obs_loc_check = obs.get_location()
        if obs_loc_check.x == 0 and obs_loc_check.y == 0:
            logger.warning("Obstacle at (0,0) — may not have spawned properly")
            time.sleep(1.0)
            obs_loc_check = obs.get_location()
        logger.info(f"Obstacle spawned {distance_m}m ahead at {obs_loc_check} (ego_id={ego_id})")
        return obs, ego_id
    except Exception as e:
        logger.error(f"Failed to spawn obstacle: {e}")
        obs_loc.z += 1.0
        obs_tf = carla.Transform(obs_loc, obs_rot)
        try:
            obs = world.spawn_actor(bp, obs_tf)
            time.sleep(1.0)
            logger.info(f"Obstacle spawned (z-adjusted) at {obs.get_location()}")
            return obs, ego_id
        except Exception as e2:
            logger.error(f"Spawn failed even with z adjust: {e2}")
            return None, ego_id


def monitor_avoidance(client, obstacle, ego_id, duration_s=15.0):
    """Monitor ego vehicle while approaching obstacle"""
    if not obstacle:
        return

    world = client.get_world()
    start = time.time()
    min_dist = float('inf')
    min_lat = float('inf')
    collision = False
    speed_at_pass = 0.0

    while time.time() - start < duration_s:
        actors = world.get_actors()
        ego = None
        # Find ego by ID (most reliable)
        if ego_id:
            try:
                ego = actors.find(ego_id)
            except Exception:
                pass

        # Fallback: lincoln.mkz that's moving fastest
        if not ego:
            best_speed = 0
            for a in actors:
                if a.type_id.startswith('vehicle.lincoln') and a.id != obstacle.id:
                    try:
                        v = a.get_velocity()
                        speed = math.sqrt(v.x**2 + v.y**2)
                        if speed > best_speed:
                            best_speed = speed
                            ego = a
                    except Exception:
                        pass

        if not ego:
            time.sleep(0.1)
            continue

        # Debug: log first iteration
        if min_dist == float('inf'):
            ego_loc = ego.get_location()
            obs_loc = obstacle.get_location()
            d = math.sqrt((ego_loc.x - obs_loc.x)**2 + (ego_loc.y - obs_loc.y)**2)
            logger.info(f"Monitor start: ego_id={ego.id} ego_pos=({ego_loc.x:.1f},{ego_loc.y:.1f}) "
                       f"obs_pos=({obs_loc.x:.1f},{obs_loc.y:.1f}) dist={d:.1f}m")

        if not ego:
            time.sleep(0.1)
            continue

        ego_loc = ego.get_location()
        obs_loc = obstacle.get_location()
        dx = obs_loc.x - ego_loc.x
        dy = obs_loc.y - ego_loc.y
        dist = math.sqrt(dx**2 + dy**2)
        min_dist = min(min_dist, dist)

        # Lateral distance
        ego_rot = ego.get_transform().rotation
        right = ego_rot.get_right_vector()
        lat = abs(dx * right.x + dy * right.y)
        min_lat = min(min_lat, lat)

        # Forward distance
        fwd = ego_rot.get_forward_vector()
        fwd_dist = dx * fwd.x + dy * fwd.y

        v = ego.get_velocity()
        speed = math.sqrt(v.x**2 + v.y**2) * 3.6

        if dist < 2.0:
            collision = True
            logger.warning(f"COLLISION! dist={dist:.2f}m")
            speed_at_pass = speed
            break

        if fwd_dist < -3:
            logger.info(f"Ego passed obstacle. min_dist={min_dist:.2f}m min_lat={min_lat:.2f}m")
            speed_at_pass = speed
            break

        time.sleep(0.1)

    # Cleanup
    try:
        obstacle.destroy()
    except Exception:
        pass

    return {
        'collision': collision,
        'min_distance_m': min_dist,
        'min_lateral_m': min_lat,
        'speed_at_pass_kmh': speed_at_pass,
    }


if __name__ == '__main__':
    client = carla.Client('localhost', 2000)
    client.set_timeout(10.0)

    print("Spawning obstacle in 5 seconds...")
    obs, ego_id = spawn_obstacle_ahead(client, delay_s=5.0, distance_m=30.0)
    if obs:
        result = monitor_avoidance(client, obs, ego_id, duration_s=20.0)
        print(f"\nResult: {result}")
    else:
        print("Failed to spawn obstacle")
