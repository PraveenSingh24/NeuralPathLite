#!/usr/bin/env python3
"""
Online2DMapper: State-Machine Navigation with Neural Path Planning
States: IDLE -> PLANNING -> NAVIGATING
"""

import os
import cv2
import json
import rospy
import numpy as np
import threading
from enum import Enum
from scipy.spatial import cKDTree
from geometry_msgs.msg import PointStamped, PoseStamped
from nav_msgs.msg import Path, Odometry
from sensor_msgs.msg import PointCloud2
import sensor_msgs.point_cloud2 as pc2

from pointsfilters import DuplicatePointFilter, TemporalConsistencyFilter


class NavState(Enum):
    IDLE = 0
    PLANNING = 1
    NAVIGATING = 2


class Online2DMapper:
    def __init__(self):
        rospy.init_node('online_2d_mapper', anonymous=True)

        # ---- Parameters ----
        self.grid_resolution = float(rospy.get_param("~grid_resolution", 0.05))
        self.grid_size_pixels = int(rospy.get_param("~grid_size_pixels", 500))
        self.center_pixel = self.grid_size_pixels // 2
        self.vehicle_width = float(rospy.get_param("~vehicle_width", 0.8))
        self.safety_margin = float(rospy.get_param("~safety_margin", 0.1))

        # ---- Map State ----
        self.global_points = []
        self.kdtree = None
        self.current_map_img = np.full(
            (self.grid_size_pixels, self.grid_size_pixels), 255, dtype=np.uint8
        )
        self.map_lock = threading.Lock()
        self.dup_threshold = 0.05
        self.points_processed = 0       # For incremental map building
        self.max_points = 50000         # Cap to prevent unbounded growth

        # ---- Navigation State (State Machine) ----
        self.state = NavState.IDLE
        self.goal = None
        self.current_pose = None
        self.current_goal_id = None

        self.waypoints = []
        self.current_waypoint_index = 0
        self.waypoint_lock = threading.Lock()
        self.waypoint_reach_radius = 0.5
        self.waypoint_reach_counter = 0

        # ---- Filters ----
        self.dup_filter = DuplicatePointFilter(threshold=self.dup_threshold)
        self.dynamic_filter = TemporalConsistencyFilter(persistence=3, voxel_size=0.1)

        # ---- ROS Setup ----
        self.sub_scan = rospy.Subscriber(
            '/registered_scan', PointCloud2, self.cloud_callback, queue_size=1
        )
        self.sub_goal = rospy.Subscriber(
            '/way_point', PointStamped, self.goal_callback, queue_size=1
        )
        self.sub_odom = rospy.Subscriber(
            '/state_estimation', Odometry, self.odom_callback, queue_size=10
        )
        self.pub_waypoint = rospy.Publisher('/wpts', PointStamped, queue_size=5)
        self.pub_path = rospy.Publisher('/radha', Path, queue_size=1, latch=True)

        # ---- Server Config ----
        self.remote_user = "praveen_k"
        self.remote_host = "172.26.185.47"
        self.remote_dir = "/home/praveen_k/ros_bridge/shared"
        self.local_tmp = "/home/praveensingh/ros_tmp"
        os.makedirs(self.local_tmp, exist_ok=True)

        # ---- Threads & Timers ----
        self.plan_event = threading.Event()

        rospy.Timer(rospy.Duration(3.0), self.timer_build_map)

        threading.Thread(target=self.planner_loop, daemon=True).start()
        threading.Thread(target=self.waypoint_publisher_loop, daemon=True).start()

        self.cleanup_tmp_folder()
        rospy.on_shutdown(self.cleanup_tmp_folder)

        rospy.loginfo("✅ Online2DMapper initialized. State: IDLE")

    # ================================================================
    #  HELPERS
    # ================================================================
    def world_to_pixel(self, x, y):
        px = int(x / self.grid_resolution + self.center_pixel)
        py = int(y / self.grid_resolution + self.center_pixel)
        px = max(0, min(self.grid_size_pixels - 1, px))
        py = max(0, min(self.grid_size_pixels - 1, py))
        return px, py

    def is_path_blocked(self, p1, p2):
        """Vectorized collision check using cv2.line — much faster than pixel loop."""
        u1, v1 = self.world_to_pixel(p1[0], p1[1])
        u2, v2 = self.world_to_pixel(p2[0], p2[1])
        mask = np.zeros_like(self.current_map_img)
        cv2.line(mask, (u1, v1), (u2, v2), 255, thickness=3)
        return bool(np.any((mask == 255) & (self.current_map_img == 0)))

    def extract_corner_waypoints(self, path_points, angle_thresh_deg=150, min_dist=1.0):
        """Reduce dense path to corners with minimum spacing."""
        if not path_points or len(path_points) < 3:
            return path_points

        corners = [tuple(path_points[0])]

        for i in range(1, len(path_points) - 1):
            p_prev = np.array(path_points[i - 1])
            p_curr = np.array(path_points[i])
            p_next = np.array(path_points[i + 1])

            v1 = p_prev - p_curr
            v2 = p_next - p_curr
            n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
            if n1 < 1e-6 or n2 < 1e-6:
                continue

            cos_theta = np.clip(np.dot(v1, v2) / (n1 * n2), -1.0, 1.0)
            ang = np.degrees(np.arccos(cos_theta))

            if ang < angle_thresh_deg:
                dist_from_last = np.linalg.norm(p_curr - np.array(corners[-1]))
                if dist_from_last > min_dist:
                    corners.append(tuple(path_points[i]))

        corners.append(tuple(path_points[-1]))
        return list(dict.fromkeys(corners))

    def remove_backtracking_points(self, path):
        """Remove initial backward segments."""
        if len(path) < 3:
            return path
        start = np.array(path[0])
        lookahead = np.array(path[min(5, len(path) - 1)]) - start
        norm = np.linalg.norm(lookahead)
        if norm < 1e-3:
            return path
        fwd = lookahead / norm
        seg = np.array(path[1]) - start
        seg_norm = np.linalg.norm(seg)
        if seg_norm > 0 and np.dot(seg / seg_norm, fwd) < 0.2:
            return [path[0]] + path[2:]
        return path

    def cleanup_tmp_folder(self):
        try:
            for f in os.listdir(self.local_tmp):
                os.remove(os.path.join(self.local_tmp, f))
        except Exception:
            pass

    # ================================================================
    #  CALLBACKS
    # ================================================================
    def cloud_callback(self, msg):
        xyz = np.array([
            [p[0], p[1], p[2]]
            for p in pc2.read_points(msg, field_names=("x", "y", "z"), skip_nans=True)
        ])
        if xyz.size == 0:
            return
        xyz = self.dup_filter.filter(xyz)
        static_xyz = self.dynamic_filter.filter(xyz)
        pts = static_xyz[:, :2]
        if pts.size == 0:
            return

        with self.map_lock:
            added = 0
            if self.kdtree is None:
                self.global_points.extend(pts.tolist())
                added = len(pts)
            else:
                dists, _ = self.kdtree.query(pts, distance_upper_bound=self.dup_threshold)
                for i, dist in enumerate(dists):
                    if np.isinf(dist):
                        self.global_points.append(pts[i].tolist())
                        added += 1

            # Cap points to prevent unbounded growth
            if len(self.global_points) > self.max_points:
                self.global_points = self.global_points[-self.max_points:]

            if added > 0:
                self.kdtree = cKDTree(np.array(self.global_points))

    def goal_callback(self, msg):
        self.goal = (float(msg.point.x), float(msg.point.y))
        self.current_goal_id = rospy.Time.now().to_sec()

        with self.waypoint_lock:
            self.waypoints = []
            self.current_waypoint_index = 0

        # Clear old path visualization
        empty = Path()
        empty.header.frame_id = "map"
        self.pub_path.publish(empty)

        rospy.loginfo(f"[Goal] New goal: {self.goal}")
        self.state = NavState.PLANNING
        self.plan_event.set()

    def odom_callback(self, msg):
        x, y = msg.pose.pose.position.x, msg.pose.pose.position.y
        self.current_pose = (x, y)

        if self.state != NavState.NAVIGATING:
            return

        # --- Goal reached? ---
        if self.goal is not None:
            if np.hypot(x - self.goal[0], y - self.goal[1]) < 0.5:
                rospy.loginfo("✅ GOAL REACHED. -> IDLE")
                self.state = NavState.IDLE
                self.goal = None
                self.current_goal_id = None
                with self.waypoint_lock:
                    self.waypoints = []
                    self.current_waypoint_index = 0
                return

        # --- Waypoint reached? ---
        self._check_waypoint_reached(x, y)

    # ================================================================
    #  MAP BUILDER (Timer - 3s)
    # ================================================================
    def timer_build_map(self, event):
        # Only rebuild if new points were added
        with self.map_lock:
            n_pts = len(self.global_points)
            if n_pts < 5:
                return
            if n_pts == self.points_processed:
                return  # No new data, skip

            # Copy points under lock, release lock before heavy computation
            pts_world = np.array(self.global_points)
            self.points_processed = n_pts

        # --- Heavy computation OUTSIDE the lock ---
        img = np.full((self.grid_size_pixels, self.grid_size_pixels), 255, dtype=np.uint8)
        pts_pix = np.floor(pts_world / self.grid_resolution).astype(int) + self.center_pixel

        valid = (
            (pts_pix[:, 0] >= 0) & (pts_pix[:, 0] < self.grid_size_pixels) &
            (pts_pix[:, 1] >= 0) & (pts_pix[:, 1] < self.grid_size_pixels)
        )
        pts_pix = pts_pix[valid]
        img[pts_pix[:, 1], pts_pix[:, 0]] = 0

        inflate_px = int(np.ceil((self.vehicle_width / 2.0 + self.safety_margin) / self.grid_resolution))
        if inflate_px > 0:
            kernel = cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE, (2 * inflate_px + 1, 2 * inflate_px + 1)
            )
            img = cv2.erode(img, kernel, iterations=1)

        # Atomic update — only the assignment holds the implicit GIL
        self.current_map_img = img

    # ================================================================
    #  PLANNER LOOP (Background Thread)
    # ================================================================
    def planner_loop(self):
        while not rospy.is_shutdown():
            self.plan_event.wait()
            self.plan_event.clear()

            if self.state != NavState.PLANNING:
                continue
            if self.current_pose is None or self.goal is None:
                continue

            snap_start = tuple(self.current_pose)
            snap_goal = self.goal
            snap_id = self.current_goal_id

            rospy.loginfo(f"[Planner] Planning... (goal_id={snap_id})")

            try:
                # 1. Prepare map image
                with self.map_lock:
                    map_img = self.current_map_img.copy()

                map_color = cv2.cvtColor(map_img, cv2.COLOR_GRAY2BGR)
                sx, sy = self.world_to_pixel(*snap_start)
                gx, gy = self.world_to_pixel(*snap_goal)
                cv2.circle(map_color, (sx, sy), 15, (0, 255, 0), -1)
                cv2.circle(map_color, (gx, gy), 15, (255, 0, 0), -1)

                # 2. Save & upload
                task_id = str(int(rospy.Time.now().to_sec()))
                local_map = os.path.join(self.local_tmp, f"input_{task_id}.png")
                cv2.imwrite(local_map, map_color)

                local_task = os.path.join(self.local_tmp, f"task_{task_id}.json")
                with open(local_task, "w") as f:
                    json.dump({
                        "start": list(snap_start),
                        "goal": list(snap_goal),
                        "task_id": task_id,
                        "map_image_file": f"input_{task_id}.png"
                    }, f)

                remote_map = f"{self.remote_dir}/input_{task_id}.png"
                os.system(f"scp -q {local_map} {self.remote_user}@{self.remote_host}:{remote_map}")
                os.system(f"scp -q {local_task} {self.remote_user}@{self.remote_host}:{self.remote_dir}/task.json.tmp")
                os.system(f"ssh {self.remote_user}@{self.remote_host} 'mv {self.remote_dir}/task.json.tmp {self.remote_dir}/task.json'")

                # 3. Wait for result
                local_result = os.path.join(self.local_tmp, "result.json")
                remote_result = f"{self.remote_dir}/result.json"
                if os.path.exists(local_result):
                    os.remove(local_result)

                start_wait = rospy.Time.now()
                valid = False
                while (rospy.Time.now() - start_wait).to_sec() < 60:
                    if self.current_goal_id != snap_id:
                        raise Exception("Goal Obsolete")
                    os.system(f"scp -q {self.remote_user}@{self.remote_host}:{remote_result} {local_result} 2>/dev/null")
                    if os.path.exists(local_result):
                        try:
                            with open(local_result) as f:
                                res = json.load(f)
                            if str(res.get("task_id")) == task_id:
                                valid = True
                                break
                        except Exception:
                            pass
                    rospy.sleep(0.2)

                if not valid:
                    rospy.logwarn("[Planner] Timeout. Retrying...")
                    self.plan_event.set()
                    continue

                # 4. Process path
                with open(local_result) as f:
                    res = json.load(f)
                world_path = res.get("path_world", [])
                if not world_path or len(world_path) < 2:
                    rospy.logwarn("[Planner] Invalid path. Retrying...")
                    self.plan_event.set()
                    continue

                world_path[0] = list(snap_start)
                world_path[-1] = list(snap_goal)

                world_path = self.remove_backtracking_points(world_path)
                sparse_wps = self.extract_corner_waypoints(world_path)

                rospy.loginfo(f"[Planner] {len(world_path)} dense -> {len(sparse_wps)} waypoints")

                # 5. Find starting waypoint index
                robot = np.array(self.current_pose)
                start_idx = 0
                min_d = float('inf')
                for i, wp in enumerate(sparse_wps):
                    d = np.linalg.norm(np.array(wp) - robot)
                    if d < min_d:
                        min_d = d
                        start_idx = i

                # Skip the one we're already at
                if min_d < 0.5 and start_idx + 1 < len(sparse_wps):
                    start_idx += 1

                # 6. Verify robot -> first waypoint is collision-free
                target_wp = sparse_wps[start_idx]
                if self.is_path_blocked(self.current_pose, target_wp):
                    rospy.logwarn("[Planner] First waypoint blocked! Retrying...")
                    rospy.sleep(1.0)
                    self.plan_event.set()
                    continue

                # 7. Commit waypoints AND publish viz together (both update or neither)
                with self.waypoint_lock:
                    self.waypoints = sparse_wps
                    self.current_waypoint_index = start_idx
                    self.waypoint_reach_counter = 0

                # Publish dense path visualization
                path_msg = Path()
                path_msg.header.frame_id = "map"
                path_msg.header.stamp = rospy.Time.now()
                for wx, wy in world_path:
                    p = PoseStamped()
                    p.header = path_msg.header
                    p.pose.position.x = wx
                    p.pose.position.y = wy
                    p.pose.orientation.w = 1.0
                    path_msg.poses.append(p)
                self.pub_path.publish(path_msg)

                self.state = NavState.NAVIGATING
                rospy.loginfo(f"[Nav] -> NAVIGATING. Waypoint {start_idx}/{len(sparse_wps)}: ({target_wp[0]:.2f}, {target_wp[1]:.2f})")

            except Exception as e:
                if "Goal Obsolete" not in str(e):
                    rospy.logerr(f"[Planner] Error: {e}")

    # ================================================================
    #  WAYPOINT PUBLISHER (5 Hz, only during NAVIGATING)
    # ================================================================
    def waypoint_publisher_loop(self):
        r = rospy.Rate(2)  # 2Hz — sufficient for collision monitoring
        while not rospy.is_shutdown():
            if self.state == NavState.NAVIGATING and self.current_pose is not None:
                with self.waypoint_lock:
                    if self.waypoints and self.current_waypoint_index < len(self.waypoints):
                        wx, wy = self.waypoints[self.current_waypoint_index]
                    else:
                        wx, wy = None, None

                if wx is not None:
                    # Continuous collision check: robot -> current waypoint
                    blocked = self.is_path_blocked(self.current_pose, (wx, wy))
                    if blocked:
                        rospy.logwarn("🚨 Path to waypoint blocked! -> PLANNING")
                        self.state = NavState.PLANNING
                        self.plan_event.set()
                    else:
                        # Path confirmed clear -> publish waypoint
                        msg = PointStamped()
                        msg.header.frame_id = "map"
                        msg.header.stamp = rospy.Time.now()
                        msg.point.x, msg.point.y = wx, wy
                        self.pub_waypoint.publish(msg)
            r.sleep()

    # ================================================================
    #  WAYPOINT REACHED
    # ================================================================
    def _check_waypoint_reached(self, x, y):
        with self.waypoint_lock:
            if not self.waypoints or self.current_waypoint_index >= len(self.waypoints):
                return
            target = self.waypoints[self.current_waypoint_index]
            dist = np.linalg.norm([x - target[0], y - target[1]])

            if dist < self.waypoint_reach_radius:
                self.waypoint_reach_counter += 1
                if self.waypoint_reach_counter >= 3:
                    rospy.loginfo(f"🎯 Reached waypoint {self.current_waypoint_index}")
                    self.current_waypoint_index += 1
                    self.waypoint_reach_counter = 0
                    if self.current_waypoint_index < len(self.waypoints):
                        nxt = self.waypoints[self.current_waypoint_index]
                        rospy.loginfo(f"➡️ Next: waypoint {self.current_waypoint_index} ({nxt[0]:.2f}, {nxt[1]:.2f})")
                    else:
                        rospy.loginfo("🏁 All waypoints reached!")
            else:
                self.waypoint_reach_counter = 0


if __name__ == "__main__":
    try:
        Online2DMapper()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass