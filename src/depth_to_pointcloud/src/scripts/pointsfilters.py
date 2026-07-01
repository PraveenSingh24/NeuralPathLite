"""""
#!/usr/bin/env python3
import numpy as np
from scipy.spatial import cKDTree
import time

class DuplicatePointFilter:
    
    #Removes duplicate or near-duplicate points in a 2D/3D point cloud
    #using spatial proximity (within a small radius).
    
    def __init__(self, threshold=0.02):
        
        #Args:
        #    threshold (float): distance below which two points are considered duplicates
        
        self.threshold = threshold

    def filter(self, points):
        if points is None or len(points) == 0:
            return points
        if len(points) < 2:
            return points

        tree = cKDTree(points)
        pairs = tree.query_pairs(self.threshold)
        if not pairs:
            return points

        # Keep only unique indices
        to_remove = set(j for _, j in pairs)
        mask = np.array([i not in to_remove for i in range(len(points))])
        return points[mask]


class TemporalConsistencyFilter:
    
    #Filters out dynamic (moving) points across consecutive frames.
    #Maintains a temporal buffer of previous point clouds and removes
    #points that appear inconsistently over time.
    
    def __init__(self, persistence=3, voxel_size=0.05):
        
        #Args:
        #    persistence (int): number of frames a point must persist to be considered static
        #    voxel_size (float): grid size for point matching
        
        self.persistence = persistence
        self.voxel_size = voxel_size
        self.history = {}  # voxel -> count
        self.last_cleanup = time.time()

    def filter(self, points):
        if points is None or len(points) == 0:
            return points

        # Quantize points into voxel grid
        qx = np.floor(points[:, 0] / self.voxel_size).astype(int)
        qy = np.floor(points[:, 1] / self.voxel_size).astype(int)
        voxels = list(zip(qx, qy))

        # Update persistence counts
        for v in voxels:
            self.history[v] = self.history.get(v, 0) + 1

        # Clean up occasionally
        now = time.time()
        if now - self.last_cleanup > 5.0:
            # Remove old low-frequency entries to prevent growth
            self.history = {v: c for v, c in self.history.items() if c < self.persistence * 5}
            self.last_cleanup = now

        # Keep only stable (persistent) points
        stable_mask = np.array([self.history.get(v, 0) >= self.persistence for v in voxels])
        return points[stable_mask]
"""""
#!/usr/bin/env python3
import numpy as np
from scipy.spatial import cKDTree
import time


class DuplicatePointFilter:
    """
    Removes duplicate or near-duplicate points in a 2D/3D point cloud
    using spatial proximity (within a small radius).
    """
    def __init__(self, threshold=0.02):
        """
        Args:
            threshold (float): distance below which two points are considered duplicates
        """
        self.threshold = threshold

    def filter(self, points):
        if points is None or len(points) == 0:
            return points
        if len(points) < 2:
            return points

        tree = cKDTree(points)
        pairs = tree.query_pairs(self.threshold)
        if not pairs:
            return points

        # Keep only unique indices
        to_remove = set(j for _, j in pairs)
        mask = np.array([i not in to_remove for i in range(len(points))])
        return points[mask]


class TemporalConsistencyFilter:
    """
    Filters out dynamic (moving) points across consecutive frames.
    Maintains a temporal buffer of previous point clouds and removes
    points that appear inconsistently over time.
    """
    def __init__(self, persistence=3, voxel_size=0.05):
        """
        Args:
            persistence (int): number of frames a point must persist to be considered static
            voxel_size (float): grid size for point matching
        """
        self.persistence = persistence
        self.voxel_size = voxel_size
        self.history = {}  # voxel -> count
        self.last_cleanup = time.time()

    def filter(self, points):
        if points is None or len(points) == 0:
            return points

        # Quantize points into voxel grid
        qx = np.floor(points[:, 0] / self.voxel_size).astype(int)
        qy = np.floor(points[:, 1] / self.voxel_size).astype(int)
        voxels = list(zip(qx, qy))

        # Update persistence counts
        for v in voxels:
            self.history[v] = self.history.get(v, 0) + 1

        # Periodic cleanup to limit memory usage
        now = time.time()
        if now - self.last_cleanup > 5.0:
            self.history = {v: c for v, c in self.history.items() if c < self.persistence * 5}
            self.last_cleanup = now

        # Keep only stable (persistent) points
        stable_mask = np.array([self.history.get(v, 0) >= self.persistence for v in voxels])
        return points[stable_mask]

class GlobalMap2D:
    """
    Grid-based global map that supports:
      - replace-nearby-before-insert (to avoid local accumulation)
      - returning only newly-updated points
      - pruning by age
    """
    def __init__(self, grid_size=0.05, rebuild_interval=10.0):
        self.grid_size = grid_size
        self.map_dict = {}  # (gx, gy) -> [x, y, last_update_time]
        self.last_rebuild = time.time()
        self.rebuild_interval = rebuild_interval

    def _point_to_key(self, x, y):
        return (int(np.floor(x / self.grid_size)), int(np.floor(y / self.grid_size)))

    def update(self, new_points):
        """Old API kept for compatibility: merges and returns whole map."""
        return self.update_and_get_updates(new_points)

    def update_and_get_updates(self, new_points, replace_radius=1):
        """
        Insert new_points into the grid, removing overlapping neighbors in a small radius.
        Returns: Nx2 numpy array of points that were newly inserted/updated (not the whole map).
        replace_radius: radius in grid-cells (integer) to treat as overlap; default 1 -> 3x3 neighborhood.
        """
        if new_points is None or len(new_points) == 0:
            return np.empty((0, 2))

        current_time = time.time()

        # Quantize to cell keys
        qx = np.floor(new_points[:, 0] / self.grid_size).astype(int)
        qy = np.floor(new_points[:, 1] / self.grid_size).astype(int)
        new_cells = list(zip(qx, qy))

        # Collect keys to delete: any existing key within +/- replace_radius of any new cell
        keys_to_delete = set()
        for nx, ny in set(new_cells):
            for dx in range(-replace_radius, replace_radius + 1):
                for dy in range(-replace_radius, replace_radius + 1):
                    key = (nx + dx, ny + dy)
                    if key in self.map_dict:
                        keys_to_delete.add(key)
        # Delete those keys
        for k in keys_to_delete:
            del self.map_dict[k]

        # Insert/update new cells (new observation dominates)
        updated_keys = set()
        for i, (kx, ky) in enumerate(new_cells):
            key = (int(kx), int(ky))
            self.map_dict[key] = [float(new_points[i, 0]), float(new_points[i, 1]), current_time]
            updated_keys.add(key)

        # Periodic rebuild (age-based)
        if current_time - self.last_rebuild > self.rebuild_interval:
            self._rebuild_map(current_time)
            self.last_rebuild = current_time

        # Return only newly-updated points (values for updated_keys)
        updated_list = [self.map_dict[k][:2] for k in updated_keys if k in self.map_dict]
        if not updated_list:
            return np.empty((0, 2))
        return np.array(updated_list)

    def prune_by_age(self, max_age=30.0):
        """Remove entries older than max_age (seconds)."""
        current_time = time.time()
        keys_to_keep = {}
        for k, v in self.map_dict.items():
            if (current_time - v[2]) <= max_age:
                keys_to_keep[k] = v
        self.map_dict = keys_to_keep

    def _rebuild_map(self, current_time, max_age=30.0):
        """Keep only entries younger than max_age (backwards-compatible helper)."""
        self.prune_by_age(max_age)

    def get_points(self):
        """Return all current map points as Nx2 array."""
        if not self.map_dict:
            return np.empty((0, 2))
        pts = np.array([[v[0], v[1]] for v in self.map_dict.values()])
        return pts


    def replace_front_points(self, new_points, pose, front_distance=5.0, width=2.0):
        """
        Replace old points in front of the robot with new points safely.
        """
        x0, y0, yaw = pose
        cos_y, sin_y = np.cos(yaw), np.sin(yaw)

        # Compute new point keys
        qx_new = np.floor(new_points[:, 0] / self.grid_size).astype(int)
        qy_new = np.floor(new_points[:, 1] / self.grid_size).astype(int)
        new_keys = set((qx_new[i], qy_new[i]) for i in range(len(new_points)))

        # Collect keys to remove first
        keys_to_remove = []
        map_items = list(self.map_dict.items())  # convert to list to avoid runtime error
        for key, val in map_items:
            x, y = val[0], val[1]

            # Transform point into robot frame
            dx = x - x0
            dy = y - y0
            x_r = dx * cos_y + dy * sin_y
            y_r = -dx * sin_y + dy * cos_y

            if 0 <= x_r <= front_distance and abs(y_r) <= width / 2.0:
                keys_to_remove.append(key)
                continue

            # Remove if collides with new points
            qx_old = int(np.floor(x / self.grid_size))
            qy_old = int(np.floor(y / self.grid_size))
            if (qx_old, qy_old) in new_keys:
                keys_to_remove.append(key)

        # Delete old points after iteration
        for k in keys_to_remove:
            self.map_dict.pop(k, None)

        # Add new points
        current_time = time.time()
        for i in range(len(new_points)):
            key = (qx_new[i], qy_new[i])
            self.map_dict[key] = [new_points[i, 0], new_points[i, 1], current_time]

        return new_points





