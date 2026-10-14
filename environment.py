"""
Continuous 3D Environment for drone disaster simulation.
Buildings, disaster zones, obstacles, safe zone — all in continuous coordinates.
"""

import numpy as np
import sys
sys.path.insert(0, r'c:\robo\project')
from config import *


class Building:
    """Rectangular building obstacle in continuous 3D space."""

    def __init__(self, x, y, width, depth, height):
        """
        Args:
            x, y: centre of footprint (metres)
            width: extent along x-axis
            depth: extent along y-axis
            height: vertical extent (metres)
        """
        self.x = x
        self.y = y
        self.width = width
        self.depth = depth
        self.height = height
        self.x_min = x - width / 2
        self.x_max = x + width / 2
        self.y_min = y - depth / 2
        self.y_max = y + depth / 2

    def contains_point(self, px, py, pz=None):
        inside_2d = (self.x_min <= px <= self.x_max and
                     self.y_min <= py <= self.y_max)
        if pz is not None:
            return inside_2d and 0 <= pz <= self.height
        return inside_2d

    def get_footprint_corners(self):
        return [
            (self.x_min, self.y_min),
            (self.x_max, self.y_min),
            (self.x_max, self.y_max),
            (self.x_min, self.y_max),
        ]

    def distance_to_point(self, px, py, pz=None):
        dx = max(self.x_min - px, 0, px - self.x_max)
        dy = max(self.y_min - py, 0, py - self.y_max)
        d2 = np.sqrt(dx ** 2 + dy ** 2)
        if pz is not None and 0 < pz < self.height:
            return d2
        if pz is not None:
            dz = max(0 - pz, 0, pz - self.height) if pz is not None else 0
            return np.sqrt(d2 ** 2 + dz ** 2)
        return d2


class DisasterZone:
    """Circular disaster / fire zone."""

    def __init__(self, center_x, center_y, radius, intensity=1.0):
        self.center_x = center_x
        self.center_y = center_y
        self.radius = radius
        self.intensity = intensity

    def contains_point(self, px, py):
        return np.sqrt((px - self.center_x) ** 2 + (py - self.center_y) ** 2) <= self.radius

    def distance_to_boundary(self, px, py):
        d = np.sqrt((px - self.center_x) ** 2 + (py - self.center_y) ** 2)
        return d - self.radius


class Obstacle:
    """Generic cylindrical obstacle (tree, debris)."""

    def __init__(self, x, y, radius, height):
        self.x = x
        self.y = y
        self.radius = radius
        self.height = height

    def contains_point(self, px, py, pz=None):
        d = np.sqrt((px - self.x) ** 2 + (py - self.y) ** 2)
        inside_2d = d <= self.radius
        if pz is not None:
            return inside_2d and 0 <= pz <= self.height
        return inside_2d

    def distance_to_point(self, px, py):
        return max(0, np.sqrt((px - self.x) ** 2 + (py - self.y) ** 2) - self.radius)


class Environment:
    """Continuous 3D disaster environment (500 m × 500 m)."""

    def __init__(self, seed=RANDOM_SEED):
        self.rng = np.random.RandomState(seed)
        self.size = ENVIRONMENT_SIZE
        self.buildings = []
        self.disaster_zones = []
        self.obstacles = []
        self.safe_zone_center = SAFE_ZONE_CENTER.copy()
        self.safe_zone_radius = SAFE_ZONE_RADIUS
        self.base_station = BASE_STATION_POS.copy()
        self.dynamic_obstacles = []
        self._generate()

    # ------------------------------------------------------------------
    def _generate(self):
        self._generate_buildings()
        self._generate_disaster_zones()
        self._generate_trees()

    def _generate_buildings(self):
        for _ in range(NUM_BUILDINGS):
            for _attempt in range(200):
                w = self.rng.uniform(BUILDING_MIN_SIZE, BUILDING_MAX_SIZE)
                d = self.rng.uniform(BUILDING_MIN_SIZE, BUILDING_MAX_SIZE)
                h = self.rng.uniform(BUILDING_MIN_HEIGHT, BUILDING_MAX_HEIGHT)
                x = self.rng.uniform(40, self.size - 40)
                y = self.rng.uniform(40, self.size - 40)

                # Not overlapping safe zone
                if np.sqrt((x - self.safe_zone_center[0]) ** 2 +
                           (y - self.safe_zone_center[1]) ** 2) < self.safe_zone_radius + 40:
                    continue
                # Not overlapping base
                if np.sqrt((x - self.base_station[0]) ** 2 +
                           (y - self.base_station[1]) ** 2) < 50:
                    continue
                # Not overlapping other buildings
                overlap = False
                for b in self.buildings:
                    if (abs(x - b.x) < (w + b.width) / 2 + 8 and
                            abs(y - b.y) < (d + b.depth) / 2 + 8):
                        overlap = True
                        break
                if overlap:
                    continue

                self.buildings.append(Building(x, y, w, d, h))
                break

    def _generate_disaster_zones(self):
        for _ in range(NUM_DISASTER_ZONES):
            if self.buildings:
                b = self.buildings[self.rng.randint(len(self.buildings))]
                cx = b.x + self.rng.uniform(-20, 20)
                cy = b.y + self.rng.uniform(-20, 20)
            else:
                cx = self.rng.uniform(100, 400)
                cy = self.rng.uniform(100, 400)
            self.disaster_zones.append(
                DisasterZone(cx, cy, DISASTER_ZONE_RADIUS, intensity=self.rng.uniform(0.6, 1.0))
            )

    def _generate_trees(self):
        for _ in range(NUM_TREES):
            for _attempt in range(50):
                x = self.rng.uniform(20, self.size - 20)
                y = self.rng.uniform(20, self.size - 20)
                if any(b.contains_point(x, y) for b in self.buildings):
                    continue
                self.obstacles.append(Obstacle(x, y, TREE_RADIUS, TREE_HEIGHT))
                break

    # ------------------------------------------------------------------
    def is_free(self, x, y, z=0.0, inflation=0.0):
        """Check whether (x, y, z) is collision-free (with optional inflation)."""
        for b in self.buildings:
            if (b.x_min - inflation <= x <= b.x_max + inflation and
                    b.y_min - inflation <= y <= b.y_max + inflation):
                if z <= b.height + inflation:
                    return False
        for o in self.obstacles:
            if np.sqrt((x - o.x) ** 2 + (y - o.y) ** 2) < o.radius + inflation:
                if z <= o.height + inflation:
                    return False
        for do in self.dynamic_obstacles:
            if np.sqrt((x - do.x) ** 2 + (y - do.y) ** 2) < do.radius + inflation:
                if z <= do.height + inflation:
                    return False
        return True

    # ------------------------------------------------------------------
    def get_occupancy_grid(self, resolution=PLANNING_GRID_RESOLUTION):
        """
        Build a 2D binary occupancy grid for path planning.
        Returns (grid, resolution)  where grid[r, c]==1 means occupied.
        Grid axes: row ↔ y, col ↔ x.
        """
        n = int(np.ceil(self.size / resolution))
        grid = np.zeros((n, n), dtype=np.int8)
        inflate = PLANNING_OBSTACLE_INFLATION

        for b in self.buildings:
            r_min = max(0, int((b.y_min - inflate) / resolution))
            r_max = min(n - 1, int((b.y_max + inflate) / resolution))
            c_min = max(0, int((b.x_min - inflate) / resolution))
            c_max = min(n - 1, int((b.x_max + inflate) / resolution))
            grid[r_min:r_max + 1, c_min:c_max + 1] = 1

        for o in list(self.obstacles) + list(self.dynamic_obstacles):
            cr = int(o.y / resolution)
            cc = int(o.x / resolution)
            rad_cells = int(np.ceil((o.radius + inflate) / resolution))
            for dr in range(-rad_cells, rad_cells + 1):
                for dc in range(-rad_cells, rad_cells + 1):
                    r, c = cr + dr, cc + dc
                    if 0 <= r < n and 0 <= c < n:
                        grid[r, c] = 1

        return grid, resolution

    # ------------------------------------------------------------------
    def world_to_grid(self, x, y, resolution=PLANNING_GRID_RESOLUTION):
        return int(y / resolution), int(x / resolution)

    def grid_to_world(self, row, col, resolution=PLANNING_GRID_RESOLUTION):
        return col * resolution + resolution / 2, row * resolution + resolution / 2

    # ------------------------------------------------------------------
    def add_dynamic_obstacle(self, x, y, radius, height):
        obs = Obstacle(x, y, radius, height)
        self.dynamic_obstacles.append(obs)
        return obs

    # ------------------------------------------------------------------
    def get_random_free_position(self, near=None, near_radius=60.0):
        """Return a random free ground position, optionally near a point."""
        for _ in range(500):
            if near is not None:
                x = near[0] + self.rng.uniform(-near_radius, near_radius)
                y = near[1] + self.rng.uniform(-near_radius, near_radius)
            else:
                x = self.rng.uniform(30, self.size - 30)
                y = self.rng.uniform(30, self.size - 30)
            x = np.clip(x, 5, self.size - 5)
            y = np.clip(y, 5, self.size - 5)
            if self.is_free(x, y, 0, inflation=2.0):
                return x, y
        return self.size / 2, self.size / 2
