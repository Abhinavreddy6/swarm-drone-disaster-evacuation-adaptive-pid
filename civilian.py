"""
Civilian entity in continuous 2D/3D space with state tracking.
"""

from enum import Enum
import numpy as np
import sys
sys.path.insert(0, r'c:\robo\project')
from config import *

class CivilianStatus(Enum):
    UNKNOWN = 'unknown'
    DETECTED = 'detected'
    ASSIGNED = 'assigned'
    EVACUATING = 'evacuating'
    SAFE = 'safe'
    FAILED = 'failed'

class Civilian:
    def __init__(self, cid, x, y):
        self.id = cid
        self.position = np.array([float(x), float(y), 0.0])
        self.velocity = np.zeros(3)
        self.status = CivilianStatus.UNKNOWN
        self.detected_time = None
        self.evacuation_start_time = None
        self.evacuation_end_time = None
        self.assigned_drone = None
        self.evacuation_path = []  # List of continuous [x, y, z] waypoints
        self.current_waypoint_idx = 0
        self.speed = CIVILIAN_SPEED

    def detect(self, current_time):
        if self.status == CivilianStatus.UNKNOWN:
            self.status = CivilianStatus.DETECTED
            self.detected_time = current_time

    def assign(self, drone_id, path):
        self.assigned_drone = drone_id
        self.evacuation_path = [np.array(p, dtype=float) for p in path]
        self.current_waypoint_idx = 0
        self.status = CivilianStatus.ASSIGNED

    def start_evacuation(self, current_time):
        self.status = CivilianStatus.EVACUATING
        self.evacuation_start_time = current_time

    def mark_safe(self, current_time):
        self.status = CivilianStatus.SAFE
        self.evacuation_end_time = current_time
        self.velocity = np.zeros(3)

    def is_at_safe_zone(self, safe_zone_center=SAFE_ZONE_CENTER, safe_zone_radius=SAFE_ZONE_RADIUS):
        dist = np.linalg.norm(self.position[:2] - safe_zone_center[:2])
        return dist <= safe_zone_radius

    def update(self, dt, guide_drone_pos=None):
        """Move civilian continuously along assigned path toward safe zone."""
        if self.status != CivilianStatus.EVACUATING:
            return

        if self.is_at_safe_zone():
            return

        if not self.evacuation_path or self.current_waypoint_idx >= len(self.evacuation_path):
            # If path ended but safe zone target remains, head straight to safe zone
            target = SAFE_ZONE_CENTER[:2]
        else:
            target = self.evacuation_path[self.current_waypoint_idx][:2]

        delta = target - self.position[:2]
        dist = np.linalg.norm(delta)

        if dist < WAYPOINT_ACCEPTANCE_RADIUS:
            if self.current_waypoint_idx < len(self.evacuation_path) - 1:
                self.current_waypoint_idx += 1
                target = self.evacuation_path[self.current_waypoint_idx][:2]
                delta = target - self.position[:2]
                dist = np.linalg.norm(delta)
            else:
                target = SAFE_ZONE_CENTER[:2]
                delta = target - self.position[:2]
                dist = np.linalg.norm(delta)

        if dist > 0.05:
            direction = delta / dist
            self.velocity = np.array([direction[0] * self.speed, direction[1] * self.speed, 0.0])
            self.position += self.velocity * dt
        else:
            self.velocity = np.zeros(3)
