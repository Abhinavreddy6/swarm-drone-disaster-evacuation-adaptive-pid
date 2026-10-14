import numpy as np
import sys
sys.path.insert(0, r'c:\robo\project')
from config import *
from simulation.dynamics import DroneDynamics
from simulation.sensors import GPSSensor, IMUSensor, SensorFusion
from control.position_controller import PositionController
from control.attitude_controller import AttitudeController

class Drone:
    def __init__(self, drone_id, role, start_position):
        self.id = drone_id
        self.role = role  # 'scout', 'relay', 'guide', 'safezone'
        
        # Continuous state
        self.position = np.array(start_position, dtype=float)
        self.velocity = np.zeros(3)
        self.acceleration = np.zeros(3)
        self.yaw = 0.0
        self.pitch = 0.0
        self.roll = 0.0
        
        # Battery
        self.battery = BATTERY_CAPACITY
        self.battery_history = []
        
        # Systems
        self.dynamics = DroneDynamics()
        self.gps = GPSSensor()
        self.imu = IMUSensor()
        self.sensor_fusion = SensorFusion()
        self.position_controller = PositionController()
        self.attitude_controller = AttitudeController()
        
        # Navigation
        self.waypoints = []  # list of np.array([x, y, z])
        self.current_waypoint_idx = 0
        self.path_raw = []  # raw A* waypoints
        self.path_smoothed = []  # smoothed waypoints
        self.mission_complete = False
        
        # Communication
        self.comm_status = 'connected'
        self.comm_history = []
        
        # Telemetry recording
        self.position_history = []
        self.desired_position_history = []
        self.velocity_history = []
        self.error_history = []
        self.control_effort_history = []
        self.time_history = []
        
        # Role-specific state
        self.detected_civilians = []
        self.detected_disasters = []
        self.assigned_civilians = []
        self.returning_to_base = False
        
        # Distance tracking
        self.total_distance = 0.0
        self._prev_position = np.copy(self.position)
    
    def set_waypoints(self, waypoints):
        """Set navigation waypoints (list of [x,y,z] arrays)."""
        self.waypoints = [np.array(w) for w in waypoints]
        self.current_waypoint_idx = 0
        self.mission_complete = False
        self.position_controller.reset()
    
    def get_current_target(self):
        """Get current waypoint target. Returns None if mission complete."""
        if self.current_waypoint_idx >= len(self.waypoints):
            self.mission_complete = True
            return None
        return self.waypoints[self.current_waypoint_idx]
    
    def update(self, dt, current_time, wind=None, environment=None):
        """
        Main update loop for one timestep.
        
        1. Get sensor measurements (GPS, IMU) with noise
        2. Run sensor fusion (Kalman filter)
        3. Get current waypoint target
        4. Compute position error
        5. Run PID position controller
        6. Run attitude controller
        7. Apply wind disturbance
        8. Update dynamics (physics)
        9. Update battery
        10. Check waypoint reached
        11. Check collision avoidance
        12. Record telemetry
        """
        if wind is None:
            wind = np.zeros(3)
        
        # 1. Sensor measurements
        gps_measurement = self.gps.measure(self.position, current_time)
        imu_accel = self.imu.measure_accel(self.acceleration)
        
        # 2. Sensor fusion
        self.sensor_fusion.predict(imu_accel, dt)
        if gps_measurement is not None:
            self.sensor_fusion.update_gps(gps_measurement)
        
        estimated_pos = self.sensor_fusion.get_estimated_position()
        
        # 3. Current target
        target = self.get_current_target()
        if target is None:
            # Hover in place
            target = self.position.copy()
        
        # 4-5. Position control
        desired_accel, error_mag, control_effort = self.position_controller.compute(
            target, estimated_pos, dt
        )
        
        # 6. Attitude control
        desired_yaw, desired_pitch, desired_roll, _ = self.attitude_controller.compute(
            self.velocity, self.yaw, desired_accel, dt
        )
        
        # 7-8. Dynamics update with wind
        state = {
            'position': self.position.copy(),
            'velocity': self.velocity.copy(),
            'acceleration': self.acceleration.copy(),
            'yaw': self.yaw,
            'pitch': self.pitch,
            'roll': self.roll,
        }
        new_state = self.dynamics.update(state, desired_accel, wind, dt)
        
        self.position = new_state['position']
        self.velocity = new_state['velocity']
        self.acceleration = new_state['acceleration']
        self.yaw = new_state['yaw']
        self.pitch = new_state['pitch']
        self.roll = new_state['roll']

        # Building collision avoidance
        if environment is not None and hasattr(environment, 'buildings'):
            margin = 3.5
            for b in environment.buildings:
                if self.position[2] <= b.height + 2.0:
                    if (b.x_min - margin <= self.position[0] <= b.x_max + margin and
                        b.y_min - margin <= self.position[1] <= b.y_max + margin):
                        dx_min = abs(self.position[0] - (b.x_min - margin))
                        dx_max = abs(self.position[0] - (b.x_max + margin))
                        dy_min = abs(self.position[1] - (b.y_min - margin))
                        dy_max = abs(self.position[1] - (b.y_max + margin))
                        min_d = min(dx_min, dx_max, dy_min, dy_max)
                        if min_d == dx_min:
                            self.position[0] = b.x_min - margin
                            self.velocity[0] = 0.0
                        elif min_d == dx_max:
                            self.position[0] = b.x_max + margin
                            self.velocity[0] = 0.0
                        elif min_d == dy_min:
                            self.position[1] = b.y_min - margin
                            self.velocity[1] = 0.0
                        else:
                            self.position[1] = b.y_max + margin
                            self.velocity[1] = 0.0
        
        # 9. Battery
        dist = np.linalg.norm(self.position - self._prev_position)
        self.total_distance += dist
        speed = np.linalg.norm(self.velocity)
        self.battery -= dist * BATTERY_MOVEMENT_COST
        if speed < 0.5:
            self.battery -= dt * BATTERY_HOVER_COST
        self.battery -= control_effort * BATTERY_CONTROL_COST
        self.battery = max(0.0, self.battery)
        self.battery_history.append(self.battery)
        
        # 10. Waypoint check
        if self.current_waypoint_idx < len(self.waypoints):
            wp = self.waypoints[self.current_waypoint_idx]
            if np.linalg.norm(self.position - wp) < WAYPOINT_ACCEPTANCE_RADIUS:
                self.current_waypoint_idx += 1
                if self.current_waypoint_idx >= len(self.waypoints):
                    self.mission_complete = True
        
        # 11. Record telemetry
        self.position_history.append(self.position.copy())
        self.desired_position_history.append(target.copy())
        self.velocity_history.append(self.velocity.copy())
        self.error_history.append(error_mag)
        self.time_history.append(current_time)
        self._prev_position = self.position.copy()
        
        # Check low battery
        if self.battery < BATTERY_LOW_THRESHOLD and not self.returning_to_base:
            self.returning_to_base = True
    
    def detect_civilians(self, civilians, current_time):
        """Scout drone: detect civilians within DETECTION_RADIUS."""
        detected = []
        for c in civilians:
            if c.status.value == 'unknown':
                dist = np.linalg.norm(self.position[:2] - c.position[:2])
                if dist < DETECTION_RADIUS:
                    c.detect(current_time)
                    detected.append(c)
                    self.detected_civilians.append(c.id)
        return detected
    
    def detect_disasters(self, disaster_zones):
        """Scout drone: detect disaster zones within FIRE_DETECTION_RADIUS."""
        detected = []
        for dz in disaster_zones:
            center = np.array([dz.center_x, dz.center_y])
            dist = np.linalg.norm(self.position[:2] - center)
            if dist < FIRE_DETECTION_RADIUS:
                if dz not in self.detected_disasters:
                    self.detected_disasters.append(dz)
                    detected.append(dz)
        return detected
    
    def check_communication(self, other_drones, relay_drone=None):
        """Check if within communication range of other drones or relay."""
        for other in other_drones:
            if other.id == self.id:
                continue
            dist = np.linalg.norm(self.position - other.position)
            if dist > COMMUNICATION_RADIUS:
                self.comm_status = 'lost'
                return False
        self.comm_status = 'connected'
        return True

    def get_telemetry(self):
        """Return current state as dict."""
        return {
            'id': self.id,
            'role': self.role,
            'position': self.position.copy(),
            'velocity': self.velocity.copy(),
            'acceleration': self.acceleration.copy(),
            'yaw': self.yaw,
            'pitch': self.pitch,
            'roll': self.roll,
            'battery': self.battery,
            'speed': np.linalg.norm(self.velocity),
            'altitude': self.position[2],
            'current_waypoint': self.current_waypoint_idx,
            'total_waypoints': len(self.waypoints),
            'error': self.error_history[-1] if self.error_history else 0.0,
            'comm_status': self.comm_status,
            'mission_complete': self.mission_complete,
            'total_distance': self.total_distance,
        }
