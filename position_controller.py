import sys
sys.path.insert(0, r'c:\robo\project')
from config import (
    PID_POS_KP, PID_POS_KI, PID_POS_KD,
    PID_ALT_KP, PID_ALT_KI, PID_ALT_KD,
    MAX_ACCELERATION
)
import numpy as np
from control.pid import PIDController

class PositionController:
    def __init__(self):
        self.pid_x = PIDController(PID_POS_KP, PID_POS_KI, PID_POS_KD)
        self.pid_y = PIDController(PID_POS_KP, PID_POS_KI, PID_POS_KD)
        self.pid_z = PIDController(PID_ALT_KP, PID_ALT_KI, PID_ALT_KD)
        self.control_effort_history = []
    
    def compute(self, desired_position, estimated_position, dt):
        """
        Compute desired acceleration vector to move toward target.
        
        Args:
            desired_position: np.array([x, y, z])
            estimated_position: np.array([x, y, z]) from sensor fusion
            dt: timestep
        
        Returns:
            desired_acceleration: np.array([ax, ay, az])
            error_magnitude: float
            control_effort: float (sum of absolute control outputs)
        """
        error = desired_position - estimated_position
        ax = self.pid_x.compute(error[0], dt)
        ay = self.pid_y.compute(error[1], dt)
        az = self.pid_z.compute(error[2], dt)
        
        desired_accel = np.array([ax, ay, az])
        # Clamp to MAX_ACCELERATION
        mag = np.linalg.norm(desired_accel)
        if mag > MAX_ACCELERATION:
            desired_accel = desired_accel / mag * MAX_ACCELERATION
        
        control_effort = abs(ax) + abs(ay) + abs(az)
        self.control_effort_history.append(control_effort)
        
        return desired_accel, np.linalg.norm(error), control_effort
    
    def reset(self):
        self.pid_x.reset()
        self.pid_y.reset()
        self.pid_z.reset()
        self.control_effort_history = []
