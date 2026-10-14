import sys
sys.path.insert(0, r'c:\robo\project')
from config import PID_YAW_KP, PID_YAW_KI, PID_YAW_KD
import numpy as np
from control.pid import PIDController

class AttitudeController:
    def __init__(self):
        self.pid_yaw = PIDController(PID_YAW_KP, PID_YAW_KI, PID_YAW_KD)
        self.desired_yaw = 0.0
        self.desired_pitch = 0.0
        self.desired_roll = 0.0
    
    def compute(self, current_velocity, current_yaw, desired_acceleration, dt):
        """
        Compute desired attitude from velocity and acceleration.
        
        - Yaw: orient toward velocity direction
        - Pitch: proportional to forward acceleration
        - Roll: proportional to lateral acceleration
        
        Returns:
            desired_yaw, desired_pitch, desired_roll (radians)
            yaw_correction: float (from PID)
        """
        vel_norm = np.linalg.norm(current_velocity[:2])
        if vel_norm > 0.1:
            self.desired_yaw = np.arctan2(current_velocity[1], current_velocity[0])
            
        # Wrap yaw error to [-pi, pi]
        yaw_error = self.desired_yaw - current_yaw
        yaw_error = (yaw_error + np.pi) % (2 * np.pi) - np.pi
        
        yaw_correction = self.pid_yaw.compute(yaw_error, dt)
        
        # Simple mapping from acceleration to roll/pitch assuming small angles
        # Rotate acceleration into body frame based on current yaw
        cos_yaw = np.cos(current_yaw)
        sin_yaw = np.sin(current_yaw)
        
        ax_body = desired_acceleration[0] * cos_yaw + desired_acceleration[1] * sin_yaw
        ay_body = -desired_acceleration[0] * sin_yaw + desired_acceleration[1] * cos_yaw
        
        g = 9.81
        self.desired_pitch = np.arctan2(ax_body, g)
        self.desired_roll = np.arctan2(-ay_body, g)
        
        return self.desired_yaw, self.desired_pitch, self.desired_roll, yaw_correction
    
    def compute_desired_heading(self, current_pos, target_pos):
        """Calculate desired yaw angle from current position to target."""
        delta = target_pos - current_pos
        return np.arctan2(delta[1], delta[0])
