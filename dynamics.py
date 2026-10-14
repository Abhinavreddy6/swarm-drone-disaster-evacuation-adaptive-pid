"""
Continuous Quadrotor Dynamics Simulation.
Models translational motion, drag, wind effects, heading, and attitude angles.
"""

import numpy as np
import sys
sys.path.insert(0, r'c:\robo\project')
from config import *

class DroneDynamics:
    """Simplified quadrotor dynamics in continuous 3D space."""
    
    def __init__(self, mass=DRONE_MASS, drag=DRAG_COEFFICIENT):
        self.mass = mass
        self.drag = drag
    
    def update(self, state, control_accel, wind, dt):
        """
        state: dict with 'position', 'velocity', 'acceleration', 'yaw', 'pitch', 'roll'
        control_accel: commanded 3D acceleration from controller (ax, ay, az)
        wind: 3D wind velocity vector
        dt: simulation timestep
        """
        pos = state['position'].copy()
        vel = state['velocity'].copy()
        
        # Relative velocity with respect to wind
        rel_vel = vel - wind
        rel_speed = np.linalg.norm(rel_vel)
        
        # Quadratic/linear aerodynamic drag force opposing relative motion
        drag_force = -self.drag * rel_vel * rel_speed
        drag_accel = drag_force / self.mass
        
        # Net acceleration: commanded acceleration + external aerodynamic forces
        net_accel = control_accel + drag_accel
        
        # Limit acceleration
        accel_mag = np.linalg.norm(net_accel)
        if accel_mag > MAX_ACCELERATION:
            net_accel = (net_accel / accel_mag) * MAX_ACCELERATION
            
        # Velocity update
        new_vel = vel + net_accel * dt
        
        # Speed limit
        speed = np.linalg.norm(new_vel)
        if speed > MAX_DRONE_SPEED:
            new_vel = (new_vel / speed) * MAX_DRONE_SPEED
            
        # Continuous position update
        new_pos = pos + new_vel * dt
        
        # Altitude boundaries
        new_pos[2] = np.clip(new_pos[2], MIN_ALTITUDE, MAX_ALTITUDE)
        
        # Update orientation/attitude
        yaw = self.compute_heading(new_vel, state.get('yaw', 0.0))
        pitch, roll = self.compute_attitude(net_accel, yaw)
        
        return {
            'position': new_pos,
            'velocity': new_vel,
            'acceleration': net_accel,
            'yaw': yaw,
            'pitch': pitch,
            'roll': roll
        }
    
    def compute_heading(self, velocity, current_yaw):
        """Compute heading (yaw) aligning with horizontal velocity."""
        horiz_speed = np.linalg.norm(velocity[:2])
        if horiz_speed > 0.15:
            target_yaw = np.arctan2(velocity[1], velocity[0])
            # Smooth transition
            diff = (target_yaw - current_yaw + np.pi) % (2 * np.pi) - np.pi
            return current_yaw + 0.2 * diff
        return current_yaw
    
    def compute_attitude(self, acceleration, yaw):
        """Compute bank pitch and roll angles given acceleration vector."""
        # Project acceleration to body frame
        cos_y = np.cos(yaw)
        sin_y = np.sin(yaw)
        ax_body = acceleration[0] * cos_y + acceleration[1] * sin_y
        ay_body = -acceleration[0] * sin_y + acceleration[1] * cos_y
        
        g = 9.81
        pitch = np.arctan2(ax_body, g)
        roll = np.arctan2(-ay_body, g)
        return float(np.clip(pitch, -0.4, 0.4)), float(np.clip(roll, -0.4, 0.4))
