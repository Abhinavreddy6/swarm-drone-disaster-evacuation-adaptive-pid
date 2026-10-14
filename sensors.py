"""
Sensor Simulation and Kalman Filter Sensor Fusion.
Simulates noisy GPS, IMU, and fuses them for continuous state estimation.
"""

import numpy as np
import sys
sys.path.insert(0, r'c:\robo\project')
from config import *

class GPSSensor:
    def __init__(self, noise_std=GPS_NOISE_STD, update_rate=GPS_UPDATE_RATE):
        self.noise_std = noise_std
        self.update_rate = update_rate
        self.dt_update = 1.0 / update_rate
        self.last_update = -1.0
        self.degraded = False

    def set_degraded(self, degraded, noise_std=GPS_NOISE_STD_DEGRADED):
        self.degraded = degraded
        self.noise_std = noise_std if degraded else GPS_NOISE_STD

    def measure(self, true_position, current_time):
        """Returns noisy position measurement if update interval reached."""
        if (current_time - self.last_update) >= self.dt_update or self.last_update < 0:
            self.last_update = current_time
            noise = np.random.normal(0.0, self.noise_std, size=3)
            return true_position + noise
        return None

class IMUSensor:
    def __init__(self, accel_noise=IMU_ACCEL_NOISE_STD, gyro_noise=IMU_GYRO_NOISE_STD):
        self.accel_noise = accel_noise
        self.gyro_noise = gyro_noise
        self.bias = np.array([IMU_ACCEL_BIAS, -IMU_ACCEL_BIAS, 0.0])

    def measure_accel(self, true_accel):
        noise = np.random.normal(0.0, self.accel_noise, size=3)
        return true_accel + self.bias + noise

    def measure_gyro(self, true_angular_vel):
        noise = np.random.normal(0.0, self.gyro_noise, size=3)
        return true_angular_vel + noise

class SensorFusion:
    """
    Linear Kalman Filter fusing IMU acceleration and GPS position.
    State vector: [x, y, z, vx, vy, vz]^T
    """
    def __init__(self, init_pos=None):
        self.state = np.zeros(6)
        if init_pos is not None:
            self.state[:3] = init_pos
        self.P = np.eye(6) * 1.0
        self.Q = np.eye(6) * KALMAN_PROCESS_NOISE
        self.R_gps = np.eye(3) * (KALMAN_GPS_MEASUREMENT_NOISE ** 2)

    def predict(self, accel_meas, dt):
        """Prediction step using kinematic motion model and IMU acceleration."""
        # F: State transition matrix
        F = np.eye(6)
        F[0, 3] = dt
        F[1, 4] = dt
        F[2, 5] = dt

        # B: Control input matrix for acceleration
        B = np.zeros((6, 3))
        B[0:3, :] = 0.5 * (dt ** 2) * np.eye(3)
        B[3:6, :] = dt * np.eye(3)

        self.state = F @ self.state + B @ accel_meas
        self.P = F @ self.P @ F.T + self.Q

    def update_gps(self, gps_meas):
        """Correction step using GPS position fix."""
        H = np.zeros((3, 6))
        H[0:3, 0:3] = np.eye(3)

        # Innovation
        y = gps_meas - (H @ self.state)
        S = H @ self.P @ H.T + self.R_gps
        K = self.P @ H.T @ np.linalg.inv(S)

        self.state = self.state + K @ y
        I = np.eye(6)
        self.P = (I - K @ H) @ self.P

    def get_estimated_position(self):
        return self.state[:3].copy()

    def get_estimated_velocity(self):
        return self.state[3:6].copy()

    def set_gps_noise(self, noise_std):
        self.R_gps = np.eye(3) * (noise_std ** 2)
