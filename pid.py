import sys
sys.path.insert(0, r'c:\robo\project')
from config import PID_INTEGRAL_LIMIT
import numpy as np

class PIDController:
    def __init__(self, kp, ki, kd, integral_limit=PID_INTEGRAL_LIMIT, output_limit=None):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.integral_limit = integral_limit
        self.output_limit = output_limit
        self.integral = 0.0
        self.prev_error = 0.0
        self.prev_time = None
    
    def compute(self, error, dt):
        """
        PID control law:
        u(t) = Kp * e(t) + Ki * integral(e) + Kd * de/dt
        
        With anti-windup on integral term.
        Returns control output u.
        """
        if dt <= 0:
            return 0.0
            
        self.integral += error * dt
        
        # Anti-windup
        if self.integral_limit is not None:
            self.integral = np.clip(self.integral, -self.integral_limit, self.integral_limit)
            
        derivative = (error - self.prev_error) / dt
        
        u = self.kp * error + self.ki * self.integral + self.kd * derivative
        
        if self.output_limit is not None:
            u = np.clip(u, -self.output_limit, self.output_limit)
            
        self.prev_error = error
        return u
    
    def reset(self):
        """Reset integrator and derivative state."""
        self.integral = 0.0
        self.prev_error = 0.0
