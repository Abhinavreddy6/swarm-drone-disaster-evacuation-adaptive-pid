"""
Adaptive PID Tuner Module
Evaluates trained Decision Tree ensembles or feature rules to predict dynamic PID gains.
"""

import os
import json
import numpy as np

class AdaptivePIDTuner:
    def __init__(self, weights_path=None):
        if weights_path is None:
            weights_path = os.path.join(os.path.dirname(__file__), '..', 'ml_tuning', 'ml_model_weights.json')
            
        self.model_data = None
        if os.path.exists(weights_path):
            with open(weights_path, 'r') as f:
                self.model_data = json.load(f)
                
        self.baseline_gains = {
            'kp_pos': 2.2, 'ki_pos': 0.05, 'kd_pos': 1.0,
            'kp_alt': 2.0, 'ki_alt': 0.10, 'kd_alt': 1.0,
            'kp_yaw': 1.5, 'ki_yaw': 0.02, 'kd_yaw': 0.4
        }
        self.gain_bounds = {
            'kp_pos': (1.2, 4.0), 'ki_pos': (0.01, 0.25), 'kd_pos': (0.5, 2.4),
            'kp_alt': (1.2, 3.8), 'ki_alt': (0.02, 0.30), 'kd_alt': (0.5, 2.2),
            'kp_yaw': (1.0, 2.8), 'ki_yaw': (0.005, 0.08), 'kd_yaw': (0.2, 0.9)
        }
        self.current_gains = dict(self.baseline_gains)

    def _eval_tree(self, node, features):
        if 'val' in node:
            return float(node['val'])
        feat_idx = node['feat']
        threshold = node['th']
        if features[feat_idx] <= threshold:
            return self._eval_tree(node['left'], features)
        else:
            return self._eval_tree(node['right'], features)

    def predict(self, feature_vec):
        """
        Predict adaptive PID gains from 14-element feature vector:
        [pos_err_x, pos_err_y, pos_err_z, pos_err_3d, vel_err_mag,
         vel_mag, accel_mag, wind_speed, obstacle_dist, repulsion_mag,
         drone_role, battery_soc, error_rate, heading_curvature]
        """
        if not self.model_data or 'trees' not in self.model_data:
            return dict(self.baseline_gains)

        preds = {}
        for target_name, tree_root in self.model_data['trees'].items():
            val = self._eval_tree(tree_root, feature_vec)
            # Bound and clamp
            b_min, b_max = self.gain_bounds[target_name]
            val = float(np.clip(val, b_min, b_max))
            preds[target_name] = val

        return preds

    def step(self, feature_vec, rate_limit=0.15):
        """Predicts and applies rate-limited exponential smoothing to gains."""
        target_gains = self.predict(feature_vec)
        for k in self.current_gains:
            self.current_gains[k] = (1.0 - rate_limit) * self.current_gains[k] + rate_limit * target_gains[k]
        return dict(self.current_gains)

    def reset(self):
        self.current_gains = dict(self.baseline_gains)
