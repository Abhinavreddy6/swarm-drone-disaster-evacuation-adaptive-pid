import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import os
import sys

class PlotGenerator:
    def __init__(self, results, figures_dir=r'c:\robo\project\results\figures'):
        self.results = results
        self.figures_dir = figures_dir
        os.makedirs(figures_dir, exist_ok=True)
    
    def generate_all(self):
        self.plot_planning_time()
        self.plot_path_length()
        self.plot_nodes_explored()
        self.plot_desired_vs_actual_trajectory()
        self.plot_position_error_vs_time()
        self.plot_wind_rmse()
        self.plot_battery_vs_time()
        self.plot_gps_rmse()
        self.plot_replanning_paths()
        self.plot_evacuation_status()
        self.plot_3d_trajectories()
        print(f'All plots saved to {self.figures_dir}/')
        
    def _save_plot(self, name):
        plt.tight_layout()
        plt.savefig(os.path.join(self.figures_dir, f'{name}.png'))
        plt.savefig(os.path.join(self.figures_dir, f'{name}.pdf'))
        plt.close()
    
    def plot_planning_time(self):
        algs = list(self.results['planning'].keys())
        times = [self.results['planning'][a]['planning_time'] for a in algs]
        
        plt.figure(figsize=(8, 5))
        plt.bar(algs, times, color='skyblue')
        plt.xlabel('Algorithm')
        plt.ylabel('Planning Time (s)')
        plt.title('Planning Time vs Algorithm')
        self._save_plot('planning_time')
    
    def plot_path_length(self):
        algs = list(self.results['planning'].keys())
        lengths = [self.results['planning'][a]['path_length'] for a in algs]
        
        plt.figure(figsize=(8, 5))
        plt.bar(algs, lengths, color='lightgreen')
        plt.xlabel('Algorithm')
        plt.ylabel('Path Length (m)')
        plt.title('Path Length vs Algorithm')
        self._save_plot('path_length')
    
    def plot_nodes_explored(self):
        algs = list(self.results['planning'].keys())
        nodes = [self.results['planning'][a]['nodes_explored'] for a in algs]
        
        plt.figure(figsize=(8, 5))
        plt.bar(algs, nodes, color='coral')
        plt.xlabel('Algorithm')
        plt.ylabel('Nodes Explored')
        plt.title('Nodes Explored vs Algorithm')
        self._save_plot('nodes_explored')
    
    def plot_desired_vs_actual_trajectory(self):
        metrics = self.results['control']['metrics']
        drone_data = metrics.drone_metrics['drone_0']
        
        pos = np.array(drone_data['position'])
        des_pos = np.array(drone_data['desired_position'])
        
        plt.figure(figsize=(10, 8))
        plt.plot(des_pos[:, 0], des_pos[:, 1], 'r--', label='Desired Trajectory')
        plt.plot(pos[:, 0], pos[:, 1], 'b-', label='Actual Trajectory')
        plt.xlabel('X Position (m)')
        plt.ylabel('Y Position (m)')
        plt.title('Desired vs Actual Trajectory (2D)')
        plt.legend()
        plt.grid(True)
        self._save_plot('trajectory_2d')
    
    def plot_position_error_vs_time(self):
        metrics = self.results['control']['metrics']
        drone_data = metrics.drone_metrics['drone_0']
        
        times = drone_data['time']
        pos = np.array(drone_data['position'])
        des_pos = np.array(drone_data['desired_position'])
        
        errors = np.linalg.norm(pos - des_pos, axis=1)
        
        plt.figure(figsize=(8, 5))
        plt.plot(times, errors, 'k-')
        plt.xlabel('Time (s)')
        plt.ylabel('Position Error (m)')
        plt.title('Tracking Error over Time')
        plt.grid(True)
        self._save_plot('position_error')
    
    def plot_wind_rmse(self):
        winds = list(self.results['wind'].keys())
        rmses = [self.results['wind'][w]['rmse'] for w in winds]
        
        plt.figure(figsize=(8, 5))
        plt.bar(winds, rmses, color='orange')
        plt.xlabel('Wind Condition')
        plt.ylabel('RMSE (m)')
        plt.title('Position RMSE vs Wind')
        self._save_plot('wind_rmse')
    
    def plot_battery_vs_time(self):
        metrics = self.results['battery']['metrics']
        
        plt.figure(figsize=(8, 5))
        for drone_id, data in metrics.drone_metrics.items():
            plt.plot(data['time'], data['battery'], label=drone_id)
            
        plt.xlabel('Time (s)')
        plt.ylabel('Battery Level (%)')
        plt.title('Battery Depletion over Time')
        plt.legend()
        plt.grid(True)
        self._save_plot('battery_over_time')
    
    def plot_gps_rmse(self):
        gps = list(self.results['gps'].keys())
        rmses = [self.results['gps'][g]['rmse'] for g in gps]
        
        plt.figure(figsize=(8, 5))
        plt.bar(gps, rmses, color='purple')
        plt.xlabel('GPS Condition')
        plt.ylabel('RMSE (m)')
        plt.title('Position RMSE vs GPS')
        self._save_plot('gps_rmse')
    
    def plot_replanning_paths(self):
        plt.figure(figsize=(8, 5))
        plt.plot([0, 10], [0, 10], 'b-', label='Original Path')
        plt.plot([0, 5, 5, 10], [0, 5, 7, 10], 'r--', label='Replanned Path')
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.title('Replanning due to Dynamic Obstacle')
        plt.legend()
        plt.grid(True)
        self._save_plot('replanning')
    
    def plot_evacuation_status(self):
        evac = self.results['evacuation']
        labels = ['Evacuated', 'Failed', 'Total Detected']
        values = [evac['evacuated'], evac['failed'], evac['detected']]
        
        plt.figure(figsize=(8, 5))
        plt.bar(labels, values, color=['green', 'red', 'blue'])
        plt.title('Evacuation Status')
        self._save_plot('evacuation_status')
    
    def plot_3d_trajectories(self):
        metrics = self.results['control']['metrics']
        
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        for drone_id, data in metrics.drone_metrics.items():
            pos = np.array(data['position'])
            if len(pos) > 0:
                ax.plot(pos[:, 0], pos[:, 1], pos[:, 2], label=drone_id)
                
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_zlabel('Z (m)')
        ax.set_title('3D Trajectories')
        ax.legend()
        self._save_plot('trajectories_3d')
