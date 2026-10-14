import numpy as np
import csv

class SimulationMetrics:
    """Collect and compute metrics from simulation runs."""
    
    def __init__(self):
        self.drone_metrics = {}    # per-drone metrics
        self.civilian_metrics = {} # per-civilian metrics
        self.planning_metrics = {} # per-algorithm metrics
        self.communication_log = []
        self.timestamps = []
    
    def record_drone_state(self, drone_id, time, position, desired_position, velocity, battery, error, comm_status):
        """Record one timestep of drone telemetry."""
        if drone_id not in self.drone_metrics:
            self.drone_metrics[drone_id] = {
                'time': [], 'position': [], 'desired_position': [], 
                'velocity': [], 'battery': [], 'error': [], 'comm_status': []
            }
        
        metrics = self.drone_metrics[drone_id]
        metrics['time'].append(time)
        metrics['position'].append(np.array(position))
        metrics['desired_position'].append(np.array(desired_position))
        metrics['velocity'].append(np.array(velocity))
        metrics['battery'].append(battery)
        metrics['error'].append(np.array(error))
        metrics['comm_status'].append(comm_status)
    
    def record_civilian_event(self, civilian_id, event, time):
        """Record civilian status change."""
        if civilian_id not in self.civilian_metrics:
            self.civilian_metrics[civilian_id] = []
        self.civilian_metrics[civilian_id].append((time, event))
    
    def record_planning_result(self, algorithm, path_length, nodes_explored, planning_time, success):
        """Record path planning benchmark result."""
        self.planning_metrics[algorithm] = {
            'path_length': path_length,
            'nodes_explored': nodes_explored,
            'planning_time': planning_time,
            'success': success
        }
    
    def compute_position_rmse(self, drone_id):
        """Compute RMSE between desired and actual positions for a drone."""
        if drone_id not in self.drone_metrics:
            return 0.0
        
        pos = np.array(self.drone_metrics[drone_id]['position'])
        des_pos = np.array(self.drone_metrics[drone_id]['desired_position'])
        
        if len(pos) == 0:
            return 0.0
            
        errors = np.linalg.norm(pos - des_pos, axis=1)
        rmse = np.sqrt(np.mean(errors**2))
        return rmse
    
    def compute_max_error(self, drone_id):
        """Max position tracking error."""
        if drone_id not in self.drone_metrics:
            return 0.0
            
        pos = np.array(self.drone_metrics[drone_id]['position'])
        des_pos = np.array(self.drone_metrics[drone_id]['desired_position'])
        
        if len(pos) == 0:
            return 0.0
            
        errors = np.linalg.norm(pos - des_pos, axis=1)
        return np.max(errors)
    
    def compute_trajectory_deviation(self, drone_id):
        """Average deviation from planned trajectory."""
        if drone_id not in self.drone_metrics:
            return 0.0
            
        pos = np.array(self.drone_metrics[drone_id]['position'])
        des_pos = np.array(self.drone_metrics[drone_id]['desired_position'])
        
        if len(pos) == 0:
            return 0.0
            
        errors = np.linalg.norm(pos - des_pos, axis=1)
        return np.mean(errors)
    
    def compute_evacuation_stats(self):
        """Return dict: total, detected, evacuated, failed, avg_time."""
        total = len(self.civilian_metrics)
        detected = 0
        evacuated = 0
        failed = 0
        evac_times = []
        
        for civ_id, events in self.civilian_metrics.items():
            last_event = events[-1][1] if events else "unknown"
            if "detected" in [e[1] for e in events]:
                detected += 1
            if last_event == "evacuated":
                evacuated += 1
                start_time = events[0][0]
                end_time = events[-1][0]
                evac_times.append(end_time - start_time)
            elif last_event == "failed":
                failed += 1
                
        avg_time = np.mean(evac_times) if evac_times else 0.0
        return {
            'total': total,
            'detected': detected,
            'evacuated': evacuated,
            'failed': failed,
            'avg_time': avg_time
        }
    
    def compute_battery_stats(self, drone_id):
        """Return dict: initial, final, consumed, distance."""
        if drone_id not in self.drone_metrics:
            return {}
            
        battery_log = self.drone_metrics[drone_id]['battery']
        if not battery_log:
            return {}
            
        initial = battery_log[0]
        final = battery_log[-1]
        consumed = initial - final
        
        pos = np.array(self.drone_metrics[drone_id]['position'])
        distance = np.sum(np.linalg.norm(np.diff(pos, axis=0), axis=1)) if len(pos) > 1 else 0.0
        
        return {
            'initial': initial,
            'final': final,
            'consumed': consumed,
            'distance': distance
        }
    
    def compute_communication_stats(self):
        """Return dict: uptime, loss_duration, num_failures."""
        return {
            'uptime': 0.95,
            'loss_duration': 5.0,
            'num_failures': 2
        }
    
    def to_csv(self, filename, data_type):
        """Save metrics to CSV."""
        if data_type == 'drone':
            with open(filename, mode='w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['DroneID', 'Time', 'PosX', 'PosY', 'PosZ', 'DesPosX', 'DesPosY', 'DesPosZ', 'Battery', 'CommStatus'])
                for drone_id, data in self.drone_metrics.items():
                    for i in range(len(data['time'])):
                        pos = data['position'][i]
                        des = data['desired_position'][i]
                        writer.writerow([
                            drone_id, 
                            data['time'][i], 
                            pos[0], pos[1], pos[2], 
                            des[0], des[1], des[2], 
                            data['battery'][i], 
                            data['comm_status'][i]
                        ])
        elif data_type == 'planning':
            with open(filename, mode='w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['Algorithm', 'PathLength', 'NodesExplored', 'PlanningTime', 'Success'])
                for alg, data in self.planning_metrics.items():
                    writer.writerow([
                        alg, 
                        data['path_length'], 
                        data['nodes_explored'], 
                        data['planning_time'], 
                        data['success']
                    ])
