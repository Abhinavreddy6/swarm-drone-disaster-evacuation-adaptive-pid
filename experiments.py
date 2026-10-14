"""
Comprehensive experiment runner executing all 7 required benchmarks using actual simulation runs.
Saves genuine data to CSV files.
"""

import numpy as np
import sys
import os
import time
import csv

sys.path.insert(0, r'c:\robo\project')
from config import *
from simulation.environment import Environment
from simulation.simulator import Simulator
from evaluation.metrics import SimulationMetrics

import planning.astar as astar
import planning.dijkstra as dijkstra
import planning.bfs as bfs
import planning.greedy as greedy
import planning.dstar_lite as dstar_lite

class ExperimentRunner:
    def __init__(self):
        self.results = {}
        os.makedirs(CSV_DIR, exist_ok=True)

    def run_all(self):
        print('=== Experiment 1: Path Planning Comparison (A*, Dijkstra, BFS, Greedy, D* Lite) ===')
        self.run_planning_comparison()
        
        print('=== Experiment 2: PID Tracking Performance ===')
        self.run_pid_tracking()
        
        print('=== Experiment 3: Wind Disturbance Comparison (None, Moderate, Strong) ===')
        self.run_wind_comparison()
        
        print('=== Experiment 4: Sensor Fusion Under GPS Degradation ===')
        self.run_gps_comparison()
        
        print('=== Experiment 5: Dynamic Obstacle Replanning ===')
        self.run_dynamic_obstacle()
        
        print('=== Experiment 6: Battery Depletion Analysis ===')
        self.run_battery_analysis()
        
        print('=== Experiment 7: Civilian Evacuation Performance ===')
        self.run_evacuation_analysis()

        self.save_all_csv()
        return self.results

    def run_planning_comparison(self):
        env = Environment(seed=RANDOM_SEED)
        grid, res = env.get_occupancy_grid(PLANNING_GRID_RESOLUTION)
        
        # Pick realistic start and goal on grid
        start_cell = env.world_to_grid(50.0, 50.0, res)
        goal_cell = env.world_to_grid(420.0, 420.0, res)
        
        planners = {
            'A*': astar.plan,
            'Dijkstra': dijkstra.plan,
            'BFS': bfs.plan,
            'Greedy': greedy.plan,
            'D* Lite': dstar_lite.plan
        }
        
        self.results['planning'] = {}
        for name, plan_func in planners.items():
            print(f"  Testing planner: {name}...")
            res_dict = plan_func(grid, start_cell, goal_cell, res, (0.0, 0.0))
            self.results['planning'][name] = {
                'planning_time': max(res_dict['planning_time'], 0.001),
                'path_length': res_dict['path_length'],
                'nodes_explored': res_dict['nodes_explored'],
                'success': res_dict['success']
            }

    def run_pid_tracking(self):
        print("  Running continuous PID tracking experiment...")
        sim = Simulator(wind_mode='none', gps_mode='normal', seed=RANDOM_SEED)
        sim.run(max_steps=500)
        metrics = sim.get_metrics()
        
        drone_key = 'drone_0'
        rmse = metrics.compute_position_rmse(drone_key)
        max_err = metrics.compute_max_error(drone_key)
        
        self.results['control'] = {
            'rmse': rmse,
            'max_error': max_err,
            'metrics': metrics
        }

    def run_wind_comparison(self):
        self.results['wind'] = {}
        for wind_level in ['none', 'moderate', 'strong']:
            print(f"  Running wind experiment: {wind_level}...")
            sim = Simulator(wind_mode=wind_level, gps_mode='normal', seed=RANDOM_SEED)
            sim.run(max_steps=400)
            m = sim.get_metrics()
            drone_key = 'drone_0'
            rmse = m.compute_position_rmse(drone_key)
            max_e = m.compute_max_error(drone_key)
            self.results['wind'][wind_level] = {
                'rmse': rmse,
                'max_error': max_e,
                'metrics': m
            }

    def run_gps_comparison(self):
        self.results['gps'] = {}
        for gps_cond in ['normal', 'noisy', 'degraded']:
            print(f"  Running GPS experiment: {gps_cond}...")
            sim = Simulator(wind_mode='none', gps_mode=gps_cond, seed=RANDOM_SEED)
            sim.run(max_steps=400)
            m = sim.get_metrics()
            rmse = m.compute_position_rmse('drone_0')
            self.results['gps'][gps_cond] = {
                'rmse': rmse,
                'metrics': m
            }

    def run_dynamic_obstacle(self):
        print("  Running dynamic obstacle replanning experiment...")
        sim = Simulator(wind_mode='none', gps_mode='normal', dynamic_obstacles=True, seed=RANDOM_SEED)
        sim.run(max_steps=800)
        m = sim.get_metrics()
        self.results['dynamic'] = {
            'replanning_count': sim.replanning_count,
            'metrics': m
        }

    def run_battery_analysis(self):
        print("  Running battery analysis...")
        sim = Simulator(wind_mode='moderate', gps_mode='normal', seed=RANDOM_SEED)
        sim.run(max_steps=800)
        m = sim.get_metrics()
        self.results['battery'] = {
            'metrics': m
        }

    def run_evacuation_analysis(self):
        print("  Running civilian evacuation analysis...")
        sim = Simulator(wind_mode='moderate', gps_mode='normal', seed=RANDOM_SEED)
        sim.run(max_steps=1200)
        m = sim.get_metrics()
        evac_stats = m.compute_evacuation_stats()
        self.results['evacuation'] = {
            'total': evac_stats['total'],
            'detected': evac_stats['detected'],
            'evacuated': evac_stats['evacuated'],
            'failed': evac_stats['failed'],
            'avg_time': evac_stats['avg_time'],
            'metrics': m
        }

    def save_all_csv(self):
        # 1. algorithm_results.csv
        with open(os.path.join(CSV_DIR, 'algorithm_results.csv'), 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Algorithm', 'PlanningTime_s', 'PathLength_m', 'NodesExplored', 'Success'])
            for alg, val in self.results['planning'].items():
                writer.writerow([alg, f"{val['planning_time']:.4f}", f"{val['path_length']:.2f}", val['nodes_explored'], val['success']])

        # 2. control_results.csv
        with open(os.path.join(CSV_DIR, 'control_results.csv'), 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['DroneID', 'PositionRMSE_m', 'MaxError_m'])
            writer.writerow(['drone_0', f"{self.results['control']['rmse']:.3f}", f"{self.results['control']['max_error']:.3f}"])

        # 3. wind_results.csv
        with open(os.path.join(CSV_DIR, 'wind_results.csv'), 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['WindCondition', 'PositionRMSE_m', 'MaxError_m'])
            for w, val in self.results['wind'].items():
                writer.writerow([w, f"{val['rmse']:.3f}", f"{val['max_error']:.3f}"])

        # 4. gps_results.csv
        with open(os.path.join(CSV_DIR, 'gps_results.csv'), 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['GPSCondition', 'PositionRMSE_m'])
            for g, val in self.results['gps'].items():
                writer.writerow([g, f"{val['rmse']:.3f}"])

        # 5. battery_results.csv
        with open(os.path.join(CSV_DIR, 'battery_results.csv'), 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['DroneID', 'InitialBattery', 'FinalBattery', 'ConsumedBattery'])
            bm = self.results['battery']['metrics']
            for d_id in bm.drone_metrics.keys():
                b_stats = bm.compute_battery_stats(d_id)
                writer.writerow([d_id, b_stats['initial'], f"{b_stats['final']:.2f}", f"{b_stats['consumed']:.2f}"])

        # 6. evacuation_results.csv
        with open(os.path.join(CSV_DIR, 'evacuation_results.csv'), 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['TotalCivilians', 'Detected', 'Evacuated', 'Failed', 'AvgEvacuationTime_s'])
            ev = self.results['evacuation']
            writer.writerow([ev['total'], ev['detected'], ev['evacuated'], ev['failed'], f"{ev['avg_time']:.2f}"])

        # 7. replanning_results.csv
        with open(os.path.join(CSV_DIR, 'replanning_results.csv'), 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['DynamicObstaclesPresent', 'ReplanningCount'])
            writer.writerow([True, self.results['dynamic']['replanning_count']])

        print(f"All 7 CSV result files saved successfully to {CSV_DIR}")
