"""
Complete continuous-space multi-drone disaster response simulator.
Coordinates 4 autonomous drones, civilians, sensor fusion, PID control, and dynamic replanning.
"""

import numpy as np
import sys
sys.path.insert(0, r'c:\robo\project')
from config import *
from simulation.environment import Environment
from simulation.drone import Drone
from simulation.civilian import Civilian, CivilianStatus
from planning.astar import plan as astar_plan
from planning.utils import smooth_path
from planning.dstar_lite import DStarLite
from evaluation.metrics import SimulationMetrics

class Simulator:
    def __init__(self, wind_mode='none', gps_mode='normal', dynamic_obstacles=False, seed=RANDOM_SEED):
        np.random.seed(seed)
        self.wind_mode = wind_mode
        self.gps_mode = gps_mode
        self.dynamic_obstacles = dynamic_obstacles
        self.seed = seed
        
        # 1. Continuous 3D Environment
        self.env = Environment(seed=seed)
        
        # 2. Four Autonomous Drones in continuous space
        self.drones = {
            ROLE_SCOUT: Drone(0, ROLE_SCOUT, DRONE_START_POSITIONS[ROLE_SCOUT]),
            ROLE_RELAY: Drone(1, ROLE_RELAY, DRONE_START_POSITIONS[ROLE_RELAY]),
            ROLE_GUIDE: Drone(2, ROLE_GUIDE, DRONE_START_POSITIONS[ROLE_GUIDE]),
            ROLE_SAFEZONE: Drone(3, ROLE_SAFEZONE, DRONE_START_POSITIONS[ROLE_SAFEZONE]),
        }
        
        # 3. Environment Navigation Grid
        self.grid, self.grid_resolution = self.env.get_occupancy_grid(PLANNING_GRID_RESOLUTION)
        
        # 4. Civilians
        self.civilians = self._create_civilians()
        
        # 5. Metrics Collector
        self.metrics = SimulationMetrics()
        
        # Wind configuration
        self.base_wind = self._get_wind_vector()
        
        # Simulation parameters
        self.time = 0.0
        self.dt = SIMULATION_DT
        self.running = True
        
        # Phase tracking
        self.phase = 'scout'
        self.scout_waypoints_set = False
        self.evacuation_paths = {}
        self.replanning_count = 0
        self.dynamic_obstacle_added = False
        
        self._setup_gps()

    def _create_civilians(self):
        civilians = []
        for i in range(NUMBER_OF_CIVILIANS):
            # Place civilians near disaster zones in continuous coordinates
            near_dz = self.env.disaster_zones[i % len(self.env.disaster_zones)] if self.env.disaster_zones else None
            near_pos = (near_dz.center_x, near_dz.center_y) if near_dz else None
            cx, cy = self.env.get_random_free_position(near=near_pos, near_radius=50.0)
            civilians.append(Civilian(i, cx, cy))
        return civilians

    def _get_wind_vector(self):
        if self.wind_mode == 'moderate':
            return WIND_MODERATE.copy()
        elif self.wind_mode == 'strong':
            return WIND_STRONG.copy()
        return WIND_NONE.copy()

    def _get_current_wind(self):
        turbulence = np.random.randn(3) * WIND_TURBULENCE_STD
        return self.base_wind + turbulence

    def _setup_gps(self):
        for drone in self.drones.values():
            if self.gps_mode == 'noisy':
                drone.gps.noise_std = GPS_NOISE_STD * 2.0
                drone.sensor_fusion.set_gps_noise(GPS_NOISE_STD * 2.0)
            elif self.gps_mode == 'degraded':
                drone.gps.noise_std = GPS_NOISE_STD
                drone.sensor_fusion.set_gps_noise(GPS_NOISE_STD)

    def _run_scout_phase(self):
        """Build lawnmower / continuous scanning waypoints for scout drone."""
        scout = self.drones[ROLE_SCOUT]
        waypoints = []
        x_coords = np.arange(40.0, ENVIRONMENT_SIZE - 40.0, 60.0)
        forward = True
        for x in x_coords:
            if forward:
                waypoints.append(np.array([x, 40.0, SCOUT_SCAN_ALTITUDE]))
                waypoints.append(np.array([x, ENVIRONMENT_SIZE - 40.0, SCOUT_SCAN_ALTITUDE]))
            else:
                waypoints.append(np.array([x, ENVIRONMENT_SIZE - 40.0, SCOUT_SCAN_ALTITUDE]))
                waypoints.append(np.array([x, 40.0, SCOUT_SCAN_ALTITUDE]))
            forward = not forward
        smoothed_wps = smooth_path(waypoints, num_samples=35, default_altitude=SCOUT_SCAN_ALTITUDE)
        scout.set_waypoints(smoothed_wps)

    def _plan_evacuation(self, civilian):
        """Plan continuous smooth trajectory from civilian to safe zone using A* and spline smoothing."""
        cr, cc = self.env.world_to_grid(civilian.position[0], civilian.position[1], self.grid_resolution)
        zr, zc = self.env.world_to_grid(SAFE_ZONE_CENTER[0], SAFE_ZONE_CENTER[1], self.grid_resolution)
        
        res = astar_plan(self.grid, (cr, cc), (zr, zc), self.grid_resolution, (0.0, 0.0))
        if res['success'] and len(res['path_world']) > 1:
            smoothed = smooth_path(res['path_world'], default_altitude=GUIDE_APPROACH_ALTITUDE)
            self.evacuation_paths[civilian.id] = smoothed
            civilian.assign(ROLE_GUIDE, smoothed)
            return smoothed
        else:
            # Fallback direct continuous path
            direct = [civilian.position.copy(), SAFE_ZONE_CENTER.copy()]
            direct[0][2] = GUIDE_APPROACH_ALTITUDE
            direct[1][2] = GUIDE_APPROACH_ALTITUDE
            self.evacuation_paths[civilian.id] = direct
            civilian.assign(ROLE_GUIDE, direct)
            return direct

    def _manage_relay(self):
        """Position Relay drone near the centroid of all other active drones."""
        active_pos = [d.position for r, d in self.drones.items() if r != ROLE_RELAY]
        if active_pos:
            centroid = np.mean(active_pos, axis=0)
            target = np.array([centroid[0], centroid[1], RELAY_OPTIMAL_ALTITUDE])
            self.drones[ROLE_RELAY].set_waypoints([target])

    def _manage_safezone(self):
        """Safe-Zone drone monitors and hovers around safe extraction perimeter."""
        sz_pos = np.array([SAFE_ZONE_CENTER[0], SAFE_ZONE_CENTER[1], SAFEZONE_PATROL_ALTITUDE])
        self.drones[ROLE_SAFEZONE].set_waypoints([sz_pos])

    def _add_dynamic_obstacle(self):
        """Spawn dynamic obstacle and trigger replanning."""
        cx = (BASE_STATION_POS[0] + SAFE_ZONE_CENTER[0]) / 2.0
        cy = (BASE_STATION_POS[1] + SAFE_ZONE_CENTER[1]) / 2.0
        self.env.add_dynamic_obstacle(cx, cy, DYNAMIC_OBSTACLE_RADIUS, 35.0)
        self.grid, _ = self.env.get_occupancy_grid(self.grid_resolution)
        self.dynamic_obstacle_added = True
        
        # Trigger dynamic replanning for evacuating civilians
        for civ in self.civilians:
            if civ.status == CivilianStatus.EVACUATING:
                self.replanning_count += 1
                new_path = self._plan_evacuation(civ)
                civ.start_evacuation(self.time)
                self.drones[ROLE_GUIDE].set_waypoints(new_path)

    def run(self, max_steps=1500):
        """Execute continuous simulation loop."""
        step = 0
        while self.time < SIMULATION_MAX_TIME and self.running and step < max_steps:
            step += 1
            
            # 1. GPS Degradation dynamics
            if self.gps_mode == 'degraded':
                in_degraded_window = (GPS_DEGRADATION_START <= self.time <= (GPS_DEGRADATION_START + GPS_DEGRADATION_DURATION))
                for d in self.drones.values():
                    d.gps.set_degraded(in_degraded_window)
                    if in_degraded_window:
                        d.sensor_fusion.set_gps_noise(GPS_NOISE_STD_DEGRADED)
                    else:
                        d.sensor_fusion.set_gps_noise(GPS_NOISE_STD)

            # 2. Mission Phases
            if self.phase == 'scout':
                if not self.scout_waypoints_set:
                    self._run_scout_phase()
                    self.scout_waypoints_set = True
                    self._manage_relay()
                    self._manage_safezone()

                # Scout drone detects civilians and disasters within radius
                scout = self.drones[ROLE_SCOUT]
                scout.detect_civilians(self.civilians, self.time)
                scout.detect_disasters(self.env.disaster_zones)
                
                # Check how many detected or transition once initial search sweep completes
                detected_cnt = sum(1 for c in self.civilians if c.status != CivilianStatus.UNKNOWN)
                if detected_cnt >= 3 or self.time >= 15.0 or scout.mission_complete:
                    self.phase = 'plan'

            elif self.phase == 'plan':
                # Plan evacuation routes for detected civilians
                for civ in self.civilians:
                    if civ.status == CivilianStatus.UNKNOWN:
                        civ.detect(self.time)
                    if civ.status in (CivilianStatus.DETECTED, CivilianStatus.ASSIGNED):
                        path = self._plan_evacuation(civ)
                        civ.start_evacuation(self.time)
                
                # Assign Guide drone to lead first evacuating civilian
                evac_civs = [c for c in self.civilians if c.status == CivilianStatus.EVACUATING]
                if evac_civs:
                    self.drones[ROLE_GUIDE].set_waypoints(self.evacuation_paths[evac_civs[0].id])
                self.phase = 'evacuate'

            elif self.phase == 'evacuate':
                self._manage_relay()
                self._manage_safezone()
                
                guide = self.drones[ROLE_GUIDE]
                
                # Update civilians continuous movement
                all_safe = True
                for civ in self.civilians:
                    if civ.status != CivilianStatus.SAFE:
                        all_safe = False
                        civ.update(self.dt, guide_drone_pos=guide.position)
                        if civ.is_at_safe_zone():
                            civ.mark_safe(self.time)
                            self.metrics.record_civilian_event(f"civ_{civ.id}", "evacuated", self.time)
                
                if all_safe or self.time > 300.0:
                    self.phase = 'complete'
                    self.running = False

            # 3. Dynamic Obstacle Check
            if self.dynamic_obstacles and not self.dynamic_obstacle_added and self.time >= 20.0:
                self._add_dynamic_obstacle()

            # 4. Update Drones (PID, dynamics, sensor fusion, wind)
            current_wind = self._get_current_wind()
            drone_list = list(self.drones.values())
            for d in drone_list:
                d.update(self.dt, self.time, wind=current_wind, environment=self.env)
                d.check_communication(drone_list, relay_drone=self.drones[ROLE_RELAY])
                
                # Record metrics
                target = d.get_current_target()
                if target is None:
                    target = d.position
                error_vec = d.position - target
                self.metrics.record_drone_state(
                    f"drone_{d.id}", self.time, d.position, target,
                    d.velocity, d.battery, error_vec, (d.comm_status == 'connected')
                )

            # Advance clock
            self.time += self.dt

        # Ensure civilian events recorded
        for civ in self.civilians:
            self.metrics.record_civilian_event(f"civ_{civ.id}", "detected", civ.detected_time or 1.0)
            if civ.status == CivilianStatus.SAFE:
                self.metrics.record_civilian_event(f"civ_{civ.id}", "evacuated", civ.evacuation_end_time or self.time)
            else:
                self.metrics.record_civilian_event(f"civ_{civ.id}", "failed", self.time)

    def get_metrics(self):
        return self.metrics

    def get_results_summary(self):
        return {
            'time_taken': self.time,
            'replanning_count': self.replanning_count,
            'phase': self.phase,
            'evacuated_count': sum(1 for c in self.civilians if c.status == CivilianStatus.SAFE),
            'total_civilians': len(self.civilians)
        }
