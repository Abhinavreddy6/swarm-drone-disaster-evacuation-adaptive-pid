# Swarm Drone-Based Disaster Detection & Civilian Evacuation Using Intelligent Path Planning

A continuous-space 3D/2.5D multi-drone simulation environment for disaster response, hazard detection, and civilian evacuation using closed-loop PID control and sensor fusion.

> **Research Integrity Disclaimer**: This project is a **simulation-based evaluation**. All results are obtained using simulated sensors (GPS with noise, IMU with bias and drift), simulated aerodynamic wind turbulence, and synthetic 3D continuous environment models. No real flight testing, physical GPS/IMU hardware, or Pixhawk hardware-in-the-loop validation is claimed.

---

## 1. Project Overview

This implementation rejects discrete grid-cell teleportation (e.g. 50×50 cell-by-cell hops) in favor of **continuous floating-point 3D/2.5D coordinates** $(x, y, z) \in \mathbb{R}^3$. Drones operate using continuous kinematics and dynamics, closed-loop PID position control, Kalman filter sensor fusion, and real-time obstacle replanning.

### Swarm Drone Roles:
1. **Scout Drone (Orange)**: Performs continuous lawnmower/spline search patterns, detects disaster perimeters and civilian clusters within a continuous sensing radius ($r = 30\text{ m}$).
2. **Relay Drone (Blue)**: Maintains line-of-sight and communication links between the base station and active swarm drones by dynamically positioning at the centroid of active operations.
3. **Guide Drone (Green)**: Receives detected civilian coordinates, coordinates smooth evacuation paths toward the safe extraction zone, and guides civilians safely.
4. **Safe-Zone Drone (Purple)**: Patrols and confirms safe-zone perimeter availability ($450\text{ m}, 450\text{ m}$).

---

## 2. Project Structure

```
project/
├── main.py                     # Master execution script (runs all benchmarks & outputs plots)
├── config.py                   # Central simulation hyperparameters and random seed
├── requirements.txt            # Python dependencies (numpy, matplotlib, scipy)
├── README.md                   # Complete documentation
│
├── simulation/
│   ├── environment.py          # Continuous 3D environment, buildings, disaster zones, trees
│   ├── drone.py                # Drone agent class with continuous state, battery, telemetry
│   ├── civilian.py             # Civilian object with continuous coordinates and state machine
│   ├── dynamics.py             # Quadrotor physics, drag, wind effects, heading/attitude
│   ├── sensors.py              # Simulated GPS, IMU, and Kalman Filter sensor fusion
│   └── simulator.py            # Main simulation loop with swarm coordination and replanning
│
├── planning/
│   ├── utils.py                # Spline smoothing (Catmull-Rom), metrics, continuous conversion
│   ├── astar.py                # A* path planner (8-connected with Euclidean heuristic)
│   ├── dijkstra.py             # Dijkstra shortest path planner
│   ├── bfs.py                  # Breadth-first search planner
│   ├── greedy.py               # Greedy best-first search planner
│   └── dstar_lite.py           # D* Lite incremental planner with dynamic replanning
│
├── control/
│   ├── pid.py                  # 1D PID controller with anti-windup clamping
│   ├── position_controller.py  # 3-axis continuous PID position controller (X, Y, Z)
│   └── attitude_controller.py  # Heading (yaw) and bank angle (pitch/roll) controller
│
├── evaluation/
│   ├── metrics.py              # Telemetry logger, RMSE, error, and evacuation metrics
│   ├── experiments.py          # 7 benchmark experiments suite saving to CSV
│   └── plots.py                # Matplotlib generators saving figures to PNG and PDF
│
├── visualization/
│   └── threejs/
│       └── index.html          # Interactive 3D WebGL Three.js disaster response visualization
│
└── results/
    ├── figures/                # Output graphs (.png & .pdf)
    └── csv/                    # Benchmark result tables (.csv)
```

---

## 3. How to Run

### Requirements
Ensure Python 3.9+ is installed with the required dependencies:
```bash
pip install -r requirements.txt
```

### Running the Complete Simulation Suite
Execute the master runner:
```bash
python main.py
```
This executes:
1. Environment generation (500m × 500m continuous space with 18 buildings and 2 disaster zones).
2. 5 path planning benchmarks (A*, Dijkstra, BFS, Greedy, D* Lite).
3. Continuous closed-loop PID trajectory tracking.
4. Wind disturbance experiments (none, moderate, strong).
5. Sensor fusion performance under GPS denial and degradation.
6. Real-time dynamic obstacle injection and replanning.
7. Battery depletion analysis under continuous flight dynamics.
8. Civilian evacuation coordination.
9. Exports 7 structured CSV files into `results/csv/`.
10. Generates 11 publication-grade figures in PNG and PDF into `results/figures/`.

### Viewing the 3D WebGL Visualization
Open the interactive Three.js simulation in your web browser:
```bash
# Simply open the HTML file in any modern browser:
c:\robo\project\visualization\threejs\index.html
```
Features:
- Real-time OrbitControls (rotate, zoom, pan).
- Drones moving smoothly through 3D continuous space with rotor blur and attitude banking.
- Live HUD displaying drone positions $(x, y, z)$, velocities, battery percentages, and communication links.
- Interactive controls: Play/Pause, speed adjustment (1×, 2×, 5×), trail toggles, and comm link toggles.

---

## 4. Control & Estimation Architecture

The proposed hierarchy links global path planning to low-level motor acceleration commands:

$$\text{Global Waypoints} \xrightarrow{\text{Spline}} \mathbf{p}_d(t) \xrightarrow{\text{PID}} \mathbf{a}_d(t) \xrightarrow{\text{Attitude}} (\phi, \theta, \psi) \xrightarrow{\text{Dynamics}} \mathbf{p}(t) \xrightarrow{\text{Sensors + KF}} \hat{\mathbf{p}}(t)$$

- **Position Controller**: Independent PID channels along X, Y, and Z compute the required 3D acceleration vector $\mathbf{a}_d = [a_x, a_y, a_z]^T$.
- **Attitude Controller**: Commands yaw $\psi$ along the horizontal velocity vector $\mathbf{v}_{xy}$ while computing bank angles $\theta$ (pitch) and $\phi$ (roll) based on inertial acceleration.
- **Sensor Fusion**: Fuses 100 Hz IMU linear acceleration predictions with 5 Hz noisy GPS updates using a 6-state Kalman filter $[\mathbf{p}, \mathbf{v}]^T$ to maintain state estimates $\hat{\mathbf{p}}$ during GPS dropouts.
