#!/usr/bin/env python3
"""
Root entry point for 3D Drone Swarm Disaster Evacuation & Adaptive PID Simulation.
Delegates to project/main.py.
"""
import sys
import os

project_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'project')
if project_dir not in sys.path:
    sys.path.insert(0, project_dir)

from main import main

if __name__ == '__main__':
    main()
