"""
A* Path Planning Algorithm on 8-connected grid with continuous waypoint generation.
"""

import heapq
import time
import numpy as np
from planning.utils import cells_to_world, path_length_world, smooth_path

def heuristic(a, b):
    # Euclidean distance
    return np.sqrt((a[0] - b[0])**2 + (a[1] - b[1])**2)

def plan(grid, start_cell, goal_cell, resolution, origin=(0.0, 0.0)):
    """
    A* planner.
    start_cell, goal_cell: (row, col)
    """
    start_time = time.time()
    rows, cols = grid.shape
    start = (int(start_cell[0]), int(start_cell[1]))
    goal = (int(goal_cell[0]), int(goal_cell[1]))

    if grid[start[0], start[1]] == 1 or grid[goal[0], goal[1]] == 1:
        # If start or goal in obstacle, find nearest free cell
        for radius in range(1, 15):
            found = False
            for dr in range(-radius, radius + 1):
                for dc in range(-radius, radius + 1):
                    nr, nc = goal[0] + dr, goal[1] + dc
                    if 0 <= nr < rows and 0 <= nc < cols and grid[nr, nc] == 0:
                        goal = (nr, nc)
                        found = True
                        break
                if found:
                    break
            if found:
                break

    open_set = []
    heapq.heappush(open_set, (0.0, start))
    came_from = {}
    g_score = {start: 0.0}
    nodes_explored = 0

    # 8-connected motions with costs
    motions = [
        (-1, 0, 1.0), (1, 0, 1.0), (0, -1, 1.0), (0, 1, 1.0),
        (-1, -1, np.sqrt(2)), (-1, 1, np.sqrt(2)), (1, -1, np.sqrt(2)), (1, 1, np.sqrt(2))
    ]

    success = False
    while open_set:
        _, current = heapq.heappop(open_set)
        nodes_explored += 1

        if current == goal:
            success = True
            break

        for dr, dc, cost in motions:
            neighbor = (current[0] + dr, current[1] + dc)
            if 0 <= neighbor[0] < rows and 0 <= neighbor[1] < cols:
                if grid[neighbor[0], neighbor[1]] == 1:
                    continue
                tentative_g = g_score[current] + cost
                if neighbor not in g_score or tentative_g < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                    f_score = tentative_g + heuristic(neighbor, goal)
                    heapq.heappush(open_set, (f_score, neighbor))

    planning_time = time.time() - start_time
    path_cells = []
    if success:
        curr = goal
        while curr in came_from:
            path_cells.append(curr)
            curr = came_from[curr]
        path_cells.append(start)
        path_cells.reverse()

    path_world = cells_to_world(path_cells, resolution, origin)
    length = path_length_world(path_world)

    return {
        'path_cells': path_cells,
        'path_world': path_world,
        'nodes_explored': nodes_explored,
        'planning_time': planning_time,
        'path_length': length,
        'success': success
    }
