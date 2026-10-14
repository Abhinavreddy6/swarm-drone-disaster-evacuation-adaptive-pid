"""
D* Lite Incremental Path Planning and Replanning Algorithm.
Reuses previous search state when obstacles change dynamically.
"""

import heapq
import time
import numpy as np
from planning.utils import cells_to_world, path_length_world, smooth_path

class DStarLite:
    def __init__(self, grid, start_cell, goal_cell, resolution, origin=(0.0, 0.0)):
        self.grid = np.copy(grid)
        self.rows, self.cols = grid.shape
        self.start = (int(start_cell[0]), int(start_cell[1]))
        self.goal = (int(goal_cell[0]), int(goal_cell[1]))
        self.resolution = resolution
        self.origin = origin

        self.k_m = 0.0
        self.rhs = {}
        self.g = {}
        self.U = [] # Priority queue of (key, counter, u)
        self.counter = 0
        self.entry_finder = {} # cell -> entry
        self.REMOVED = (-999, -999)
        self.nodes_explored = 0

        # Motions
        self.motions = [
            (-1, 0, 1.0), (1, 0, 1.0), (0, -1, 1.0), (0, 1, 1.0),
            (-1, -1, np.sqrt(2)), (-1, 1, np.sqrt(2)), (1, -1, np.sqrt(2)), (1, 1, np.sqrt(2))
        ]

        self.initialize()

    def heuristic(self, u, v):
        return np.sqrt((u[0] - v[0])**2 + (u[1] - v[1])**2)

    def calculate_key(self, s):
        g_rhs = min(self.g.get(s, float('inf')), self.rhs.get(s, float('inf')))
        return (g_rhs + self.heuristic(self.start, s) + self.k_m, g_rhs)

    def initialize(self):
        self.U = []
        self.entry_finder = {}
        self.counter = 0
        self.k_m = 0.0
        self.rhs = {self.goal: 0.0}
        self.g = {}
        self.insert_u(self.goal, self.calculate_key(self.goal))

    def insert_u(self, u, key):
        self.counter += 1
        entry = [key, self.counter, u]
        self.entry_finder[u] = entry
        heapq.heappush(self.U, entry)

    def remove_u(self, u):
        if u in self.entry_finder:
            entry = self.entry_finder.pop(u)
            entry[-1] = self.REMOVED

    def update_vertex(self, u):
        if u != self.goal:
            min_rhs = float('inf')
            for dr, dc, cost in self.motions:
                nbr = (u[0] + dr, u[1] + dc)
                if 0 <= nbr[0] < self.rows and 0 <= nbr[1] < self.cols:
                    if self.grid[nbr[0], nbr[1]] == 0:
                        val = cost + self.g.get(nbr, float('inf'))
                        if val < min_rhs:
                            min_rhs = val
            self.rhs[u] = min_rhs

        self.remove_u(u)
        if self.g.get(u, float('inf')) != self.rhs.get(u, float('inf')):
            self.insert_u(u, self.calculate_key(u))

    def compute_shortest_path(self):
        max_expansions = self.rows * self.cols * 2
        expansions = 0
        while self.U and expansions < max_expansions:
            while self.U and self.U[0][-1] == self.REMOVED:
                heapq.heappop(self.U)
            if not self.U:
                break

            k_old, _, u = self.U[0]
            k_new = self.calculate_key(u)

            if k_old < k_new:
                heapq.heappop(self.U)
                if u != self.REMOVED:
                    self.insert_u(u, k_new)
            elif self.g.get(self.start, float('inf')) > self.rhs.get(self.start, float('inf')) or k_old < self.calculate_key(self.start):
                heapq.heappop(self.U)
                if u == self.REMOVED:
                    continue
                self.nodes_explored += 1
                expansions += 1
                if self.g.get(u, float('inf')) > self.rhs.get(u, float('inf')):
                    self.g[u] = self.rhs.get(u, float('inf'))
                    for dr, dc, cost in self.motions:
                        s = (u[0] + dr, u[1] + dc)
                        if 0 <= s[0] < self.rows and 0 <= s[1] < self.cols:
                            if self.grid[s[0], s[1]] == 0:
                                self.update_vertex(s)
                else:
                    self.g[u] = float('inf')
                    self.update_vertex(u)
                    for dr, dc, cost in self.motions:
                        s = (u[0] + dr, u[1] + dc)
                        if 0 <= s[0] < self.rows and 0 <= s[1] < self.cols:
                            if self.grid[s[0], s[1]] == 0:
                                self.update_vertex(s)
            else:
                break

    def extract_path(self):
        curr = self.start
        path = [curr]
        visited = {curr}
        for _ in range(self.rows * self.cols):
            if curr == self.goal:
                break
            best_nbr = None
            min_cost = float('inf')
            for dr, dc, cost in self.motions:
                nbr = (curr[0] + dr, curr[1] + dc)
                if 0 <= nbr[0] < self.rows and 0 <= nbr[1] < self.cols:
                    if self.grid[nbr[0], nbr[1]] == 0:
                        val = cost + self.g.get(nbr, float('inf'))
                        if val < min_cost:
                            min_cost = val
                            best_nbr = nbr
            if best_nbr is None or best_nbr in visited:
                break
            curr = best_nbr
            path.append(curr)
            visited.add(curr)
        return path

    def plan(self):
        t0 = time.time()
        self.compute_shortest_path()
        t_elapsed = time.time() - t0
        path_cells = self.extract_path()
        success = (len(path_cells) > 0 and path_cells[-1] == self.goal)
        path_world = cells_to_world(path_cells, self.resolution, self.origin)
        length = path_length_world(path_world)
        return {
            'path_cells': path_cells,
            'path_world': path_world,
            'nodes_explored': self.nodes_explored,
            'planning_time': t_elapsed,
            'path_length': length,
            'success': success
        }

    def update_obstacles(self, changed_cells, new_state=1):
        for r, c in changed_cells:
            if 0 <= r < self.rows and 0 <= c < self.cols:
                self.grid[r, c] = new_state
                self.update_vertex((r, c))
                for dr, dc, _ in self.motions:
                    nbr = (r + dr, c + dc)
                    if 0 <= nbr[0] < self.rows and 0 <= nbr[1] < self.cols:
                        self.update_vertex(nbr)

    def replan(self, current_pos_cell):
        t0 = time.time()
        self.k_m += self.heuristic(self.start, current_pos_cell)
        self.start = current_pos_cell
        self.compute_shortest_path()
        t_elapsed = time.time() - t0
        path_cells = self.extract_path()
        success = (len(path_cells) > 0 and path_cells[-1] == self.goal)
        path_world = cells_to_world(path_cells, self.resolution, self.origin)
        length = path_length_world(path_world)
        return {
            'path_cells': path_cells,
            'path_world': path_world,
            'nodes_explored': self.nodes_explored,
            'planning_time': t_elapsed,
            'path_length': length,
            'success': success
        }

def plan(grid, start_cell, goal_cell, resolution, origin=(0.0, 0.0)):
    planner = DStarLite(grid, start_cell, goal_cell, resolution, origin)
    return planner.plan()
