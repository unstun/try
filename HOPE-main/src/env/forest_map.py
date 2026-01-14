import math
from typing import List, Tuple

import numpy as np
from shapely.geometry import LinearRing, Point, Polygon, box

from env.vehicle import State
from env.map_base import Area
from configs import (
    MAP_LEVEL,
    FOREST_GRID_RES,
    FOREST_MAP_SIZE,
    FOREST_OBS_DENSITY,
    FOREST_OBS_RADIUS_RANGE,
    FOREST_MIN_CORRIDOR,
    FOREST_VEHICLE_CLEARANCE,
    FOREST_BOUNDARY_MARGIN,
    FOREST_SPAWN_MARGIN,
    FOREST_MAX_SAMPLE_RETRY,
    FOREST_MIN_GOAL_SEPARATION,
    FOREST_MAX_GOAL_SEPARATION,
    FOREST_POLY_RESOLUTION,
    GOAL_TOLERANCE,
)


class ForestMap:
    def __init__(self, map_level: str = MAP_LEVEL, seed: int = None) -> None:
        self.map_level = map_level if map_level in FOREST_MAP_SIZE else MAP_LEVEL
        self.seed = seed
        self.rng = np.random.default_rng(seed)

        self.case_id: int = None
        self.start: State = None
        self.dest: State = None
        self.start_box: LinearRing = None
        self.dest_box: LinearRing = None
        self.xmin = 0.0
        self.xmax = 0.0
        self.ymin = 0.0
        self.ymax = 0.0
        self.n_obstacle = 0
        self.obstacles: List[Area] = []
        self.occupancy_grid: np.ndarray = None
        self.origin: Tuple[float, float] = (0.0, 0.0)  # lower-left corner of grid
        self.grid_resolution = FOREST_GRID_RES
        self.goal_tolerance = GOAL_TOLERANCE

        self.params = self._get_level_params(self.map_level)
        self._obstacle_polygons: List[Polygon] = []
        self.grid_width = 0
        self.grid_height = 0

    def _get_level_params(self, level: str) -> dict:
        if level not in FOREST_MAP_SIZE:
            level = MAP_LEVEL
        return {
            "size": FOREST_MAP_SIZE[level],
            "density": FOREST_OBS_DENSITY[level],
            "radius_range": FOREST_OBS_RADIUS_RANGE[level],
            "min_corridor": FOREST_MIN_CORRIDOR[level],
            "min_goal": FOREST_MIN_GOAL_SEPARATION[level],
            "max_goal": FOREST_MAX_GOAL_SEPARATION[level],
        }

    def reset(self, case_id: int = None, path: str = None) -> State:
        """
        Generate a new forest map and sample start/goal.
        The return value matches the parking map API (returns the start state).
        """
        self.case_id = int(self.rng.integers(0, 1e6)) if case_id is None else case_id
        if case_id is not None:
            base_seed = self.seed if self.seed is not None else 0
            # deterministic but unique per case id
            self.rng = np.random.default_rng(base_seed + case_id)

        for _ in range(FOREST_MAX_SAMPLE_RETRY):
            self._init_bounds()
            if not self._generate_obstacles():
                continue
            if self._sample_start_and_goal():
                return self.start
        raise RuntimeError("Failed to sample a valid forest map after multiple attempts.")

    def _init_bounds(self) -> None:
        width, height = self.params["size"]
        self.origin = (-width / 2.0, -height / 2.0)
        self.xmin, self.ymin = self.origin
        self.xmax = self.origin[0] + width
        self.ymax = self.origin[1] + height
        self.grid_width = int(math.ceil(width / self.grid_resolution))
        self.grid_height = int(math.ceil(height / self.grid_resolution))
        self._obstacle_polygons.clear()
        self.occupancy_grid = np.zeros((self.grid_height, self.grid_width), dtype=np.uint8)

    def _target_obstacle_count(self) -> int:
        area = self.params["size"][0] * self.params["size"][1]
        expected = max(1, int(area * self.params["density"]))
        low = max(1, int(expected * 0.6))
        high = max(low + 1, int(expected * 1.4))
        return int(self.rng.integers(low, high))

    def _shape_inside_bounds(self, shape: Polygon) -> bool:
        minx, miny, maxx, maxy = shape.bounds
        return (
            minx >= self.xmin + FOREST_BOUNDARY_MARGIN
            and maxx <= self.xmax - FOREST_BOUNDARY_MARGIN
            and miny >= self.ymin + FOREST_BOUNDARY_MARGIN
            and maxy <= self.ymax - FOREST_BOUNDARY_MARGIN
        )

    def _rasterize_shape(self, shape: Polygon, grid_shape: Tuple[int, int]) -> np.ndarray:
        minx, miny, maxx, maxy = shape.bounds
        gx_min = int(math.floor((minx - self.origin[0]) / self.grid_resolution))
        gx_max = int(math.ceil((maxx - self.origin[0]) / self.grid_resolution))
        gy_min = int(math.floor((miny - self.origin[1]) / self.grid_resolution))
        gy_max = int(math.ceil((maxy - self.origin[1]) / self.grid_resolution))

        gx_min = max(gx_min, 0)
        gy_min = max(gy_min, 0)
        gx_max = min(gx_max, grid_shape[1])
        gy_max = min(gy_max, grid_shape[0])
        cells = []
        for gx in range(gx_min, gx_max):
            for gy in range(gy_min, gy_max):
                cell_poly = box(
                    self.origin[0] + gx * self.grid_resolution,
                    self.origin[1] + gy * self.grid_resolution,
                    self.origin[0] + (gx + 1) * self.grid_resolution,
                    self.origin[1] + (gy + 1) * self.grid_resolution,
                )
                if cell_poly.intersects(shape):
                    cells.append((gx, gy))
        if not cells:
            return None
        return np.array(cells, dtype=int)

    def _generate_obstacles(self) -> bool:
        self.obstacles.clear()
        target_count = self._target_obstacle_count()
        attempts = 0
        min_buffer = max(self.params["min_corridor"] / 2.0, 0.0)
        while len(self._obstacle_polygons) < target_count and attempts < target_count * 30:
            attempts += 1
            radius = float(self.rng.uniform(*self.params["radius_range"]))
            cx = float(self.rng.uniform(self.xmin + FOREST_BOUNDARY_MARGIN + radius,
                                        self.xmax - FOREST_BOUNDARY_MARGIN - radius))
            cy = float(self.rng.uniform(self.ymin + FOREST_BOUNDARY_MARGIN + radius,
                                        self.ymax - FOREST_BOUNDARY_MARGIN - radius))
            obstacle = Point(cx, cy).buffer(radius, resolution=FOREST_POLY_RESOLUTION)
            if not self._shape_inside_bounds(obstacle):
                continue
            padded = obstacle.buffer(min_buffer + FOREST_VEHICLE_CLEARANCE)
            if any(padded.intersects(exist.buffer(FOREST_VEHICLE_CLEARANCE)) for exist in self._obstacle_polygons):
                continue

            inflated = obstacle.buffer(FOREST_VEHICLE_CLEARANCE)
            occupied_cells = self._rasterize_shape(inflated, self.occupancy_grid.shape)
            if occupied_cells is None:
                continue
            if self.occupancy_grid[occupied_cells[:, 1], occupied_cells[:, 0]].any():
                continue

            self.occupancy_grid[occupied_cells[:, 1], occupied_cells[:, 0]] = 1
            self._obstacle_polygons.append(obstacle)

        self.obstacles = [
            Area(shape=poly.exterior, subtype="obstacle", color=(150, 150, 150, 255))
            for poly in self._obstacle_polygons
        ]
        self.n_obstacle = len(self.obstacles)
        return self.n_obstacle > 0

    def _world_to_grid(self, x: float, y: float) -> Tuple[int, int]:
        gx = int((x - self.origin[0]) / self.grid_resolution)
        gy = int((y - self.origin[1]) / self.grid_resolution)
        return gx, gy

    def _grid_to_world(self, gx: int, gy: int) -> Tuple[float, float]:
        wx = self.origin[0] + (gx + 0.5) * self.grid_resolution
        wy = self.origin[1] + (gy + 0.5) * self.grid_resolution
        return wx, wy

    def _sample_free_cell(self, margin_cells: int) -> Tuple[int, int]:
        max_x = self.grid_width - margin_cells
        max_y = self.grid_height - margin_cells
        if max_x <= margin_cells or max_y <= margin_cells:
            return None
        for _ in range(FOREST_MAX_SAMPLE_RETRY):
            gx = int(self.rng.integers(margin_cells, max_x))
            gy = int(self.rng.integers(margin_cells, max_y))
            if self.occupancy_grid[gy, gx] == 0:
                return gx, gy
        return None

    def _is_reachable(self, start: Tuple[int, int], goal: Tuple[int, int]) -> bool:
        """
        BFS on the inflated occupancy grid to guarantee connectivity.
        """
        goal_cells = max(1, int(math.ceil(self.goal_tolerance / self.grid_resolution)))
        visited = np.zeros_like(self.occupancy_grid, dtype=np.uint8)
        queue: List[Tuple[int, int]] = [start]
        visited[start[1], start[0]] = 1
        dirs = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]
        while queue:
            cx, cy = queue.pop(0)
            if abs(cx - goal[0]) <= goal_cells and abs(cy - goal[1]) <= goal_cells:
                return True
            for dx, dy in dirs:
                nx, ny = cx + dx, cy + dy
                if nx < 0 or ny < 0 or nx >= self.grid_width or ny >= self.grid_height:
                    continue
                if visited[ny, nx] or self.occupancy_grid[ny, nx]:
                    continue
                visited[ny, nx] = 1
                queue.append((nx, ny))
        return False

    def _max_goal_dist(self) -> float:
        diagonal = math.hypot(self.params["size"][0], self.params["size"][1])
        return min(self.params["max_goal"], diagonal - 2 * FOREST_SPAWN_MARGIN)

    def _sample_start_and_goal(self) -> bool:
        margin_cells = max(1, int(math.ceil(FOREST_SPAWN_MARGIN / self.grid_resolution)))
        max_goal_dist = self._max_goal_dist()
        for _ in range(FOREST_MAX_SAMPLE_RETRY):
            start_cell = self._sample_free_cell(margin_cells)
            if start_cell is None:
                continue
            start_pos = self._grid_to_world(*start_cell)

            for _ in range(FOREST_MAX_SAMPLE_RETRY):
                goal_cell = self._sample_free_cell(margin_cells)
                if goal_cell is None:
                    continue
                goal_pos = self._grid_to_world(*goal_cell)
                dist = math.hypot(goal_pos[0] - start_pos[0], goal_pos[1] - start_pos[1])
                if dist < self.params["min_goal"] or dist > max_goal_dist:
                    continue
                if not self._is_reachable(start_cell, goal_cell):
                    continue

                start_heading = float(self.rng.uniform(-math.pi, math.pi))
                goal_heading = math.atan2(goal_pos[1] - start_pos[1], goal_pos[0] - start_pos[0])
                self.start = State([start_pos[0], start_pos[1], start_heading, 0, 0])
                self.dest = State([goal_pos[0], goal_pos[1], goal_heading, 0, 0])
                self.start_box = self.start.create_box()
                self.dest_box = Point(goal_pos).buffer(self.goal_tolerance, resolution=FOREST_POLY_RESOLUTION).exterior
                return True
        return False
