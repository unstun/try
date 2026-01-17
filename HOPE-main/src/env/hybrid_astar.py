import heapq
import math
from dataclasses import dataclass
from typing import Callable, List, Optional, Tuple

import numpy as np

from env.vehicle import State
from configs import (
    GOAL_HEADING_TOL,
    GOAL_TOLERANCE,
    HYBRID_YAW_BINS,
    HYBRID_DT,
    HYBRID_HEADING_WEIGHT,
    HYBRID_MAX_NODES,
    HYBRID_MAX_PLAN_DIST,
    HYBRID_SIM_STEPS,
    discrete_actions,
    VALID_SPEED,
    VALID_STEER,
    WHEEL_BASE,
)

ActionValidator = Callable[[State, float, float], bool]


@dataclass
class HybridAStarPath:
    states: List[State]
    controls: List[Tuple[float, float]]
    L: float


@dataclass
class _Node:
    state: State
    g: float
    f: float
    parent: Optional["_Node"]
    action: Optional[Tuple[float, float]]
    trajectory: List[State]


class HybridAStar:
    """
    Lightweight Hybrid A* planner that respects Ackermann kinematics and
    uses an optional action validator hook to prune unsafe motion primitives.
    """

    def __init__(
        self,
        occupancy_grid: np.ndarray,
        resolution: float,
        origin: Tuple[float, float],
        action_validator: Optional[ActionValidator] = None,
        max_nodes: int = HYBRID_MAX_NODES,
        action_set: Optional[List[Tuple[float, float]]] = None,
    ) -> None:
        self.grid = occupancy_grid
        self.resolution = resolution
        self.origin = origin
        self.action_validator = action_validator
        self.max_nodes = max_nodes
        self.yaw_bins = HYBRID_YAW_BINS
        if action_set is None:
            action_set = [(float(u[0]), float(u[1])) for u in discrete_actions]
        self.action_set = [
            (
                float(np.clip(steer, VALID_STEER[0], VALID_STEER[1])),
                float(np.clip(speed, VALID_SPEED[0], VALID_SPEED[1])),
            )
            for steer, speed in action_set
        ]
        self.dt = HYBRID_DT
        self.sim_steps = HYBRID_SIM_STEPS

    def plan(
        self,
        start_state: State,
        goal_state: State,
        goal_tolerance: float = GOAL_TOLERANCE,
        heading_tolerance: Optional[float] = GOAL_HEADING_TOL,
    ) -> Optional[HybridAStarPath]:
        start_idx = self._state_to_idx(start_state)
        goal_idx = self._state_to_idx(goal_state)
        if start_idx is None or goal_idx is None:
            return None
        if not self._grid_free(start_idx[0], start_idx[1]) or not self._grid_free(goal_idx[0], goal_idx[1]):
            return None

        open_set = []
        counter = 0
        start_node = _Node(
            state=start_state,
            g=0.0,
            f=self._heuristic(start_state, goal_state, goal_tolerance, heading_tolerance),
            parent=None,
            action=None,
            trajectory=[],
        )
        heapq.heappush(open_set, (start_node.f, counter, start_node))
        counter += 1

        best_cost = {self._key(start_idx): 0.0}

        while open_set and counter < self.max_nodes:
            _, _, current = heapq.heappop(open_set)
            curr_dist = current.state.loc.distance(goal_state.loc)
            if curr_dist <= goal_tolerance and self._heading_ok(current.state, goal_state, heading_tolerance):
                return self._reconstruct(current)

            for steer, speed in self.action_set:
                if self.action_validator is not None and not self.action_validator(current.state, steer, speed):
                    continue
                trajectory = self._rollout(current.state, steer, speed)
                if not trajectory:
                    continue
                if not self._segment_valid(trajectory):
                    continue

                new_state = trajectory[-1]
                segment_cost = self._segment_cost(trajectory, speed)
                tentative_g = current.g + segment_cost
                idx = self._state_to_idx(new_state)
                if idx is None:
                    continue
                key = self._key(idx)
                if key in best_cost and tentative_g >= best_cost[key] - 1e-6:
                    continue

                best_cost[key] = tentative_g
                h = self._heuristic(new_state, goal_state, goal_tolerance, heading_tolerance)
                node = _Node(
                    state=new_state,
                    g=tentative_g,
                    f=tentative_g + h,
                    parent=current,
                    action=(steer, speed),
                    trajectory=trajectory,
                )
                heapq.heappush(open_set, (node.f, counter, node))
                counter += 1

        return None

    def _segment_cost(self, trajectory: List[State], speed: float) -> float:
        path_len = abs(speed) * self.dt * len(trajectory)
        return path_len

    def _heading_ok(self, state: State, goal_state: State, heading_tolerance: Optional[float]) -> bool:
        if heading_tolerance is None:
            return True
        heading_err = abs(self._wrap_angle(state.heading - goal_state.heading))
        return heading_err <= heading_tolerance

    def _heuristic(
        self,
        state: State,
        goal_state: State,
        goal_tolerance: float,
        heading_tolerance: Optional[float],
    ) -> float:
        dist = state.loc.distance(goal_state.loc)
        heading_err = abs(self._wrap_angle(state.heading - goal_state.heading))
        heading_weight = HYBRID_HEADING_WEIGHT if heading_tolerance is None else 0.0
        return max(0.0, dist - goal_tolerance) + heading_weight * heading_err

    def _wrap_angle(self, angle: float) -> float:
        return math.atan2(math.sin(angle), math.cos(angle))

    def _rollout(self, state: State, steer: float, speed: float) -> List[State]:
        if abs(speed) < 1e-3:
            return []
        x, y, theta = state.loc.x, state.loc.y, state.heading
        steer = float(np.clip(steer, VALID_STEER[0], VALID_STEER[1]))
        speed = float(np.clip(speed, VALID_SPEED[0], VALID_SPEED[1]))
        traj = []
        for _ in range(self.sim_steps):
            x += speed * math.cos(theta) * self.dt
            y += speed * math.sin(theta) * self.dt
            theta += speed * math.tan(steer) / WHEEL_BASE * self.dt
            traj.append(State([x, y, theta, speed, steer]))
        return traj

    def _segment_valid(self, traj: List[State]) -> bool:
        if not traj:
            return False
        for st in traj:
            gx, gy = self._state_to_grid(st.loc.x, st.loc.y)
            if gx is None or gy is None or not self._grid_free(gx, gy):
                return False
        return True

    def _state_to_idx(self, state: State) -> Optional[Tuple[int, int, int]]:
        gx, gy = self._state_to_grid(state.loc.x, state.loc.y)
        if gx is None:
            return None
        yaw_bin = self._yaw_to_bin(state.heading)
        return gx, gy, yaw_bin

    def _state_to_grid(self, x: float, y: float) -> Tuple[Optional[int], Optional[int]]:
        gx = int((x - self.origin[0]) / self.resolution)
        gy = int((y - self.origin[1]) / self.resolution)
        if gx < 0 or gy < 0 or gx >= self.grid.shape[1] or gy >= self.grid.shape[0]:
            return None, None
        return gx, gy

    def _grid_free(self, gx: int, gy: int) -> bool:
        if gx < 0 or gy < 0 or gx >= self.grid.shape[1] or gy >= self.grid.shape[0]:
            return False
        return self.grid[gy, gx] == 0

    def _yaw_to_bin(self, yaw: float) -> int:
        yaw = (yaw + 2 * math.pi) % (2 * math.pi)
        bin_size = 2 * math.pi / self.yaw_bins
        return int(yaw // bin_size)

    def _key(self, idx: Tuple[int, int, int]) -> Tuple[int, int, int]:
        return idx

    def _reconstruct(self, node: _Node) -> HybridAStarPath:
        rev_states: List[State] = []
        rev_controls: List[Tuple[float, float]] = []
        curr = node
        while curr.parent is not None:
            rev_states.extend(reversed(curr.trajectory))
            rev_controls.append(curr.action)
            curr = curr.parent
        rev_states.append(curr.state)

        states = list(reversed(rev_states))
        controls = list(reversed(rev_controls))
        total_length = sum(abs(u[1]) * self.dt * self.sim_steps for u in controls)
        return HybridAStarPath(states=states, controls=controls, L=total_length)


def plan_hybrid_astar(
    occupancy_grid: np.ndarray,
    resolution: float,
    origin: Tuple[float, float],
    start_state: State,
    goal_state: State,
    action_validator: Optional[ActionValidator] = None,
) -> Optional[HybridAStarPath]:
    """
    Convenience wrapper used by the environment.
    """
    # quick distance gate to avoid heavy planning when the goal is far away
    if start_state.loc.distance(goal_state.loc) > HYBRID_MAX_PLAN_DIST:
        return None
    planner = HybridAStar(
        occupancy_grid=occupancy_grid,
        resolution=resolution,
        origin=origin,
        action_validator=action_validator,
    )
    return planner.plan(start_state, goal_state)
