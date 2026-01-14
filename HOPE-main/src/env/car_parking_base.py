'''
Ackermann car navigation in a forest-style occupancy grid with random start/goal.
'''


import sys
sys.path.append("../")
from typing import Optional
import math
from typing import OrderedDict

import numpy as np
import gym
from gym import spaces
from gym.error import DependencyNotInstalled
from shapely.affinity import affine_transform
try:
    # As pygame is necessary for using the environment (reset and step) even without a render mode
    #   therefore, pygame is a necessary import for the environment.
    import pygame
except ImportError:
    raise DependencyNotInstalled(
        "pygame is not installed, run `pip install pygame`"
    )

from env.vehicle import *
from env.map_base import *
from env.lidar_simulator import LidarSimlator
from env.forest_map import ForestMap
from env.hybrid_astar import plan_hybrid_astar
from env.observation_processor import Obs_Processor
from model.action_mask import ActionMask
from configs import *

class CarParking(gym.Env):

    metadata = {
        "render_mode": [
            "human", 
            "rgb_array",
        ]
    }

    def __init__(
        self, 
        render_mode: str = None,
        fps: int = FPS,
        verbose: bool =True, 
        use_lidar_observation: bool =USE_LIDAR,
        use_img_observation: bool=USE_IMG,
        use_action_mask: bool=USE_ACTION_MASK,
    ):
        super().__init__()

        self.verbose = verbose
        self.use_lidar_observation = use_lidar_observation
        self.use_img_observation = use_img_observation
        self.use_action_mask = use_action_mask
        self.render_mode = "human" if render_mode is None else render_mode
        self.fps = fps
        self.screen: Optional[pygame.Surface] = None
        self.matrix = None
        self.clock = None
        self.is_open = True
        self.t = 0.0
        self.k = None
        self.level = MAP_LEVEL
        self.tgt_repr_size = 5 # relative_distance, cos(theta), sin(theta), cos(phi), sin(phi)
        self.goal_tolerance = GOAL_TOLERANCE

        self.map = ForestMap(self.level)
        self.vehicle = Vehicle(n_step=NUM_STEP, step_len=STEP_LENGTH)
        self.lidar = LidarSimlator(LIDAR_RANGE, LIDAR_NUM)
        self.reward = 0.0
        self.prev_reward = 0.0
        self.prev_action = np.zeros(2)

        self.action_space = spaces.Box(
            np.array([VALID_STEER[0], VALID_SPEED[0]]).astype(np.float32),
            np.array([VALID_STEER[1], VALID_SPEED[1]]).astype(np.float32),
        ) # steer, speed
       
        self.observation_space = {}
        if self.use_action_mask:
            self.action_filter = ActionMask()
            self.observation_space['action_mask'] = spaces.Box(low=0, high=1, 
                shape=(N_DISCRETE_ACTION,), dtype=np.float64
            )
        if self.use_img_observation:
            self.img_processor = Obs_Processor()
            self.observation_space['img'] = spaces.Box(low=0, high=255, 
                shape=(OBS_W//self.img_processor.downsample_rate, OBS_H//self.img_processor.downsample_rate, 
                self.img_processor.n_channels), dtype=np.uint8
            )
            self.raw_img_shape = (OBS_W, OBS_H, 3)
        if self.use_lidar_observation:
            # the observation is composed of lidar points and target representation
            # the target representation is (relative_distance, cos(theta), sin(theta), cos(phi), sin(phi))
            # where the theta indicates the relative angle of parking lot, and phi means the heading of 
            # parking lot in the polar coordinate of the ego car's view
            low_bound, high_bound = np.zeros((LIDAR_NUM)), np.ones((LIDAR_NUM))*LIDAR_RANGE
            self.observation_space['lidar'] = spaces.Box(
                low=low_bound, high=high_bound, shape=(LIDAR_NUM,), dtype=np.float64
            )
        low_bound = np.array([0,-1,-1,-1,-1])
        high_bound = np.array([MAX_DIST_TO_DEST,1,1,1,1])
        self.observation_space['target'] = spaces.Box(
            low=low_bound, high=high_bound, shape=(self.tgt_repr_size,), dtype=np.float64
        )
    
    def set_level(self, level:str=None):
        if level is None:
            level = MAP_LEVEL
        self.level = level
        self.map = ForestMap(self.level)

    def reset(self, case_id: int = None, data_dir: str = None, level: str = None,) -> np.ndarray:
        self.reward = 0.0
        self.prev_reward = 0.0
        self.t = 0.0
        self.prev_action = np.zeros(2)

        if level is not None:
            self.set_level(level)
        initial_state = self.map.reset(case_id, data_dir)
        self.vehicle.reset(initial_state)
        self.matrix = self.coord_transform_matrix()
        return self.step()[0]

    def coord_transform_matrix(self) -> list:
        """Get the transform matrix that convert the real world coordinate to the pygame coordinate.
        """
        k = K
        bx = 0.5 * (WIN_W - k * (self.map.xmax + self.map.xmin))
        by = 0.5 * (WIN_H - k * (self.map.ymax + self.map.ymin))
        self.k = k
        return [k, 0, 0, k, bx, by]

    def _world_to_screen(self, x: float, y: float):
        if self.matrix is None:
            return None
        return (int(self.k * x + self.matrix[4]), int(self.k * y + self.matrix[5]))

    def _chaikin_smooth(self, pts, iterations: int = 2):
        """Smooth polyline using Chaikin corner-cutting for a curve-like appearance."""
        if len(pts) < 3:
            return pts
        smoothed = np.array(pts, dtype=float)
        for _ in range(iterations):
            if len(smoothed) < 3:
                break
            new_pts = [smoothed[0]]
            for i in range(len(smoothed) - 1):
                p0, p1 = smoothed[i], smoothed[i + 1]
                q = 0.75 * p0 + 0.25 * p1
                r = 0.25 * p0 + 0.75 * p1
                new_pts.extend([q, r])
            new_pts.append(smoothed[-1])
            smoothed = np.array(new_pts, dtype=float)
        return [(int(round(p[0])), int(round(p[1]))) for p in smoothed]

    def _draw_heading_arrow(self, surface: pygame.Surface, state: State, color):
        """Draw a heading arrow at the state's position."""
        if state is None or self.matrix is None:
            return
        base = self._world_to_screen(state.loc.x, state.loc.y)
        if base is None:
            return
        length = ARROW_LENGTH
        head_scale = ARROW_HEAD_SCALE
        heading = state.heading
        tip_world = (
            state.loc.x + length / self.k * np.cos(heading),
            state.loc.y + length / self.k * np.sin(heading)
        )
        tip = self._world_to_screen(*tip_world)
        if tip is None:
            return
        pygame.draw.line(surface, color[:3], base, tip, 3)
        head_len = length * head_scale
        left_world = (
            tip_world[0] - head_len / self.k * np.cos(heading - np.pi / 6),
            tip_world[1] - head_len / self.k * np.sin(heading - np.pi / 6)
        )
        right_world = (
            tip_world[0] - head_len / self.k * np.cos(heading + np.pi / 6),
            tip_world[1] - head_len / self.k * np.sin(heading + np.pi / 6)
        )
        left_pt = self._world_to_screen(*left_world)
        right_pt = self._world_to_screen(*right_world)
        if left_pt is not None and right_pt is not None:
            pygame.draw.polygon(surface, color[:3], [tip, left_pt, right_pt])

    def _draw_trajectory_line(self, surface: pygame.Surface):
        """Draw an anti-aliased smooth trajectory line through vehicle centers."""
        if not (RENDER_TRAJ and RENDER_TRAJ_LINE):
            return
        if self.matrix is None or len(self.vehicle.trajectory) < 2:
            return
        pts = []
        for st in self.vehicle.trajectory:
            pt = self._world_to_screen(st.loc.x, st.loc.y)
            if pt is not None:
                pts.append(pt)
        if len(pts) < 2:
            return
        smoothed = self._chaikin_smooth(pts, iterations=2)
        pygame.draw.lines(surface, TRAJ_LINE_COLOR[:3], False, smoothed, TRAJ_LINE_WIDTH)
        pygame.draw.aalines(surface, TRAJ_LINE_COLOR[:3], False, smoothed, True)

    def _coord_transform(self, object) -> list:
        transformed = affine_transform(object, self.matrix)
        return list(transformed.coords)

    def _detect_collision(self):
        # return False
        for obstacle in self.map.obstacles:
            if self.vehicle.box.intersects(obstacle.shape):
                return True
        return False
    
    def _detect_outbound(self):
        x, y = self.vehicle.state.loc.x, self.vehicle.state.loc.y
        return x>self.map.xmax or x<self.map.xmin or y>self.map.ymax or y<self.map.ymin

    def _check_arrived(self):
        dist = self.vehicle.state.loc.distance(self.map.dest.loc)
        if dist > self.goal_tolerance:
            return False
        if GOAL_HEADING_TOL is None:
            return True
        heading_err = abs(math.atan2(math.sin(self.vehicle.state.heading - self.map.dest.heading), 
            math.cos(self.vehicle.state.heading - self.map.dest.heading)))
        return heading_err <= GOAL_HEADING_TOL
    
    def _check_time_exceeded(self):
        return self.t > TOLERANT_TIME

    def _check_status(self):
        if self._detect_collision():
            return Status.COLLIDED
        if self._detect_outbound():
            return Status.OUTBOUND
        if self._check_arrived():
            return Status.ARRIVED
        if self._check_time_exceeded():
            return Status.OUTTIME
        return Status.CONTINUE

    def _get_reward(self, prev_state: State, curr_state: State, status: Status, action:np.ndarray):
        prev_dist = prev_state.loc.distance(self.map.dest.loc)
        curr_dist = curr_state.loc.distance(self.map.dest.loc)
        progress = np.clip(prev_dist - curr_dist, -PROGRESS_CLIP, PROGRESS_CLIP)

        collision_penalty = 0.0
        if status == Status.COLLIDED:
            collision_penalty = -COLLISION_PENALTY
        elif status == Status.OUTBOUND:
            collision_penalty = -OUTBOUND_PENALTY
        elif status == Status.OUTTIME:
            collision_penalty = -TIMEOUT_PENALTY

        smoothness_penalty = 0.0
        if action is not None:
            delta_steer = abs(action[0] - self.prev_action[0])
            delta_speed = abs(action[1] - self.prev_action[1])
            steer_norm = max(VALID_STEER[1] - VALID_STEER[0], 1e-6)
            speed_norm = max(VALID_SPEED[1] - VALID_SPEED[0], 1e-6)
            smoothness_penalty = -(
                SMOOTHNESS_STEER_WEIGHT * delta_steer / steer_norm +
                SMOOTHNESS_SPEED_WEIGHT * delta_speed / speed_norm
            )
        return [progress, collision_penalty, smoothness_penalty]
        
    def get_reward(self, status, prev_state, action):
        return self._get_reward(prev_state, self.vehicle.state, status, action)

    def step(self, action:np.ndarray = None):
        '''
        Parameters:
        ----------
        `action`: `np.ndarray`

        Returns:
        ----------
        ``obsercation`` (Dict): 
            the observation of image based surroundings, lidar view and target representation.
            If `use_lidar_observation` is `True`, then `obsercation['img'] = None`.
            If `use_lidar_observation` is `False`, then `obsercation['lidar'] = None`. 

        ``reward_info`` (OrderedDict): reward information, including:
                progress ,collision ,smoothness
        `status` (`Status`): represent the state of vehicle, including:
                `CONTINUE`, `ARRIVED`, `COLLIDED`, `OUTBOUND`, `OUTTIME`
        `info` (`OrderedDict`): other information.
        '''
        assert self.vehicle is not None
        prev_state = self.vehicle.state
        collide = False
        arrive = False
        exec_steps = 0
        if action is not None:
            for simu_step_num in range(NUM_STEP):
                prev_info = self.vehicle.step(action,step_time=1)
                exec_steps += 1
                if self._check_arrived():
                    arrive = True
                    break
                if self._detect_collision():
                    collide = ENV_COLLIDE
                    self.vehicle.retreat(prev_info)
                    exec_steps -= 1
                    break
            if exec_steps > 1:
                del self.vehicle.trajectory[-exec_steps:-1]

        self.t += 1
        observation = self.render(self.render_mode)
        if arrive:
            status = Status.ARRIVED
        else:
            status = Status.COLLIDED if collide else self._check_status()

        reward_list = self.get_reward(status, prev_state, action)
        reward_info = OrderedDict({
            'progress': reward_list[0],
            'collision': reward_list[1],
            'smoothness': reward_list[2],
        })

        if action is not None:
            self.prev_action = np.array(action, dtype=float)

        info = OrderedDict({'reward_info':reward_info, 'path_to_dest':None})
        if self.t > 1 and status==Status.CONTINUE and \
            self.vehicle.state.loc.distance(self.map.dest.loc) <= HYBRID_MAX_PLAN_DIST:
            planner_path = self.find_hybrid_astar_path()
            if planner_path is not None:
                info['path_to_dest'] = planner_path

        return observation, reward_info, status, info

    def _render(self, surface: pygame.Surface):
        surface.fill(BG_COLOR)

        # Base layer: obstacles and ego footprint/trace
        for obstacle in self.map.obstacles:
            pygame.draw.polygon(
                surface, OBSTACLE_COLOR, self._coord_transform(obstacle.shape))

        pygame.draw.polygon(
            surface, self.vehicle.color, self._coord_transform(self.vehicle.box))

        if RENDER_TRAJ and len(self.vehicle.trajectory) > 1:
            render_len = min(len(self.vehicle.trajectory), TRAJ_RENDER_LEN)
            for i in range(render_len):
                vehicle_box = self.vehicle.trajectory[-(render_len-i)].create_box()
                pygame.draw.polygon(
                    surface, TRAJ_COLORS[-(render_len-i)][:3], self._coord_transform(vehicle_box), width=1)

        # Top layer: goal/start boxes and planning curve
        pygame.draw.polygon(
            surface, START_COLOR, self._coord_transform(self.map.start_box), width=START_BORDER_WIDTH)
        pygame.draw.polygon(
            surface, DEST_COLOR, self._coord_transform(self.map.dest_box), width=DEST_BORDER_WIDTH)
        self._draw_trajectory_line(surface)

        # heading arrows at start and current positions stay above everything else
        self._draw_heading_arrow(surface, self.vehicle.initial_state, ARROW_COLOR_START)
        self._draw_heading_arrow(surface, self.vehicle.state, ARROW_COLOR_END)

    def _get_img_observation(self, surface: pygame.Surface):
        angle = self.vehicle.state.heading
        old_center = surface.get_rect().center

        # Rotate and find the center of the vehicle
        capture = pygame.transform.rotate(surface, np.rad2deg(angle))
        rotate = pygame.Surface((WIN_W, WIN_H))
        rotate.blit(capture, capture.get_rect(center=old_center))
        
        vehicle_center = np.array(self._coord_transform(self.vehicle.box.centroid)[0])
        dx = (vehicle_center[0]-old_center[0])*np.cos(angle) \
            + (vehicle_center[1]-old_center[1])*np.sin(angle)
        dy = -(vehicle_center[0]-old_center[0])*np.sin(angle) \
            + (vehicle_center[1]-old_center[1])*np.cos(angle)
        
        # align the center of the observation with the center of the vehicle
        observation = pygame.Surface((WIN_W, WIN_H))
    
        observation.fill(BG_COLOR)
        observation.blit(rotate, (int(-dx), int(-dy)))
        observation = observation.subsurface((
            (WIN_W-OBS_W)/2, (WIN_H-OBS_H)/2), (OBS_W, OBS_H))

    
        obs_str = pygame.image.tostring(observation, "RGB")
        observation = np.frombuffer(obs_str, dtype=np.uint8)
        observation = observation.reshape(self.raw_img_shape)

        return observation

    def _process_img_observation(self, img):
        '''
        Process the img into channels of different information.

        Parameters
        ------
        img (np.ndarray): RGB image of shape (OBS_W, OBS_H, 3)

        Returns
        ------
        processed img (np.ndarray): shape (OBS_W//downsample_rate, OBS_H//downsample_rate, n_channels )
        '''
        processed_img = self.img_processor.process_img(img)
        return processed_img

    def _get_lidar_observation(self,):
        obs_list = [obs.shape for obs in self.map.obstacles]
        lidar_view = self.lidar.get_observation(self.vehicle.state, obs_list)
        return lidar_view
    
    def _get_targt_repr(self,):
        # target position representation
        dest_pos = (self.map.dest.loc.x, self.map.dest.loc.y, self.map.dest.heading)
        ego_pos = (self.vehicle.state.loc.x, self.vehicle.state.loc.y, self.vehicle.state.heading)
        rel_distance = math.sqrt((dest_pos[0]-ego_pos[0])**2 + (dest_pos[1]-ego_pos[1])**2)
        rel_angle = math.atan2(dest_pos[1]-ego_pos[1], dest_pos[0]-ego_pos[0]) - ego_pos[2]
        rel_dest_heading = dest_pos[2] - ego_pos[2]
        tgt_repr = np.array([rel_distance, math.cos(rel_angle), math.sin(rel_angle),\
            math.cos(rel_dest_heading), math.cos(rel_dest_heading)])
        return tgt_repr 

    def render(self, mode: str = "human"):
        assert mode in self.metadata["render_mode"]
        assert self.vehicle is not None

        if not pygame.get_init():
            pygame.init()
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.is_open = False
                self.render_mode = "rgb_array"

        if not self.is_open:
            mode = "rgb_array"
            if self.screen is not None:
                pygame.display.quit()
                self.screen = None
                self.clock = None
            self.is_open = True

        if mode == "human":
            display_flags = pygame.SHOWN
        else:
            display_flags = pygame.HIDDEN
        if self.screen is None:
            pygame.init()
            pygame.display.init()
            self.screen = pygame.display.set_mode((WIN_W, WIN_H), flags = display_flags)
        if self.clock is None:
            self.clock = pygame.time.Clock()

        self._render(self.screen)
        observation = {'img':None, 'lidar':None, 'target':None, 'action_mask':None}
        if self.use_img_observation:
            raw_observation = self._get_img_observation(self.screen)
            observation['img'] = self._process_img_observation(raw_observation)
        if self.use_lidar_observation:
            observation['lidar'] = self._get_lidar_observation()
        if self.use_action_mask:
            observation['action_mask'] = self.action_filter.get_steps(observation['lidar'])
        observation['target'] = self._get_targt_repr()
        pygame.display.update()
        self.clock.tick(self.fps)
        
        return observation

    def find_hybrid_astar_path(self):
        """
        Plan a feasible path with hybrid A* using the occupancy grid.

        Returns:
            HybridAStarPath or None if no feasible path is found.
        """
        if not hasattr(self.map, 'occupancy_grid') or self.map.occupancy_grid is None:
            return None
        origin = getattr(self.map, 'origin', (self.map.xmin, self.map.ymin))
        return plan_hybrid_astar(
            occupancy_grid=self.map.occupancy_grid,
            resolution=self.map.grid_resolution,
            origin=origin,
            start_state=self.vehicle.state,
            goal_state=self.map.dest,
            collision_checker=self.is_traj_valid,
        )

    def find_rs_path(self, status=None):
        # Backward compatibility shim
        return self.find_hybrid_astar_path()
    
    def is_traj_valid(self, traj):
        car_coords1 = np.array(VehicleBox.coords)[:4] # (4,2)
        car_coords2 = np.array(VehicleBox.coords)[1:] # (4,2)
        car_coords_x1 = car_coords1[:,0].reshape(1,-1)
        car_coords_y1 = car_coords1[:,1].reshape(1,-1) # (1,4)
        car_coords_x2 = car_coords2[:,0].reshape(1,-1)
        car_coords_y2 = car_coords2[:,1].reshape(1,-1) # (1,4)
        vxs = np.array([t[0] for t in traj])
        vys = np.array([t[1] for t in traj])
        # check outbound
        if np.min(vxs) < self.map.xmin or np.max(vxs) > self.map.xmax \
        or np.min(vys) < self.map.ymin or np.max(vys) > self.map.ymax:
            return False
        vthetas = np.array([t[2] for t in traj])
        cos_theta = np.cos(vthetas).reshape(-1,1) # (T,1)
        sin_theta = np.sin(vthetas).reshape(-1,1)
        vehicle_coords_x1 = cos_theta*car_coords_x1 - sin_theta*car_coords_y1 + vxs.reshape(-1,1) # (T,4)
        vehicle_coords_y1 = sin_theta*car_coords_x1 + cos_theta*car_coords_y1 + vys.reshape(-1,1)
        vehicle_coords_x2 = cos_theta*car_coords_x2 - sin_theta*car_coords_y2 + vxs.reshape(-1,1) # (T,4)
        vehicle_coords_y2 = sin_theta*car_coords_x2 + cos_theta*car_coords_y2 + vys.reshape(-1,1)
        vx1s = vehicle_coords_x1.reshape(-1,1)
        vx2s = vehicle_coords_x2.reshape(-1,1)
        vy1s = vehicle_coords_y1.reshape(-1,1)
        vy2s = vehicle_coords_y2.reshape(-1,1)
        # Line 1: the edges of vehicle box, ax + by + c = 0
        a = (vy2s - vy1s).reshape(-1,1) # (4*t,1)
        b = (vx1s - vx2s).reshape(-1,1)
        c = (vy1s*vx2s - vx1s*vy2s).reshape(-1,1)
        
        # convert obstacles(LinerRing) to edges ((x1,y1), (x2,y2))
        x_max = np.max(vx1s) + 5
        x_min = np.min(vx1s) - 5
        y_max = np.max(vy1s) + 5
        y_min = np.min(vy1s) - 5

        x1s, x2s, y1s, y2s = [], [], [], []
        for obst in self.map.obstacles:
            if isinstance(obst, Area):
                obst = obst.shape
            obst_coords = np.array(obst.coords) # (n+1,2)
            if (obst_coords[:,0] > x_max).all() or (obst_coords[:,0] < x_min).all()\
                or (obst_coords[:,1] > y_max).all() or (obst_coords[:,1] < y_min).all():
                continue
            x1s.extend(list(obst_coords[:-1, 0]))
            x2s.extend(list(obst_coords[1:, 0]))
            y1s.extend(list(obst_coords[:-1, 1]))
            y2s.extend(list(obst_coords[1:, 1]))
        if len(x1s) == 0: # no obstacle around
            return True
        x1s, x2s, y1s, y2s  = np.array(x1s).reshape(1,-1), np.array(x2s).reshape(1,-1),\
            np.array(y1s).reshape(1,-1), np.array(y2s).reshape(1,-1), 
        # Line 2: the edges of obstacles, dx + ey + f = 0
        d = (y2s - y1s).reshape(1,-1) # (1,E)
        e = (x1s - x2s).reshape(1,-1)
        f = (y1s*x2s - x1s*y2s).reshape(1,-1)

        # calculate the intersections
        det = a*e - b*d # (4, E)
        parallel_line_pos = (det==0) # (4, E)
        det[parallel_line_pos] = 1 # temporarily set "1" to avoid "divided by zero"
        raw_x = (b*f - c*e)/det # (4, E)
        raw_y = (c*d - a*f)/det

        collide_map_x = np.ones_like(raw_x, dtype=np.uint8)
        collide_map_y = np.ones_like(raw_x, dtype=np.uint8)
        # the false positive intersections on line L2(not on edge L2)
        collide_map_x[raw_x>np.maximum(x1s, x2s)] = 0
        collide_map_x[raw_x<np.minimum(x1s, x2s)] = 0
        collide_map_y[raw_y>np.maximum(y1s, y2s)] = 0
        collide_map_y[raw_y<np.minimum(y1s, y2s)] = 0
        # the false positive intersections on line L1(not on edge L1)
        collide_map_x[raw_x>np.maximum(vx1s, vx2s)] = 0
        collide_map_x[raw_x<np.minimum(vx1s, vx2s)] = 0
        collide_map_y[raw_y>np.maximum(vy1s, vy2s)] = 0
        collide_map_y[raw_y<np.minimum(vy1s, vy2s)] = 0

        collide_map = collide_map_x*collide_map_y
        collide_map[parallel_line_pos] = 0
        collide = np.sum(collide_map) > 0

        if collide:
            return False
        return True

    def close(self):
        if self.screen is not None:
            pygame.display.quit()
            self.is_open = False
            self.screen = None
            self.clock = None
            pygame.quit()
