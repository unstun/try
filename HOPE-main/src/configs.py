"""
Global configuration for the HOPE parking environment, agents, and evaluation
pipeline. Values here are consumed by:
- env/*.py for geometry, rendering, sensors, rewards, and action masks
- model/* for network shapes and training hyperparameters
- evaluation/* for logging, saving frames, and map difficulty settings
"""

import os
# Optional environment overrides (uncomment to force headless or GPU pinning)
# os.environ["SDL_VIDEODRIVER"]="dummy"
# os.environ["CUDA_VISIBLE_DEVICES"] = '3'

import numpy as np
import torch

# --------------------------------------------------------------------------- #
# Reproducibility / device
# --------------------------------------------------------------------------- #
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
SEED = 42  # used by eval/train scripts to seed env, action space, numpy, torch

# --------------------------------------------------------------------------- #
# Vehicle geometry and dynamic limits (env/vehicle.py)
# --------------------------------------------------------------------------- #
# Real platform (924mm L x 740mm W x 350mm H, 0.6m wheelbase/track)
WHEEL_BASE = 0.6           # axle-to-axle distance; drives turning radius and RS planner
FRONT_HANG = 0.162         # front overhang (axle to bumper), meters
REAR_HANG = 0.162          # rear overhang (axle to bumper), meters
LENGTH = WHEEL_BASE + FRONT_HANG + REAR_HANG  # total bumper-to-bumper length
WIDTH = 0.74               # vehicle width used for collisions and lot sizing
TRACK_WIDTH = 0.6          # wheel track width center-to-center

from shapely.geometry import LinearRing
VehicleBox = LinearRing([
    (-REAR_HANG, -WIDTH / 2),
    (FRONT_HANG + WHEEL_BASE, -WIDTH / 2),
    (FRONT_HANG + WHEEL_BASE,  WIDTH / 2),
    (-REAR_HANG,  WIDTH / 2)])

COLOR_POOL = [
    (30, 144, 255, 255),  # dodger blue
    (255, 127, 80, 255),  # coral
    (255, 215, 0, 255)    # gold
]

# Action/kinematics bounds enforced in Vehicle.step and rsCurve planner
VALID_SPEED = [-2.6944, 2.6944]          # m/s, from 9.7 km/h top speed (symmetric fwd/rev)
VALID_STEER = [-0.4712389, 0.4712389]    # radians, ±27 deg steering angle clamp
VALID_ACCEL = [-1.5, 1.5]                # m/s^2, accel/decel from platform
VALID_ANGULAR_SPEED = [-1.22348, 1.22348] # rad/s, ±70.1 deg/s yaw-rate clamp

# Internal sim sub-steps per env step (controls smoothness of motion)
NUM_STEP = 10
STEP_LENGTH = 5e-2  # seconds per internal sub-step

# --------------------------------------------------------------------------- #
# Scenario generation (env/forest_map.py)
# --------------------------------------------------------------------------- #
MAP_LEVEL = 'Normal'  # default difficulty: ['Normal', 'Complex', 'Extrem', 'dlp']
FOREST_GRID_RES = 0.25  # meters per occupancy cell
FOREST_POLY_RESOLUTION = 10  # polygon resolution when approximating circles
FOREST_MAP_SIZE = {  # (width, height) meters
    'Normal': (32.0, 32.0),
    'Complex': (40.0, 40.0),
    'Extrem': (48.0, 48.0),
    'dlp': (36.0, 36.0),
}
FOREST_OBS_DENSITY = {  # approximate obstacles per square meter
    'Normal': 0.025,
    'Complex': 0.04,
    'Extrem': 0.055,
    'dlp': 0.03,
}
FOREST_OBS_RADIUS_RANGE = {  # meters
    'Normal': (0.25, 0.45),
    'Complex': (0.25, 0.55),
    'Extrem': (0.30, 0.65),
    'dlp': (0.25, 0.50),
}
FOREST_MIN_CORRIDOR = {  # minimum desired free corridor width between obstacles
    'Normal': 1.5,
    'Complex': 1.2,
    'Extrem': 1.0,
    'dlp': 1.3,
}
FOREST_VEHICLE_CLEARANCE = max(WIDTH, LENGTH) / 2 + 0.15
FOREST_BOUNDARY_MARGIN = 0.5  # padding to keep obstacles away from the map edge
FOREST_SPAWN_MARGIN = 1.5  # clearance to keep start/goal away from edges/obstacles
FOREST_MAX_SAMPLE_RETRY = 200  # attempts when sampling obstacles/start/goal
FOREST_MIN_GOAL_SEPARATION = {
    'Normal': 10.0,
    'Complex': 12.0,
    'Extrem': 14.0,
    'dlp': 11.0,
}
FOREST_MAX_GOAL_SEPARATION = {
    'Normal': 22.0,
    'Complex': 26.0,
    'Extrem': 30.0,
    'dlp': 24.0,
}

# Legacy parking map parameters are kept for compatibility with unused parking generators.
MIN_PARK_LOT_LEN_DICT = {'Extrem': LENGTH + 0.6,
                            'Complex': LENGTH + 0.9,
                            'Normal': LENGTH * 1.25,}
MAX_PARK_LOT_LEN_DICT = {'Extrem': LENGTH + 0.9,
                            'Complex': LENGTH * 1.25,
                            'Normal': LENGTH * 1.25 + 0.5}
MIN_PARK_LOT_WIDTH_DICT = {
    'Complex': WIDTH + 0.4,
    'Normal': WIDTH + 0.85,
}
MAX_PARK_LOT_WIDTH_DICT = {
    'Complex': WIDTH + 0.85,
    'Normal': WIDTH + 1.2,
}
PARA_PARK_WALL_DIST_DICT = {
    'Extrem': 3.5,
    'Complex': 4.0,
    'Normal': 4.5,
}
BAY_PARK_WALL_DIST_DICT = {
    'Complex': 6.0,
    'Normal': 7.0,
}
N_OBSTACLE_DICT = {
    'Extrem': 8,
    'Complex': 5,
    'Normal': 3,
}
MIN_DIST_TO_OBST = 0.1
MAX_DRIVE_DISTANCE = 15.0
DROUP_OUT_OBST = 0.0

# --------------------------------------------------------------------------- #
# Environment rendering, observations, and timing (env/car_parking_base.py)
# --------------------------------------------------------------------------- #
ENV_COLLIDE = True  # if True, collision immediately ends; False allows retreat logic

# Colors used throughout pygame rendering and Obs_Processor masking
BG_COLOR = (255, 255, 255, 255)
START_COLOR = (0, 100, 0, 255)      # dark green start box (acceleration cue)
DEST_COLOR = (139, 0, 0, 255)       # dark red goal box
START_BORDER_WIDTH = 4              # outline thickness for start box
DEST_BORDER_WIDTH = 4               # outline thickness for goal box
OBSTACLE_COLOR = (150, 150, 150, 255)
TRAJ_COLOR_HIGH = (10, 10, 200, 255)
TRAJ_COLOR_LOW = (10, 10, 10, 255)
TRAJ_LINE_COLOR = (255, 215, 0, 255)  # smooth curve color
TRAJ_LINE_WIDTH = 3
RENDER_TRAJ_LINE = True
ARROW_COLOR_START = (255, 215, 0, 255)
ARROW_COLOR_END = (0, 191, 255, 255)
ARROW_LENGTH = 20          # screen pixels
ARROW_HEAD_SCALE = 0.45
TRAJ_RENDER_LEN = 1000       # number of historical vehicle footprints to render
TRAJ_COLORS = list(map(tuple, np.linspace(
    np.array(TRAJ_COLOR_LOW), np.array(TRAJ_COLOR_HIGH), TRAJ_RENDER_LEN, endpoint=True, dtype=np.uint8)))

# Image/capture resolutions: pygame renders to WIN_*, ego crop OBS_*, optional video uses VIDEO_*
OBS_W = 256   # cropped ego-centric view width (px) before downsample in Obs_Processor
OBS_H = 256   # cropped ego-centric view height (px) before downsample in Obs_Processor
VIDEO_W = 600 # export width for eval videos/frames
VIDEO_H = 400 # export height for eval videos/frames
WIN_W = 2000  # pygame window width (px) for full scene render
WIN_H = 2000  # pygame window height (px) for full scene render

# Sensors and temporal limits
LIDAR_RANGE = 10.0   # meters
LIDAR_NUM = 120      # number of beams over 360 deg
FPS = 100            # target render/update rate for pygame loop
TOLERANT_TIME = 200  # max env steps before OUTTIME
USE_LIDAR = True     # include lidar observation channel
USE_IMG = True       # include image observation channel
USE_ACTION_MASK = True  # include discrete action mask channel
MAX_DIST_TO_DEST = 40   # cap for normalized target distance feature
GOAL_TOLERANCE = 0.5    # success radius around goal (meters)
GOAL_HEADING_TOL = None # optional heading tolerance (radians), set None to ignore
K = 24  # render scale factor (px per world unit) used in world->screen transform
RS_MAX_DIST = 10  # max distance to try RS planner guidance
RENDER_TRAJ = True  # render vehicle footprints

# --------------------------------------------------------------------------- #
# Discrete action mask (model/action_mask.py uses these)
# --------------------------------------------------------------------------- #
PRECISION = 10  # steering discretization steps across [-max, max]
step_speed = round(VALID_SPEED[1] * 0.4, 3)  # base speed magnitude for discrete action grid (forward/backward)
discrete_actions = []
for i in np.arange(VALID_STEER[-1], -(VALID_STEER[-1] + VALID_STEER[-1] / PRECISION), -VALID_STEER[-1] / PRECISION):
    discrete_actions.append([i, step_speed])
for i in np.arange(VALID_STEER[-1], -(VALID_STEER[-1] + VALID_STEER[-1] / PRECISION), -VALID_STEER[-1] / PRECISION):
    discrete_actions.append([i, -step_speed])
N_DISCRETE_ACTION = len(discrete_actions)

# --------------------------------------------------------------------------- #
# Navigation planner defaults (env/hybrid_astar.py)
# --------------------------------------------------------------------------- #
HYBRID_MAX_PLAN_DIST = 0.5  # meters within which to trigger fallback planner
HYBRID_YAW_BINS = 48         # heading discretization for hybrid A*
HYBRID_STEER_SET = np.linspace(VALID_STEER[0], VALID_STEER[1], 5)
HYBRID_SPEED_SET = [1.5, -1.0]  # forward and reverse speeds used by planner
HYBRID_STEP_TIME = 5         # env sub-steps per planner primitive
HYBRID_SIM_STEPS = 5         # internal integration slices per primitive
HYBRID_DT = STEP_LENGTH      # integration timestep
HYBRID_MAX_NODES = 4000      # search budget
HYBRID_HEADING_WEIGHT = 30  # weight on heading error in heuristic

# --------------------------------------------------------------------------- #
# RL model/training hyperparameters (model/agent/*.py)
# --------------------------------------------------------------------------- #
GAMMA = 0.98           # discount factor
BATCH_SIZE = 8192      # replay/sample batch size
LR = 5e-6              # learning rate for actor/critic optimizers
TAU = 0.1              # soft target update coefficient
MAX_TRAIN_STEP = 1e6   # training horizon used in scripts
ORTHOGONAL_INIT = True # initialize linear/convolutional layers orthogonally
LR_DECAY = False       # toggle scheduler in agents
UPDATE_IMG_ENCODE = False  # if True, finetune image encoder; otherwise freeze

# Shared image encoder shapes for actor/critic
C_CONV = [4, 8]    # conv output channels per layer
SIZE_FC = [256]    # linear layer widths after conv flatten

# Optional attention block applied to image embeddings
ATTENTION_CONFIG = {
    'depth': 1,
    'heads': 8,
    'dim_head': 32,
    'mlp_dim': 128,
    'hidden_dim': 128,
}
USE_ATTENTION = True

# Actor network config (model/agent/ppo_agent.py, sac_agent.py)
ACTOR_CONFIGS = {
    'n_modal': 2 + int(USE_IMG) + int(USE_ACTION_MASK),  # lidar + target + optional img/mask
    'lidar_shape': LIDAR_NUM,
    'target_shape': 5,  # distance + cos/sin target/heading
    'action_mask_shape': N_DISCRETE_ACTION if USE_ACTION_MASK else None,
    'img_shape': (3, 64, 64) if USE_IMG else None,  # after Obs_Processor downsample
    'output_size': 2,  # steer, speed
    'embed_size': 128,
    'hidden_size': 256,
    'n_hidden_layers': 3,
    'n_embed_layers': 2,
    'img_conv_layers': C_CONV,
    'img_linear_layers': SIZE_FC,
    'k_img_conv': 3,
    'orthogonal_init': True,
    'use_tanh_output': True,
    'use_tanh_activate': True,
    'attention_configs': ATTENTION_CONFIG if USE_ATTENTION else None,
}

# Critic network config mirrors actor except output size
CRITIC_CONFIGS = {
    'n_modal': 2 + int(USE_IMG) + int(USE_ACTION_MASK),
    'lidar_shape': LIDAR_NUM,
    'target_shape': 5,
    'action_mask_shape': N_DISCRETE_ACTION if USE_ACTION_MASK else None,
    'img_shape': (3, 64, 64) if USE_IMG else None,
    'output_size': 1,
    'embed_size': 128,
    'hidden_size': 256,
    'n_hidden_layers': 3,
    'n_embed_layers': 2,
    'img_conv_layers': C_CONV,
    'img_linear_layers': SIZE_FC,
    'k_img_conv': 3,
    'orthogonal_init': True,
    'use_tanh_output': False,
    'use_tanh_activate': True,
    'attention_configs': ATTENTION_CONFIG if USE_ATTENTION else None,
}

# Reward shaping (env/env_wrapper.py uses these to scale components)
REWARD_RATIO = 1.0  # overall scalar applied to shaped reward
from typing import OrderedDict
REWARD_WEIGHT = OrderedDict({
    'progress': 1.0,     # positive when moving toward the goal
    'collision': 1.0,    # large penalty applied on collision/out-of-bounds/timeout
    'smoothness': 0.1,   # small penalty on abrupt control changes
})
COLLISION_PENALTY = 25.0
OUTBOUND_PENALTY = 25.0
TIMEOUT_PENALTY = 10.0
PROGRESS_CLIP = 2.0
SMOOTHNESS_STEER_WEIGHT = 0.5
SMOOTHNESS_SPEED_WEIGHT = 0.1

# Auxiliary discrete action head config (used by planner/aux models)
CONFIGS_ACTION = {
    'use_tanh_activate': True,
    'hidden_size': 256,
    'lidar_shape': LIDAR_NUM,
    'n_hidden_layers': 4,
    'n_action': len(discrete_actions),
    'discrete_actions': discrete_actions
}
