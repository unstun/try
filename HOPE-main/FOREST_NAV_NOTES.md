# Forest Navigation Migration Notes

This repo now uses a forest-style occupancy grid instead of the parking-lot sampler while keeping observations/actions stable. Key points and quick commands are below.

## What changed
- **Map sampler**: `env/forest_map.py` replaces parking map generation. It samples circular/polygonal obstacles, builds a binary occupancy grid, and guarantees start/goal connectivity. Exposes the same fields (`start`, `dest`, `start_box`, `dest_box`, `obstacles`, `xmin/xmax/ymin/ymax`) consumed by sensors/rendering.
- **Termination**: Success is reaching the goal within `GOAL_TOLERANCE` (meters) in `configs.py`; optional heading tolerance via `GOAL_HEADING_TOL`.
- **Rewards**: Navigation-shaped rewards: `progress` (distance reduction), `collision` (large negative on collision/outbound/timeout), `smoothness` (penalizes action deltas). Scaled by `REWARD_WEIGHT` and `REWARD_RATIO` in `configs.py`.
- **Fallback planner**: `env/hybrid_astar.py` provides Hybrid A* (Ackermann) used via `info['path_to_dest']`. `model/agent/parking_agent.py` keeps the planner wrapper name but now consumes Hybrid A* controls.
- **Configs**: Forest parameters (grid resolution, obstacle density/size, map size, goal separation) plus planner/reward settings live in `configs.py`. Legacy parking params remain for compatibility but are unused by default.

## Files to skim
- `src/env/forest_map.py`: occupancy-grid sampling + connectivity check.
- `src/env/car_parking_base.py`: uses forest map, new rewards/termination, Hybrid A* hook.
- `src/env/hybrid_astar.py`: planner implementation.
- `src/env/env_wrapper.py`: reward aggregation matching new reward keys.
- `src/model/agent/parking_agent.py`: planner controller now accepts Hybrid A* paths.
- `src/configs.py`: forest/planner/reward knobs.

## Quick sanity checks
From `HOPE-main/`:
1) Python byte-compile (already clean locally):
   ```bash
   python -m compileall -q src
   ```
2) Smoke-test env reset/step without render (headless):
   ```bash
   python - <<'PY'
   from env.car_parking_base import CarParking
   env = CarParking(render_mode='rgb_array', use_img_observation=False, use_action_mask=False)
   obs = env.reset()
   for _ in range(10):
       obs, rew_info, status, info = env.step(env.action_space.sample())
   print("done:", status, "path_to_dest:", info.get("path_to_dest") is not None)
   PY
   ```
3) (Optional) Visual check with pygame window:
   ```bash
   python - <<'PY'
   from env.car_parking_base import CarParking
   env = CarParking()
   env.reset()
   for _ in range(200):
       env.step(env.action_space.sample())
   PY
   ```

## How to tune
- Goal success radius: `GOAL_TOLERANCE` in `configs.py`.
- Reward weights/penalties: `REWARD_WEIGHT`, `COLLISION_PENALTY`, `SMOOTHNESS_*` in `configs.py`.
- Forest difficulty: obstacle density/size/corridor in `FOREST_*` constants.
- Planner budget: `HYBRID_MAX_NODES`, `HYBRID_YAW_BINS`, steer/speed sets in `configs.py`.

## Train / eval commands (same scripts, now forest by default)
- Train SAC (uses forest sampler via `CarParking` defaults):
  ```bash
  python ./train/train_HOPE_sac.py
  ```
- Eval a checkpoint on mixed scenes (now forest difficulty levels):
  ```bash
  python ./evaluation/eval_mix_scene.py ./model/ckpt/HOPE_SAC0.pt --eval_episode 10 --visualize True
  ```
