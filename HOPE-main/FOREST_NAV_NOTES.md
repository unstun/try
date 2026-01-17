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

## Hybrid A* + SAC fusion: issues + alternatives
**Current behavior (baseline)**: when the env emits `info['path_to_dest']`, `ParkingAgent` hard-switches to executing the planner's open-loop control sequence until it ends, otherwise SAC acts normally.

**Sanity check before comparing schemes**
- The learning interface expects **normalized actions in `[-1, 1]`**, then `CarParkingWrapper` rescales to env units (`env/env_wrapper.py:action_rescale`). Make sure any planner-provided controls are normalized the same way before stepping the wrapped env.
- If `USE_ACTION_MASK=True`, the mask is over `discrete_actions`; the sampling logic should normalize *both* steer and speed consistently with the wrapper.

### A) Receding-horizon planner (execute 1 step)
- Replan every step (or every N steps), but execute only the *first* planner action (MPC-style). Optional: keep SAC as a residual on top of the planner action.
- Pros: avoids open-loop drift; Cons: higher compute, still planner-dependent.
- Touch points: `src/model/agent/parking_agent.py` (arbitration), `src/env/car_parking_base.py` (planner trigger cadence).

### B) Planner-as-guidance (waypoints, policy stays in control)
- Use Hybrid A* to provide K waypoints / a short local reference curve; feed that as extra observation (or replace `target` with "next waypoint in ego frame") and let SAC output controls.
- Pros: smoother learning + better global context; Cons: needs obs/encoder changes + retraining.
- Touch points: `src/env/car_parking_base.py:_get_targt_repr` (subgoal), `src/model/network.py` (add modality / embed).

### C) Safety shield (policy action with last-moment correction)
- SAC proposes an action; a fast safety check (action mask, 1-step rollout, or occupancy lookup) either (1) projects to nearest safe discrete action, or (2) falls back to a planner-suggested action.
- Pros: preserves policy autonomy + safety; Cons: can introduce discontinuities; safety checker quality matters.
- Touch points: `src/model/action_mask.py` (projection), `src/model/agent/parking_agent.py` (shield logic).

### D) Options / meta-controller (learn when to call planner)
- Treat "planner-follow" as an option; SAC (or a small gating net) decides when to enter/exit it (e.g., stuck detection, low-clearance states, near-goal).
- Pros: planner used only when beneficial; Cons: extra training complexity and tuning.

### E) Planner as teacher (BC/DAgger + SAC fine-tune)
- Use Hybrid A* trajectories to (1) prefill replay buffer, (2) behavior-clone the actor, then fine-tune with SAC. Optional Q-filter: only imitate when critic estimates planner action is better.
- Pros: more sample-efficient; Cons: planner bias; needs dataset curation and failure handling.

### F) Learned heuristic / value-guided Hybrid A*
- Use a learned cost-to-go (critic or separate net) as the Hybrid A* heuristic / expansion bias, then track the resulting path with SAC (or a classical tracker).
- Pros: can improve planning speed/quality; Cons: research-heavy, harder to debug.

**Suggested ablations to report**
- Success rate + collision rate; mean steps-to-goal; mean return; % time under planner/shield; planner wall-clock per episode.

## Train / eval commands (same scripts, now forest by default)
- Train SAC (uses forest sampler via `CarParking` defaults):
  ```bash
  python ./train/train_HOPE_sac.py
  ```
- Eval a checkpoint on mixed scenes (now forest difficulty levels):
  ```bash
  python ./evaluation/eval_mix_scene.py ./model/ckpt/HOPE_SAC0.pt --eval_episode 10 --visualize True
  ```
