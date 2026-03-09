# Robotics Autoresearch

Read `program.md` for full instructions. This is an autoresearch loop where you iteratively improve `mpc.py` to maximize forward speed of a Unitree G1 humanoid robot in MuJoCo.

## Quick Reference

- **Edit only**: `mpc.py`
- **Do not edit**: `prepare.py`
- **Screen**: `uv run mpc.py --fast > run.log 2>&1` (5s eval, ~14s wall time)
- **Full eval**: `uv run mpc.py > run.log 2>&1` (30s eval, ~85s wall time)
- **Check**: `cat run.log | grep -E "avg_speed_mps|Error|Traceback"`
- **Log results**: append to `results.tsv`
- **Metric**: `avg_speed_mps` (higher is better)
- **Visualize**: `uv run mpc.py --viz` (opens MuJoCo viewer, not for automated runs)

## Available utilities from prepare.py

- `compute_grav_comp(model)` — gravity compensation generalized forces (nu,)
- `get_home_positions(model)` — home keyframe joint angles (nu,)
- `evaluate_speed(model, data, n_steps)` — imports timing constants from mpc.py
- `parallel_rollout(model, initial_state, control_seq, nstep)` — returns `(qpos_traj, qvel_traj)` at control rate. GPU (mujoco_warp) if available, CPU fallback otherwise.

## Tunable timing constants (in mpc.py)

- `SIM_DT`, `CONTROL_DT`, `CONTROL_SUBSTEPS` — simulation timing
- These are imported by prepare.py's `evaluate_speed()` at call time
