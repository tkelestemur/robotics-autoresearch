# Robotics Autoresearch: Agent Instructions

You are an AI researcher optimizing a sampling-based MPC controller to make a Unitree G1 humanoid robot run forward as fast as possible in MuJoCo simulation.

## Setup

- **Editable file**: `mpc.py` — this is the ONLY file you may modify
- **Read-only**: `prepare.py` — simulation infrastructure, evaluation harness, robot model
- **Metric**: `avg_speed_mps` — average forward speed (m/s) over 30 seconds (HIGHER is better)
- **Results log**: `results.tsv` — append every experiment result here
- **Baseline**: ~1.25 m/s with MPPI + position control (standing pose nominal)

## prepare.py API Reference

Functions you can import:
- `make_sim() -> (model, data)` — loads G1 scene, resets to home keyframe (standing pose). Does NOT set timestep — caller should do `model.opt.timestep = SIM_DT`. Contacts are restricted to foot-floor only (no body/limb collisions).
- `compute_grav_comp(model) -> np.array` — returns (nu,) gravity compensation generalized forces for the home keyframe. For position actuators, the ctrl offset needed is `grav_comp / kp` where kp is the actuator gain.
- `get_home_positions(model) -> np.array` — returns (nu,) home keyframe joint angles. Natural nominal for position control.
- `get_initial_state(model, data) -> np.array` — extracts full physics state. Layout: [time(1), qpos(nq), qvel(nv), act(na)].
- `set_state(model, data, state)` — sets full physics state
- `parallel_rollout(model, initial_state, control_seq, nstep) -> (qpos_traj, qvel_traj)` — rolls out trajectories in parallel using mujoco_warp (GPU-accelerated on CUDA, CPU fallback on macOS). `control_seq` shape: `(nbatch, horizon, nu)`, `nstep` = total physics steps. Returns `qpos_traj (nbatch, horizon, nq)` and `qvel_traj (nbatch, horizon, nv)` at control rate.
- `evaluate_speed(model, data, n_steps=None) -> float` — runs MPC calling `mpc.get_action()` at each step. Imports `CONTROL_DT`, `CONTROL_SUBSTEPS`, `N_CONTROL_STEPS` from mpc at call time. Returns avg forward speed (m/s).

## Simulation Timing Constants (defined in mpc.py — you can change these!)

```python
SIM_DT = 0.002             # physics timestep (500 Hz)
CONTROL_DT = 0.02          # control rate (50 Hz)
CONTROL_SUBSTEPS = int(round(CONTROL_DT / SIM_DT))  # = 10
SIM_DURATION = 30.0        # evaluation window
N_CONTROL_STEPS = int(round(SIM_DURATION / CONTROL_DT))  # = 1500
```

These are imported by `evaluate_speed()` from mpc at call time, so changes take effect immediately. Keep `CONTROL_SUBSTEPS` consistent with `CONTROL_DT / SIM_DT`.

## Robot Details (Unitree G1)

- 36 qpos (7 free joint + 29 actuated joints), 35 qvel, 29 actuators
- **Position-controlled actuators**: `kp=500`, `dampratio=1`. Ctrl inputs are joint angle targets (radians). MuJoCo computes `force = kp*(ctrl - q) - kd*qdot` at every physics substep.
- Actuator `i` controls `qpos[7+i]` and `qvel[6+i]`
- Ctrl ranges match joint ranges (radians)
- Home keyframe: standing pose, legs straight, arms at sides with elbows bent (elbow=1.28 rad, shoulders slightly abducted)
- Standing height: 0.793m (pelvis z-position)
- Total mass: ~35 kg
- Foot contact: 4 sphere geoms per foot, friction=0.6. Only feet can contact the floor — all other body parts have collisions disabled.

Actuator groups:
- Left leg (6): hip_pitch ±88Nm, hip_roll ±139Nm, hip_yaw ±88Nm, knee ±139Nm, ankle_pitch ±50Nm, ankle_roll ±50Nm
- Right leg (6): same as left
- Waist (3): yaw ±88Nm, roll ±50Nm, pitch ±50Nm
- Left arm (7): shoulder_pitch/roll/yaw ±25Nm, elbow ±25Nm, wrist_roll ±25Nm, wrist_pitch/yaw ±5Nm
- Right arm (7): same as left

## Rollout Return Format

`parallel_rollout` returns `(qpos_traj, qvel_traj)` at control rate:
- `qpos_traj`: shape `(nbatch, horizon, nq)` — one snapshot per control step
- `qvel_traj`: shape `(nbatch, horizon, nv)` — one snapshot per control step

Key indices (directly into qpos_traj / qvel_traj):
- `qpos[:, :, 0]`: x-position (forward)
- `qpos[:, :, 2]`: z-position (height)
- `qpos[:, :, 3:7]`: quaternion (w, x, y, z)
- `qpos[:, :, 7:]`: joint angles (29 actuated joints)
- `qvel[:, :, 0]`: forward velocity (vx)
- `qvel[:, :, 6:]`: joint velocities (29 actuated joints)

No time offset — index directly.

## Current mpc.py Design

The baseline uses MPPI (Model Predictive Path Integral) with:
- Home joint angles as the nominal trajectory
- Noise in joint angle space (radians), uniform std across all actuators
- Position control: ctrl = joint angle targets, PD converts to torques
- Key hyperparameters: `HORIZON=10`, `NUM_SAMPLES=1024`, `TEMPERATURE=0.5`, `NOISE_STD=0.15`
- Cost: forward speed reward, height penalty, upright penalty, control effort penalty (deviation from home), fall penalty

The contract: `mpc.py` must export `get_action(model, data) -> np.ndarray` returning a `(nu,)` control vector. It must also export `CONTROL_DT`, `CONTROL_SUBSTEPS`, and `N_CONTROL_STEPS` as module-level constants.

## Experiment Loop

Run this loop forever:

1. **Read** `mpc.py` and `results.tsv` to understand current state
2. **Plan** a modification to improve forward speed (document your hypothesis)
3. **Edit** `mpc.py` with your changes
4. **Screen** with fast eval first (saves time per failed experiment):
   ```
   uv run mpc.py --fast > run.log 2>&1
   ```
   Check for crashes: `cat run.log | grep -E "avg_speed_mps|Error|Traceback"`
   If it crashes, fix and retry. If fast speed looks promising, proceed to full eval.
5. **Full evaluation**:
   ```
   uv run mpc.py > run.log 2>&1
   ```
6. **Check** the output:
   ```
   cat run.log | grep -E "avg_speed_mps|Error|Traceback"
   ```
7. **Record** the result in `results.tsv`:
   - Format: `commit\tavg_speed_mps\twall_time_s\tstatus\tdescription`
   - Status: `keep` (improvement), `discard` (regression), or `crash` (error)
8. **Commit or revert**:
   - If `keep`: `git add mpc.py results.tsv && git commit -m "exp: description"`
   - If `discard` or `crash`: `git checkout mpc.py` (revert changes, still log in results.tsv)
9. **Repeat** from step 1

## Results TSV Format

The first line of `results.tsv` should be the header:
```
commit	avg_speed_mps	wall_time_s	status	description
```

Each subsequent line is one experiment. Use tab separation. Example:
```
a1b2c3d	1.246	84.4	keep	baseline MPPI
none	0.980	92.1	discard	doubled horizon to 20
```

## What You Can Change in mpc.py

- Simulation timing (SIM_DT, CONTROL_DT, CONTROL_SUBSTEPS)
- Hyperparameters (horizon, samples, temperature, noise std, cost weights)
- Cost function design (new reward terms, gait rewards, contact forces)
- Control algorithm (switch from MPPI to CEM, predictive sampling, etc.)
- Action parameterization (CPG-based, spline interpolation, etc.)
- Noise distribution (correlated noise, colored noise, per-actuator scaling, prior-guided sampling)
- Warm-starting and trajectory initialization strategies
- Any other algorithmic changes to improve forward speed

## What You CANNOT Change

- `prepare.py` — do not modify this file
- The robot model (Unitree G1 from mujoco_menagerie)

## Critical: The Robot Must Actually Run

The robot MUST actually walk or run — not exploit the physics simulation. The `avg_speed_mps` metric measures forward displacement of the pelvis over time. A naive optimizer will discover degenerate strategies that maximize this metric without producing real locomotion:

- **Falling forward**: The robot tips over and the pelvis slides along the ground. This is NOT running.
- **Flailing**: The robot throws its limbs to generate forward momentum while on the ground.
- **Body sliding**: Using arm or torso contacts to push forward. Note: body/limb contacts are already disabled, but the robot can still exploit gravity by falling.

Your cost function must ensure the robot stays upright and moves forward through actual stepping — feet pushing off the ground while maintaining balance. If the robot's height drops significantly below standing height (0.793m) while it has forward velocity, that's exploitation, not running.

Think carefully about what distinguishes real bipedal locomotion from exploitation. Real running requires:
1. Maintaining approximate standing height throughout
2. Feet alternately contacting and leaving the ground
3. Dynamic balance — the robot doesn't just fall in a controlled way

Design your cost function to reward these properties and penalize their absence.

## Important Notes

- Use `--fast` to screen for crashes and gross performance before full eval.
- The robot starts in a standing pose. It needs to learn to walk/run from there.
- Forward speed is measured along the x-axis (qpos[0]).
- A negative speed means the robot went backward — that's bad.
- The `compute_grav_comp()` and `get_home_positions()` functions are available from prepare.py — use them.
- Always check for crashes before recording results.
- With GPU (mujoco_warp), you can use NUM_SAMPLES=1024+ for much better trajectory optimization.
- Make ONE change at a time so you can attribute improvements. Don't combine multiple changes in a single experiment.
- Use `uv run mpc.py --benchmark` to measure rollout throughput on your hardware.
- Use `uv run mpc.py --viz` to visually inspect the robot's behavior in the MuJoCo viewer.
