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
- `make_sim() -> (model, data)` — loads G1 scene, resets to home keyframe (standing pose). Does NOT set timestep — caller should do `model.opt.timestep = SIM_DT`.
- `compute_grav_comp(model) -> np.array` — returns (nu,) gravity compensation generalized forces for the home keyframe. For position actuators, the ctrl offset needed is `grav_comp / kp` where kp is the actuator gain.
- `get_home_positions(model) -> np.array` — returns (nu,) home keyframe joint angles. Natural nominal for position control.
- `get_initial_state(model, data) -> np.array` — extracts full physics state
- `set_state(model, data, state)` — sets full physics state
- `make_data_list(model, nthread) -> list[MjData]` — creates per-thread MjData for parallel rollouts
- `parallel_rollout(model, data_list, initial_state, control_seq, nstep) -> state_traj` — rolls out trajectories in parallel. `control_seq` shape: `(nbatch, horizon, nu)`, `nstep` = total physics steps (control_substeps inferred as `nstep // horizon`). Returns `(nbatch, nstep, nstate)`.
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
- Foot contact: 4 sphere geoms per foot, friction=0.6

Actuator groups:
- Left leg (6): hip_pitch ±88Nm, hip_roll ±139Nm, hip_yaw ±88Nm, knee ±139Nm, ankle_pitch ±50Nm, ankle_roll ±50Nm
- Right leg (6): same as left
- Waist (3): yaw ±88Nm, roll ±50Nm, pitch ±50Nm
- Left arm (7): shoulder_pitch/roll/yaw ±25Nm, elbow ±25Nm, wrist_roll ±25Nm, wrist_pitch/yaw ±5Nm
- Right arm (7): same as left

## State Layout (CRITICAL)

The state from `parallel_rollout` and `get_initial_state` uses `mjSTATE_FULLPHYSICS` which includes time:

```
[time(1), qpos(36), qvel(35)] = 72 floats
```

**When indexing state trajectories in the cost function:**
```python
qpos = state_traj[:, :, 1:1+nq]        # skip time at index 0
qvel = state_traj[:, :, 1+nq:1+nq+nv]  # after qpos
```

Key state indices (within qpos, after skipping time):
- `qpos[0]`: x-position (forward)
- `qpos[1]`: y-position (lateral)
- `qpos[2]`: z-position (height)
- `qpos[3:7]`: quaternion (w, x, y, z)
- `qpos[7:]`: joint angles (29 actuated joints)
- `qvel[0]`: forward velocity (vx)
- `qvel[1]`: lateral velocity (vy)
- `qvel[2]`: vertical velocity (vz)
- `qvel[3:6]`: angular velocity
- `qvel[6:]`: joint velocities (29 actuated joints)

## Current mpc.py Design

The baseline uses MPPI (Model Predictive Path Integral) with:
- Home joint angles as the nominal trajectory
- Noise in joint angle space (radians), uniform std across all actuators
- Position control: ctrl = joint angle targets, PD converts to torques
- Key hyperparameters: `HORIZON=10`, `NUM_SAMPLES=128`, `TEMPERATURE=0.5`, `NOISE_STD=0.15`
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

## Hints for Improvement

1. **Cost function tuning**: The baseline cost weights may not be optimal. The speed reward might need to be much higher relative to other terms.
2. **CEM or predictive sampling**: MPPI is one approach; Cross-Entropy Method or simple predictive sampling may work better.
3. **Correlated noise**: Temporally correlated noise (e.g., via interpolation or filtering) produces smoother, more physically plausible trajectories.
4. **Gait symmetry**: Add rewards for symmetric leg motion to encourage natural walking/running gaits.
5. **Longer horizon**: More lookahead may help, but costs more compute. Balance speed vs. quality.
6. **Action parameterization**: Spline-based or CPG-based control can reduce the search space dramatically.
7. **Warm-starting**: The baseline already shifts the previous solution by one step. Consider more sophisticated warm-starting.
8. **Adaptive noise**: Reduce noise scale as the solution converges within an episode.
9. **Contact-aware costs**: Use contact information to reward proper foot placement.
10. **Per-actuator noise scaling**: Legs might benefit from different noise scales than arms/waist.
11. **Simulation timing**: Faster control rate (smaller CONTROL_DT) gives finer control but more compute. Coarser physics (larger SIM_DT) is faster but less accurate.

## Important Notes

- Full evaluation takes ~85s wall time. Fast eval (`--fast`) takes ~14s.
- Use `--fast` to screen for crashes and gross performance before full eval.
- The robot starts in a standing pose. It needs to learn to walk/run from there.
- Forward speed is measured along the x-axis (qpos[0]).
- A negative speed means the robot went backward — that's bad.
- If the robot falls (height < 0.3m), the cost function penalizes heavily.
- The `mujoco.rollout` module handles parallelism — you don't need to manage threads.
- The `compute_grav_comp()` and `get_home_positions()` functions are available from prepare.py — use them.
- Always check for crashes before recording results.
- Make ONE change at a time so you can attribute improvements. Don't combine multiple changes in a single experiment.
