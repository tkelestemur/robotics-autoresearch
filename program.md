# Robotics Autoresearch: Agent Instructions

You are an AI researcher optimizing a sampling-based MPC controller to make a Unitree H1 humanoid robot run forward as fast as possible in MuJoCo simulation.

## Setup

- **Editable file**: `mpc.py` — this is the ONLY file you may modify
- **Read-only**: `prepare.py` — simulation infrastructure, evaluation harness, robot model
- **Metric**: `avg_speed_mps` — average forward speed (m/s) over 30 seconds (HIGHER is better)
- **Results log**: `results.tsv` — append every experiment result here
- **Baseline**: -0.030 m/s with gravity-compensation MPPI (robot barely moves; lots of room to improve)

## prepare.py API Reference

Constants you can import:
- `SIM_DURATION = 30.0` — evaluation window (seconds)
- `SIM_DT = 0.002` — physics timestep (500 Hz)
- `CONTROL_DT = 0.02` — control rate (50 Hz)
- `CONTROL_SUBSTEPS = 10` — physics steps per control step
- `N_CONTROL_STEPS = 1500` — total control steps per evaluation

Functions you can import:
- `make_sim() -> (model, data)` — loads H1 scene, resets to home keyframe (standing pose)
- `get_initial_state(model, data) -> np.array` — extracts full physics state (qpos + qvel + act = 52 floats)
- `set_state(model, data, state)` — sets full physics state
- `make_data_list(model, nthread) -> list[MjData]` — creates per-thread MjData for parallel rollouts
- `parallel_rollout(model, data_list, initial_state, control_seq, nstep) -> state_traj` — rolls out trajectories in parallel. `control_seq` shape: `(nbatch, horizon, nu)`, each control held for `CONTROL_SUBSTEPS` physics steps. Returns `(nbatch, nstep, nstate)`.
- `evaluate_speed(model, data) -> float` — runs MPC for 30s calling `mpc.get_action()` at each step, returns avg forward speed (m/s)

Robot details (Unitree H1):
- 26 qpos (7 free joint + 19 actuated joints), 25 qvel, 19 actuators
- All actuators are direct torque control (gain=1, no bias)
- Actuator ranges: legs ±200-300 Nm, ankles ±40 Nm, arms ±18-40 Nm
- Actuator `i` controls `qpos[7+i]` and `qvel[6+i]`
- Home keyframe: standing with bent knees (hip_pitch=-0.4, knee=0.8, ankle=-0.4)
- State layout in rollout output: `[qpos(26), qvel(25), act(1)]` = 52 floats

## Current mpc.py Design

The baseline uses MPPI (Model Predictive Path Integral) with:
- Gravity compensation torques as the nominal trajectory (computed via `mj_inverse`)
- Per-actuator noise scaling (`NOISE_SCALE * actuator_range`)
- Horizon shifted with gravity comp fill (not zeros)
- Key hyperparameters: `HORIZON=10`, `NUM_SAMPLES=128`, `TEMPERATURE=0.5`, `NOISE_SCALE=0.1`

The contract: `mpc.py` must export `get_action(model, data) -> np.ndarray` returning a `(nu,)` torque vector.

## Experiment Loop

Run this loop forever:

1. **Read** `mpc.py` and `results.tsv` to understand current state
2. **Plan** a modification to improve forward speed (document your hypothesis)
3. **Edit** `mpc.py` with your changes
4. **Run** the experiment:
   ```
   uv run mpc.py > run.log 2>&1
   ```
5. **Check** the output:
   ```
   cat run.log | grep -E "avg_speed_mps|Error|Traceback"
   ```
6. **Record** the result in `results.tsv`:
   - Format: `commit\tavg_speed_mps\twall_time_s\tstatus\tdescription`
   - Status: `keep` (improvement), `discard` (regression), or `crash` (error)
7. **Commit or revert**:
   - If `keep`: `git add mpc.py results.tsv && git commit -m "exp: description"`
   - If `discard` or `crash`: `git checkout mpc.py` (revert changes, still log in results.tsv)
8. **Repeat** from step 1

## Results TSV Format

The first line of `results.tsv` should be the header:
```
commit	avg_speed_mps	wall_time_s	status	description
```

Each subsequent line is one experiment. Use tab separation. Example:
```
a1b2c3d	0.234	187.3	keep	baseline MPPI
none	-0.012	192.1	discard	doubled horizon to 20
```

## What You Can Change in mpc.py

- Hyperparameters (horizon, samples, temperature, noise scale, cost weights)
- Cost function design (new reward terms, gait rewards, contact forces)
- Control algorithm (switch from MPPI to CEM, predictive sampling, etc.)
- Action parameterization (PD targets, CPG-based, spline interpolation)
- Noise distribution (correlated noise, colored noise, prior-guided sampling)
- Warm-starting and trajectory initialization strategies
- Any other algorithmic changes to improve forward speed

## What You CANNOT Change

- `prepare.py` — do not modify this file
- The evaluation protocol (30s simulation, 50 Hz control, etc.)
- The robot model (Unitree H1 from mujoco_menagerie)

## Hints for Improvement

1. **Cost function tuning**: The baseline cost weights may not be optimal. The speed reward might need to be much higher relative to other terms.
2. **CEM or predictive sampling**: MPPI is one approach; Cross-Entropy Method or simple predictive sampling may work better.
3. **PD controller layer**: Instead of commanding raw torques, command joint position targets through a PD controller. Note: `parallel_rollout` takes raw torques, so PD conversion must happen before the rollout call. One approach is PD at execution time only; another is to approximate PD within the planning horizon.
4. **Correlated noise**: Temporally correlated noise (e.g., via interpolation or filtering) produces smoother, more physically plausible trajectories.
5. **Gait symmetry**: Add rewards for symmetric leg motion to encourage natural walking/running gaits.
6. **Longer horizon**: More lookahead may help, but costs more compute. Balance speed vs. quality.
7. **Action parameterization**: Spline-based or CPG-based control can reduce the search space dramatically.
8. **Warm-starting**: The baseline already shifts the previous solution by one step. Consider more sophisticated warm-starting (e.g., using multiple restarts).
9. **Adaptive noise**: Reduce noise scale as the solution converges within an episode.
10. **Contact-aware costs**: Use contact information to reward proper foot placement.

## Important Notes

- Each experiment takes ~35-40 seconds of wall time on this machine.
- The robot starts in a standing pose. It needs to learn to walk/run from there.
- Forward speed is measured along the x-axis (qpos[0]).
- A negative speed means the robot went backward — that's bad.
- If the robot falls (height < 0.3m), the cost function penalizes heavily.
- The `mujoco.rollout` module handles parallelism — you don't need to manage threads.
- Gravity compensation alone doesn't stabilize the robot — it still falls (z: 0.98 -> 0.42 in 1s). Active control is needed.
- Always check for crashes before recording results.
