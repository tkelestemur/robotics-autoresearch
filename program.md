# Robotics Autoresearch: Agent Instructions

You are an AI researcher optimizing a sampling-based MPC controller to make a Unitree H1 humanoid robot run forward as fast as possible in MuJoCo simulation.

## Setup

- **Editable file**: `mpc.py` — this is the ONLY file you may modify
- **Read-only**: `prepare.py` — simulation infrastructure, evaluation harness, robot model
- **Metric**: `avg_speed_mps` — average forward speed (m/s) over 30 seconds (HIGHER is better)
- **Results log**: `results.tsv` — append every experiment result here

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

- Hyperparameters (horizon, samples, temperature, noise std, cost weights)
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

1. **Cost function tuning**: The baseline cost weights may not be optimal. The speed reward might need to be much higher.
2. **CEM or predictive sampling**: MPPI is one approach; Cross-Entropy Method or simple predictive sampling may work better.
3. **PD controller layer**: Instead of commanding raw torques, command joint position targets through a PD controller.
4. **Correlated noise**: Temporally correlated noise (e.g., via interpolation or filtering) produces smoother trajectories.
5. **Gait symmetry**: Add rewards for symmetric leg motion to encourage natural walking/running gaits.
6. **Longer horizon**: More lookahead may help, but costs more compute. Balance speed vs. quality.
7. **Action parameterization**: Spline-based or CPG-based control can reduce the search space dramatically.
8. **Warm-starting**: Use the previous solution shifted by one step as the initial guess.
9. **Adaptive noise**: Reduce noise std as the solution converges within an episode.
10. **Contact-aware costs**: Use contact information to reward proper foot placement.

## Important Notes

- Each experiment takes 2-5 minutes of wall time. Be patient.
- The robot starts in a standing pose. It needs to learn to walk/run from there.
- Forward speed is measured along the x-axis (qpos[0]).
- A negative speed means the robot went backward — that's bad.
- If the robot falls (height < 0.3m), the cost function penalizes heavily.
- The `mujoco.rollout` module handles parallelism — you don't need to manage threads.
- Always check for crashes before recording results.
