"""MPPI-based Model Predictive Controller for Unitree G1 locomotion.

This is the ONLY file the agent may modify. The goal is to maximize
forward speed (m/s) of the G1 humanoid robot over 30 seconds.

The G1 uses POSITION CONTROL actuators: ctrl inputs are joint angle
targets (radians). MuJoCo's built-in PD (kp=500, dampratio=1) converts
these to torques at every physics substep.
"""

import os
import time

import mujoco
import numpy as np

from prepare import (
    compute_grav_comp,
    get_home_positions,
    make_data_list,
    make_sim,
    get_initial_state,
    parallel_rollout,
)

# ── Simulation timing (tune these!) ───────────────────────────────────────────

SIM_DT = 0.002             # physics timestep (500 Hz)
CONTROL_DT = 0.02          # control rate (50 Hz)
CONTROL_SUBSTEPS = int(round(CONTROL_DT / SIM_DT))  # physics steps per control step
SIM_DURATION = 30.0        # seconds of simulation
N_CONTROL_STEPS = int(round(SIM_DURATION / CONTROL_DT))  # total control steps

# ── MPPI Hyperparameters (tune these!) ────────────────────────────────────────

HORIZON = 10          # control steps lookahead (0.2s at 50 Hz)
NUM_SAMPLES = 128     # number of parallel trajectory samples
TEMPERATURE = 0.5     # MPPI inverse temperature (lower = more greedy)
NOISE_STD = 0.15      # std of joint angle perturbations (radians)

# Cost weights
SPEED_WEIGHT = 5.0    # reward for forward velocity
HEIGHT_WEIGHT = 2.0   # penalty for deviating from target height
UPRIGHT_WEIGHT = 1.0  # penalty for not being upright
CTRL_WEIGHT = 0.1     # penalty for deviation from home position

# Target values
TARGET_HEIGHT = 0.793  # desired torso height (m) — G1 standing height
FALL_HEIGHT = 0.3      # below this = fallen

# Parallelism
NTHREAD = os.cpu_count() or 4

# Deterministic seed for reproducibility
SEED = 42


# ── Cost function ──────────────────────────────────────────────────────────────

def compute_trajectory_costs(
    state_traj: np.ndarray,
    control_seq: np.ndarray,
    nq: int,
    nv: int,
    home_pos: np.ndarray,
) -> np.ndarray:
    """Compute costs for a batch of trajectories.

    Args:
        state_traj: (nbatch, nstep, nstate) full physics states.
        control_seq: (nbatch, horizon, nu) joint angle target sequences.
        nq: number of generalized positions.
        nv: number of generalized velocities.
        home_pos: (nu,) home joint positions.

    Returns:
        costs: (nbatch,) total cost per trajectory.
    """
    nstep = state_traj.shape[1]

    # State layout: [time(1), qpos(nq), qvel(nv), act(na)]
    qpos = state_traj[:, :, 1:1 + nq]
    qvel = state_traj[:, :, 1 + nq:1 + nq + nv]

    # Sample at control rate (end of each control interval)
    ctrl_indices = np.arange(CONTROL_SUBSTEPS - 1, nstep, CONTROL_SUBSTEPS)
    qpos_ctrl = qpos[:, ctrl_indices, :]
    qvel_ctrl = qvel[:, ctrl_indices, :]

    # Forward velocity reward
    vx = qvel_ctrl[:, :, 0]
    speed_cost = -SPEED_WEIGHT * vx

    # Height penalty
    z = qpos_ctrl[:, :, 2]
    height_cost = HEIGHT_WEIGHT * (z - TARGET_HEIGHT) ** 2

    # Upright penalty (quaternion w component — 1.0 means perfectly upright)
    qw = qpos_ctrl[:, :, 3]
    upright_cost = UPRIGHT_WEIGHT * (1.0 - qw ** 2)

    # Control effort: penalize deviation from home position
    target_dev = control_seq - home_pos
    ctrl_cost = CTRL_WEIGHT * np.sum(target_dev ** 2, axis=-1)

    step_cost = speed_cost + height_cost + upright_cost + ctrl_cost

    # Fall penalty
    min_height = np.min(z, axis=1)
    fall_penalty = np.where(min_height < FALL_HEIGHT, 1000.0, 0.0)

    return np.sum(step_cost, axis=1) + fall_penalty


# ── MPPI Controller ────────────────────────────────────────────────────────────

_nominal: np.ndarray | None = None
_home_pos: np.ndarray | None = None
_data_list: list | None = None
_rng: np.random.Generator | None = None


def get_action(model: mujoco.MjModel, data: mujoco.MjData) -> np.ndarray:
    """MPPI controller: called at each control step by evaluate_speed().

    Returns joint angle targets (nu,) to apply for the next control step.
    MuJoCo's built-in PD converts these to torques at every physics substep.
    """
    global _nominal, _home_pos, _data_list, _rng

    nu = model.nu
    nq = model.nq
    nv = model.nv
    ctrl_range = model.actuator_ctrlrange
    lo = ctrl_range[:, 0]
    hi = ctrl_range[:, 1]

    # Initialize on first call
    if _rng is None:
        _rng = np.random.default_rng(SEED)
    if _home_pos is None:
        _home_pos = get_home_positions(model)
    if _nominal is None or _nominal.shape != (HORIZON, nu):
        _nominal = np.tile(_home_pos, (HORIZON, 1))
    if _data_list is None:
        _data_list = make_data_list(model, NTHREAD)

    current_state = get_initial_state(model, data)

    # Sample noise in joint angle space (radians)
    noise = _rng.standard_normal((NUM_SAMPLES, HORIZON, nu)) * NOISE_STD

    # Candidate joint angle target sequences
    candidates = _nominal[np.newaxis, :, :] + noise
    candidates = np.clip(candidates, lo, hi)

    # Roll out trajectories in parallel
    nstep = HORIZON * CONTROL_SUBSTEPS
    state_traj = parallel_rollout(model, _data_list, current_state, candidates, nstep)

    # Compute costs
    costs = compute_trajectory_costs(state_traj, candidates, nq, nv, _home_pos)

    # MPPI weights
    costs_shifted = costs - np.min(costs)
    weights = np.exp(-costs_shifted / TEMPERATURE)
    weights /= np.sum(weights) + 1e-10

    # Update nominal via weighted average of noise
    weighted_noise = np.sum(weights[:, np.newaxis, np.newaxis] * noise, axis=0)
    _nominal = _nominal + weighted_noise
    _nominal = np.clip(_nominal, lo, hi)

    # Extract first action
    action = _nominal[0].copy()

    # Shift horizon, fill last step with home positions
    _nominal = np.roll(_nominal, -1, axis=0)
    _nominal[-1] = _home_pos.copy()

    return action


# ── Visualization ─────────────────────────────────────────────────────────────

def run_visualized(model: mujoco.MjModel, data: mujoco.MjData) -> None:
    """Run MPC with the MuJoCo native viewer (real-time visualization)."""
    import mujoco.viewer

    if model.nkey > 0:
        mujoco.mj_resetDataKeyframe(model, data, 0)
    mujoco.mj_forward(model, data)

    initial_x = data.qpos[0]

    with mujoco.viewer.launch_passive(model, data) as viewer:
        start = time.time()
        for step in range(N_CONTROL_STEPS):
            if not viewer.is_running():
                break
            step_start = time.time()

            action = get_action(model, data)
            data.ctrl[:] = action
            for _ in range(CONTROL_SUBSTEPS):
                mujoco.mj_step(model, data)
            viewer.sync()

            elapsed = time.time() - step_start
            sleep_time = CONTROL_DT - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

        wall_time = time.time() - start
        final_x = data.qpos[0]
        avg_speed = (final_x - initial_x) / (N_CONTROL_STEPS * CONTROL_DT)
        print(f"\n---")
        print(f"avg_speed_mps:    {avg_speed:.6f}")
        print(f"total_seconds:    {wall_time:.1f}")

        while viewer.is_running():
            time.sleep(0.1)


# ── Main: run evaluation ──────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="MPPI controller for G1")
    parser.add_argument("--viz", action="store_true",
                        help="Launch MuJoCo viewer for real-time visualization")
    parser.add_argument("--fast", action="store_true",
                        help="Quick 5s evaluation for screening")
    args = parser.parse_args()

    # Reset globals for fresh run
    _nominal = None
    _home_pos = None
    _data_list = None
    _rng = None

    print("Setting up simulation...")
    model, data = make_sim()
    model.opt.timestep = SIM_DT

    print(f"  horizon={HORIZON}, samples={NUM_SAMPLES}, "
          f"temperature={TEMPERATURE}, noise_std={NOISE_STD}")
    print(f"  SIM_DT={SIM_DT}, CONTROL_DT={CONTROL_DT}, "
          f"CONTROL_SUBSTEPS={CONTROL_SUBSTEPS}")

    if args.viz:
        print("Launching viewer...")
        run_visualized(model, data)
    else:
        from prepare import evaluate_speed

        n_steps = 250 if args.fast else N_CONTROL_STEPS
        duration = n_steps * CONTROL_DT
        print(f"Running MPPI controller for {duration:.0f}s (threads={NTHREAD})...")

        t0 = time.time()
        avg_speed = evaluate_speed(model, data, n_steps=n_steps)
        wall_time = time.time() - t0

        print(f"\n---")
        print(f"avg_speed_mps:    {avg_speed:.6f}")
        print(f"total_seconds:    {wall_time:.1f}")
        print(f"horizon:          {HORIZON}")
        print(f"num_samples:      {NUM_SAMPLES}")
        print(f"temperature:      {TEMPERATURE}")
        print(f"noise_std:        {NOISE_STD}")
