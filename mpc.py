"""MPPI-based Model Predictive Controller for Unitree H1 locomotion.

This is the ONLY file the agent may modify. The goal is to maximize
forward speed (m/s) of the H1 humanoid robot over 30 seconds.
"""

import os
import time

import mujoco
import numpy as np

from prepare import (
    CONTROL_DT,
    CONTROL_SUBSTEPS,
    make_data_list,
    make_sim,
    get_initial_state,
    parallel_rollout,
)

# ── Hyperparameters (tune these!) ──────────────────────────────────────────────

HORIZON = 10          # control steps lookahead (0.2s at 50 Hz)
NUM_SAMPLES = 128     # number of parallel trajectory samples
TEMPERATURE = 0.1     # MPPI inverse temperature (lower = more greedy)
NOISE_STD = 0.3       # std of Gaussian control perturbations

# Cost weights
SPEED_WEIGHT = 5.0    # reward for forward velocity
HEIGHT_WEIGHT = 2.0   # penalty for deviating from target height
UPRIGHT_WEIGHT = 1.0  # penalty for not being upright
CTRL_WEIGHT = 0.01    # penalty for control effort

# Target values
TARGET_HEIGHT = 1.0   # desired torso height (m)
FALL_HEIGHT = 0.3     # below this = fallen

# Parallelism
NTHREAD = os.cpu_count() or 4

# ── State indexing helpers ─────────────────────────────────────────────────────
# H1 state layout: qpos (nq) then qvel (nv) in full physics state
# qpos[0:3] = x, y, z position of torso
# qpos[3:7] = quaternion (w, x, y, z) orientation of torso
# qvel[0:3] = linear velocity (vx, vy, vz) in world frame
# qvel[3:6] = angular velocity


def _get_nq_nv(model: mujoco.MjModel) -> tuple[int, int]:
    return model.nq, model.nv


# ── Cost function ──────────────────────────────────────────────────────────────

def compute_trajectory_costs(
    state_traj: np.ndarray,
    control_seq: np.ndarray,
    nq: int,
    nv: int,
) -> np.ndarray:
    """Compute costs for a batch of trajectories.

    Args:
        state_traj: (nbatch, nstep, nstate) full physics states.
        control_seq: (nbatch, horizon, nu) control sequences.
        nq: number of generalized positions.
        nv: number of generalized velocities.

    Returns:
        costs: (nbatch,) total cost per trajectory.
    """
    nbatch = state_traj.shape[0]
    nstep = state_traj.shape[1]

    # Extract positions and velocities from state
    # State layout: [qpos (nq), qvel (nv), ...]
    qpos = state_traj[:, :, :nq]       # (nbatch, nstep, nq)
    qvel = state_traj[:, :, nq:nq+nv]  # (nbatch, nstep, nv)

    # Sample at control rate (every CONTROL_SUBSTEPS physics steps)
    ctrl_indices = np.arange(CONTROL_SUBSTEPS - 1, nstep, CONTROL_SUBSTEPS)
    qpos_ctrl = qpos[:, ctrl_indices, :]  # (nbatch, horizon, nq)
    qvel_ctrl = qvel[:, ctrl_indices, :]  # (nbatch, horizon, nv)

    # Forward velocity reward (negative cost for forward speed)
    vx = qvel_ctrl[:, :, 0]  # (nbatch, horizon)
    speed_cost = -SPEED_WEIGHT * vx  # reward forward velocity

    # Height penalty: keep torso near TARGET_HEIGHT
    z = qpos_ctrl[:, :, 2]  # (nbatch, horizon)
    height_cost = HEIGHT_WEIGHT * (z - TARGET_HEIGHT) ** 2

    # Upright penalty: quaternion w component should be close to 1
    qw = qpos_ctrl[:, :, 3]  # (nbatch, horizon)
    upright_cost = UPRIGHT_WEIGHT * (1.0 - qw ** 2)

    # Control effort
    ctrl_cost = CTRL_WEIGHT * np.sum(control_seq ** 2, axis=-1)  # (nbatch, horizon)

    # Per-step cost
    step_cost = speed_cost + height_cost + upright_cost + ctrl_cost  # (nbatch, horizon)

    # Fall penalty: large cost if robot falls
    min_height = np.min(z, axis=1)  # (nbatch,)
    fall_penalty = np.where(min_height < FALL_HEIGHT, 1000.0, 0.0)

    # Total cost: sum over horizon + fall penalty
    total_cost = np.sum(step_cost, axis=1) + fall_penalty  # (nbatch,)

    return total_cost


# ── MPPI Controller ────────────────────────────────────────────────────────────

# Persistent state across calls
_nominal: np.ndarray | None = None
_data_list: list | None = None


def get_action(model: mujoco.MjModel, data: mujoco.MjData) -> np.ndarray:
    """MPPI controller: called at each control step by evaluate_speed().

    Returns the action (nu,) to apply for the next control step.
    """
    global _nominal, _data_list

    nu = model.nu
    nq, nv = _get_nq_nv(model)

    # Initialize nominal trajectory and thread data on first call
    if _nominal is None or _nominal.shape != (HORIZON, nu):
        _nominal = np.zeros((HORIZON, nu), dtype=np.float64)
    if _data_list is None:
        _data_list = make_data_list(model, NTHREAD)

    # Get current state
    current_state = get_initial_state(model, data)

    # Sample noise
    noise = np.random.randn(NUM_SAMPLES, HORIZON, nu) * NOISE_STD

    # Create candidate control sequences
    candidates = _nominal[np.newaxis, :, :] + noise  # (NUM_SAMPLES, HORIZON, nu)

    # Clip to actuator limits
    ctrl_range = model.actuator_ctrlrange
    if ctrl_range is not None and len(ctrl_range) > 0:
        lo = ctrl_range[:, 0]
        hi = ctrl_range[:, 1]
        candidates = np.clip(candidates, lo, hi)

    # Roll out trajectories in parallel
    nstep = HORIZON * CONTROL_SUBSTEPS
    state_traj = parallel_rollout(model, _data_list, current_state, candidates, nstep)

    # Compute costs
    costs = compute_trajectory_costs(state_traj, candidates, nq, nv)

    # MPPI weights: softmax(-costs / temperature)
    costs_shifted = costs - np.min(costs)  # numerical stability
    weights = np.exp(-costs_shifted / TEMPERATURE)
    weights /= np.sum(weights) + 1e-10

    # Weighted update of nominal trajectory
    weighted_noise = np.sum(weights[:, np.newaxis, np.newaxis] * noise, axis=0)
    _nominal = _nominal + weighted_noise

    # Clip nominal to actuator limits
    if ctrl_range is not None and len(ctrl_range) > 0:
        _nominal = np.clip(_nominal, lo, hi)

    # Extract first action
    action = _nominal[0].copy()

    # Shift horizon: drop first, append zero
    _nominal = np.roll(_nominal, -1, axis=0)
    _nominal[-1] = 0.0

    return action


# ── Main: run evaluation ──────────────────────────────────────────────────────

if __name__ == "__main__":
    from prepare import evaluate_speed

    # Reset controller state
    _nominal = None
    _data_list = None

    print("Setting up simulation...")
    model, data = make_sim()

    print(f"Running MPPI controller for 30s...")
    print(f"  horizon={HORIZON}, samples={NUM_SAMPLES}, "
          f"temperature={TEMPERATURE}, noise_std={NOISE_STD}")
    print(f"  threads={NTHREAD}")

    t0 = time.time()
    avg_speed = evaluate_speed(model, data)
    wall_time = time.time() - t0

    print(f"\n---")
    print(f"avg_speed_mps:    {avg_speed:.6f}")
    print(f"total_seconds:    {wall_time:.1f}")
    print(f"horizon:          {HORIZON}")
    print(f"num_samples:      {NUM_SAMPLES}")
    print(f"temperature:      {TEMPERATURE}")
    print(f"noise_std:        {NOISE_STD}")
