"""MPPI-based Model Predictive Controller for Unitree G1 locomotion.

This is the ONLY file the agent may modify. The goal is to maximize
forward speed (m/s) of the G1 humanoid robot over 30 seconds.

The G1 uses POSITION CONTROL actuators: ctrl inputs are joint angle
targets (radians). MuJoCo's built-in PD (kp=500, dampratio=1) converts
these to torques at every physics substep.
"""

import time

import mujoco
import numpy as np

from prepare import (
    compute_grav_comp,
    get_home_positions,
    make_sim,
    get_initial_state,
    parallel_rollout,
)

# ── Simulation timing (tune these!) ───────────────────────────────────────────

SIM_DT = 0.005             # physics timestep (200 Hz)
CONTROL_DT = 0.02          # control rate (50 Hz)
CONTROL_SUBSTEPS = int(round(CONTROL_DT / SIM_DT))  # physics steps per control step
SIM_DURATION = 30.0        # seconds of simulation
N_CONTROL_STEPS = int(round(SIM_DURATION / CONTROL_DT))  # total control steps

# ── MPPI Hyperparameters (tune these!) ────────────────────────────────────────

HORIZON = 28          # control steps lookahead (0.56s at 50 Hz)
NUM_SAMPLES = 1024    # number of parallel trajectory samples (use 1024+ on GPU)
TEMPERATURE = 0.5     # MPPI inverse temperature (lower = more greedy)
NOISE_STD = 0.25      # std of joint angle perturbations (radians)

# Cost weights
SPEED_WEIGHT = 5.0    # reward for forward velocity
HEIGHT_WEIGHT = 2.0   # penalty for deviating from target height
UPRIGHT_WEIGHT = 1.0  # penalty for not being upright
CTRL_WEIGHT = 0.1     # penalty for deviation from home position

# Target values
TARGET_HEIGHT = 0.793  # desired torso height (m) — G1 standing height
FALL_HEIGHT = 0.3      # below this = fallen

# Deterministic seed for reproducibility
SEED = 42


# ── Cost function ──────────────────────────────────────────────────────────────

def compute_trajectory_costs(
    qpos_traj: np.ndarray,
    qvel_traj: np.ndarray,
    control_seq: np.ndarray,
    home_pos: np.ndarray,
) -> np.ndarray:
    """Compute costs for a batch of trajectories.

    Args:
        qpos_traj: (nbatch, horizon, nq) joint positions at control rate.
        qvel_traj: (nbatch, horizon, nv) joint velocities at control rate.
        control_seq: (nbatch, horizon, nu) joint angle target sequences.
        home_pos: (nu,) home joint positions.

    Returns:
        costs: (nbatch,) total cost per trajectory.
    """
    # Forward velocity reward
    vx = qvel_traj[:, :, 0]
    speed_cost = -SPEED_WEIGHT * vx

    # Height penalty (one-sided: only penalize below target)
    z = qpos_traj[:, :, 2]
    height_diff = np.minimum(z - TARGET_HEIGHT, 0.0)
    height_cost = HEIGHT_WEIGHT * height_diff ** 2

    # Upright penalty (quaternion w component — 1.0 means perfectly upright)
    qw = qpos_traj[:, :, 3]
    upright_cost = UPRIGHT_WEIGHT * (1.0 - qw ** 2)

    # Control effort: penalize deviation from home position
    target_dev = control_seq - home_pos
    ctrl_cost = CTRL_WEIGHT * np.sum(target_dev ** 2, axis=-1)

    # Vertical velocity penalty (discourage bouncing)
    vz = qvel_traj[:, :, 2]
    vert_cost = 0.5 * vz ** 2

    step_cost = speed_cost + height_cost + upright_cost + ctrl_cost + vert_cost

    # Fall penalty
    min_height = np.min(z, axis=1)
    fall_penalty = np.where(min_height < FALL_HEIGHT, 1000.0, 0.0)

    return np.sum(step_cost, axis=1) + fall_penalty


# ── MPPI Controller ────────────────────────────────────────────────────────────

_nominal: np.ndarray | None = None
_home_pos: np.ndarray | None = None
_rng: np.random.Generator | None = None


def get_action(model: mujoco.MjModel, data: mujoco.MjData) -> np.ndarray:
    """MPPI controller: called at each control step by evaluate_speed().

    Returns joint angle targets (nu,) to apply for the next control step.
    MuJoCo's built-in PD converts these to torques at every physics substep.
    """
    global _nominal, _home_pos, _rng

    nu = model.nu
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

    current_state = get_initial_state(model, data)

    # Sample noise in joint angle space (radians)
    noise = _rng.standard_normal((NUM_SAMPLES, HORIZON, nu)) * NOISE_STD

    # Candidate joint angle target sequences
    candidates = _nominal[np.newaxis, :, :] + noise
    candidates = np.clip(candidates, lo, hi)

    # Roll out trajectories in parallel (GPU or CPU)
    nstep = HORIZON * CONTROL_SUBSTEPS
    qpos_traj, qvel_traj = parallel_rollout(
        model, current_state, candidates, nstep
    )

    # Compute costs
    costs = compute_trajectory_costs(qpos_traj, qvel_traj, candidates, _home_pos)

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


# ── Benchmark ─────────────────────────────────────────────────────────────────

def run_benchmark(model: mujoco.MjModel, data: mujoco.MjData) -> None:
    """Benchmark parallel_rollout throughput at various sample counts."""
    import warp as wp

    device = "CUDA" if wp.is_cuda_available() else "CPU"
    print(f"Device: {wp.get_preferred_device()} ({device})")
    print(f"Horizon: {HORIZON}, CONTROL_SUBSTEPS: {CONTROL_SUBSTEPS}")
    print(f"Physics steps per rollout: {HORIZON * CONTROL_SUBSTEPS}")
    print()

    if model.nkey > 0:
        mujoco.mj_resetDataKeyframe(model, data, 0)
    mujoco.mj_forward(model, data)
    state = get_initial_state(model, data)
    home = get_home_positions(model)
    nstep = HORIZON * CONTROL_SUBSTEPS

    sample_counts = [16, 32, 64, 128, 256, 512, 1024]
    n_iters = 20  # control steps to average over

    print(f"{'samples':>8}  {'rollout_ms':>11}  {'ctrl_hz':>8}  "
          f"{'steps/s':>10}  {'realtime_x':>10}")
    print("-" * 62)

    for n_samples in sample_counts:
        ctrl_seq = np.tile(home, (n_samples, HORIZON, 1))
        noise = np.random.default_rng(0).standard_normal(ctrl_seq.shape) * NOISE_STD
        ctrl_seq = np.clip(ctrl_seq + noise, model.actuator_ctrlrange[:, 0],
                           model.actuator_ctrlrange[:, 1])

        # Warmup (triggers lazy init / JIT for this batch size)
        parallel_rollout(model, state, ctrl_seq, nstep)

        # Timed runs
        t0 = time.time()
        for _ in range(n_iters):
            parallel_rollout(model, state, ctrl_seq, nstep)
        elapsed = time.time() - t0

        ms_per_rollout = (elapsed / n_iters) * 1000
        ctrl_hz = n_iters / elapsed
        total_steps = n_samples * HORIZON * CONTROL_SUBSTEPS
        steps_per_sec = (total_steps * n_iters) / elapsed
        realtime_x = ctrl_hz * CONTROL_DT  # fraction of realtime

        print(f"{n_samples:>8}  {ms_per_rollout:>10.1f}ms  {ctrl_hz:>7.1f}Hz  "
              f"{steps_per_sec:>10.0f}  {realtime_x:>9.2f}x")

    print()
    print(f"ctrl_hz = control steps per second (higher = better)")
    print(f"realtime_x = fraction of realtime "
          f"(>1.0 means faster than realtime at CONTROL_DT={CONTROL_DT}s)")


# ── Main: run evaluation ──────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="MPPI controller for G1")
    parser.add_argument("--viz", action="store_true",
                        help="Launch MuJoCo viewer for real-time visualization")
    parser.add_argument("--fast", action="store_true",
                        help="Quick 5s evaluation for screening")
    parser.add_argument("--benchmark", action="store_true",
                        help="Benchmark parallel_rollout at various sample counts")
    args = parser.parse_args()

    # Reset globals for fresh run
    _nominal = None
    _home_pos = None
    _rng = None

    print("Setting up simulation...")
    model, data = make_sim()
    model.opt.timestep = SIM_DT

    if args.benchmark:
        run_benchmark(model, data)
    elif args.viz:
        print("Launching viewer...")
        run_visualized(model, data)
    else:
        print(f"  horizon={HORIZON}, samples={NUM_SAMPLES}, "
              f"temperature={TEMPERATURE}, noise_std={NOISE_STD}")
        print(f"  SIM_DT={SIM_DT}, CONTROL_DT={CONTROL_DT}, "
              f"CONTROL_SUBSTEPS={CONTROL_SUBSTEPS}")

        from prepare import evaluate_speed

        n_steps = 250 if args.fast else N_CONTROL_STEPS
        duration = n_steps * CONTROL_DT
        print(f"Running MPPI controller for {duration:.0f}s...")

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
