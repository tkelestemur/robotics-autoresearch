"""Fixed infrastructure for the robotics autoresearch loop.

Downloads the Unitree H1 humanoid model, provides simulation helpers,
and evaluates MPC controllers. This file is READ-ONLY to the agent.
"""

import os
import time
import urllib.request
import urllib.error
from pathlib import Path

import mujoco
import mujoco.rollout
import numpy as np

# ── Constants ──────────────────────────────────────────────────────────────────

SIM_DURATION = 30.0        # seconds of simulation
SIM_DT = 0.002             # physics timestep (500 Hz)
CONTROL_DT = 0.02          # control rate (50 Hz)
CONTROL_SUBSTEPS = 10      # physics steps per control step
N_CONTROL_STEPS = 1500     # total control steps per evaluation

CACHE_DIR = Path.home() / ".cache" / "robotics-autoresearch"

# ── H1 model download ─────────────────────────────────────────────────────────

_MENAGERIE_BASE = (
    "https://raw.githubusercontent.com/google-deepmind/mujoco_menagerie"
    "/main/unitree_h1"
)

_H1_FILES = [
    "h1.xml",
    "scene.xml",
    # STL mesh assets
    "assets/left_ankle_link.stl",
    "assets/left_hip_pitch_link.stl",
    "assets/left_hip_roll_link.stl",
    "assets/left_hip_yaw_link.stl",
    "assets/left_knee_link.stl",
    "assets/logo_link.stl",
    "assets/pelvis.stl",
    "assets/right_ankle_link.stl",
    "assets/right_hip_pitch_link.stl",
    "assets/right_hip_roll_link.stl",
    "assets/right_hip_yaw_link.stl",
    "assets/right_knee_link.stl",
    "assets/torso_link.stl",
    "assets/left_elbow_link.stl",
    "assets/left_shoulder_pitch_link.stl",
    "assets/left_shoulder_roll_link.stl",
    "assets/left_shoulder_yaw_link.stl",
    "assets/right_elbow_link.stl",
    "assets/right_shoulder_pitch_link.stl",
    "assets/right_shoulder_roll_link.stl",
    "assets/right_shoulder_yaw_link.stl",
]


def download_h1_model(max_retries: int = 3) -> Path:
    """Download H1 XML + mesh assets from mujoco_menagerie (idempotent)."""
    model_dir = CACHE_DIR / "unitree_h1"
    assets_dir = model_dir / "assets"
    assets_dir.mkdir(parents=True, exist_ok=True)

    for fname in _H1_FILES:
        dest = model_dir / fname
        if dest.exists():
            continue
        url = f"{_MENAGERIE_BASE}/{fname}"
        for attempt in range(1, max_retries + 1):
            try:
                print(f"  Downloading {fname} (attempt {attempt})...")
                urllib.request.urlretrieve(url, dest)
                break
            except urllib.error.URLError as e:
                if attempt == max_retries:
                    raise RuntimeError(f"Failed to download {url}: {e}") from e
                time.sleep(2 ** attempt)

    return model_dir


# ── Simulation helpers ─────────────────────────────────────────────────────────

def make_sim() -> tuple[mujoco.MjModel, mujoco.MjData]:
    """Load the H1 scene and reset to standing pose (home keyframe)."""
    model_dir = CACHE_DIR / "unitree_h1"
    scene_path = model_dir / "scene.xml"
    if not scene_path.exists():
        download_h1_model()

    model = mujoco.MjModel.from_xml_path(str(scene_path))
    model.opt.timestep = SIM_DT
    data = mujoco.MjData(model)

    # Reset to home keyframe (standing pose)
    if model.nkey > 0:
        mujoco.mj_resetDataKeyframe(model, data, 0)
    mujoco.mj_forward(model, data)

    return model, data


def get_initial_state(model: mujoco.MjModel, data: mujoco.MjData) -> np.ndarray:
    """Extract full physics state (qpos, qvel, act, etc.)."""
    state_spec = mujoco.mjtState.mjSTATE_FULLPHYSICS
    state_size = mujoco.mj_stateSize(model, state_spec)
    state = np.empty(state_size, dtype=np.float64)
    mujoco.mj_getState(model, data, state, state_spec)
    return state


def set_state(
    model: mujoco.MjModel, data: mujoco.MjData, state: np.ndarray
) -> None:
    """Set full physics state."""
    state_spec = mujoco.mjtState.mjSTATE_FULLPHYSICS
    mujoco.mj_setState(model, data, state, state_spec)


def make_data_list(
    model: mujoco.MjModel, nthread: int
) -> list[mujoco.MjData]:
    """Create per-thread MjData instances for parallel rollouts."""
    return [mujoco.MjData(model) for _ in range(nthread)]


def parallel_rollout(
    model: mujoco.MjModel,
    data_list: list[mujoco.MjData],
    initial_state: np.ndarray,
    control_seq: np.ndarray,
    nstep: int,
) -> np.ndarray:
    """Roll out sampled trajectories in parallel using mujoco.rollout.

    Args:
        model: MjModel instance.
        data_list: Per-thread MjData list (from make_data_list).
        initial_state: Full physics state array.
        control_seq: Control sequences, shape (nbatch, horizon, nu).
            Each control is applied for CONTROL_SUBSTEPS physics steps.
        nstep: Number of physics steps to simulate per trajectory
            (= horizon * CONTROL_SUBSTEPS).

    Returns:
        State trajectories, shape (nbatch, nstep, nstate).
    """
    nbatch = control_seq.shape[0]
    horizon = control_seq.shape[1]
    nu = control_seq.shape[2]

    # Repeat each control for CONTROL_SUBSTEPS physics steps
    ctrl_expanded = np.repeat(control_seq, CONTROL_SUBSTEPS, axis=1)
    # ctrl_expanded shape: (nbatch, nstep, nu)

    # Tile initial state for all batches
    initial_states = np.tile(initial_state, (nbatch, 1))

    state_spec = mujoco.mjtState.mjSTATE_FULLPHYSICS
    state_size = mujoco.mj_stateSize(model, state_spec)

    state_traj = np.empty((nbatch, nstep, state_size), dtype=np.float64)

    mujoco.rollout.rollout(
        model,
        data_list,
        initial_states,
        ctrl_expanded,
        state=state_traj,
    )

    return state_traj


# ── Evaluation ─────────────────────────────────────────────────────────────────

def evaluate_speed(model: mujoco.MjModel, data: mujoco.MjData) -> float:
    """Run MPC for SIM_DURATION seconds, return average forward speed (m/s).

    Imports mpc.get_action at call time so the agent's latest code is used.
    """
    from mpc import get_action

    # Reset to home keyframe
    if model.nkey > 0:
        mujoco.mj_resetDataKeyframe(model, data, 0)
    mujoco.mj_forward(model, data)

    initial_x = data.qpos[0]

    for step in range(N_CONTROL_STEPS):
        action = get_action(model, data)
        data.ctrl[:] = action
        for _ in range(CONTROL_SUBSTEPS):
            mujoco.mj_step(model, data)

    final_x = data.qpos[0]
    elapsed = N_CONTROL_STEPS * CONTROL_DT
    avg_speed = (final_x - initial_x) / elapsed

    return avg_speed


# ── Main ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Downloading Unitree H1 model...")
    model_dir = download_h1_model()
    print(f"H1 model ready at: {model_dir}")

    model, data = make_sim()
    print(f"Model loaded: {model.nq} qpos, {model.nv} qvel, {model.nu} actuators")
    print(f"Simulation: {SIM_DURATION}s at {1/SIM_DT:.0f} Hz physics, {1/CONTROL_DT:.0f} Hz control")
    print("Ready for autoresearch.")
