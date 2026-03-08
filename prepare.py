"""Fixed infrastructure for the robotics autoresearch loop.

Downloads the Unitree G1 humanoid model, provides simulation helpers,
and evaluates MPC controllers. This file is READ-ONLY to the agent.
"""

import time
import urllib.request
import urllib.error
from pathlib import Path

import mujoco
import mujoco.rollout
import numpy as np

# ── Constants ──────────────────────────────────────────────────────────────────

CACHE_DIR = Path.home() / ".cache" / "robotics-autoresearch"

# ── G1 model download ─────────────────────────────────────────────────────────

_MENAGERIE_BASE = (
    "https://raw.githubusercontent.com/google-deepmind/mujoco_menagerie"
    "/main/unitree_g1"
)

_G1_FILES = [
    "g1.xml",
    "scene.xml",
    # STL mesh assets
    "assets/pelvis.STL",
    "assets/pelvis_contour_link.STL",
    "assets/left_hip_pitch_link.STL",
    "assets/left_hip_roll_link.STL",
    "assets/left_hip_yaw_link.STL",
    "assets/left_knee_link.STL",
    "assets/left_ankle_pitch_link.STL",
    "assets/left_ankle_roll_link.STL",
    "assets/right_hip_pitch_link.STL",
    "assets/right_hip_roll_link.STL",
    "assets/right_hip_yaw_link.STL",
    "assets/right_knee_link.STL",
    "assets/right_ankle_pitch_link.STL",
    "assets/right_ankle_roll_link.STL",
    "assets/waist_yaw_link_rev_1_0.STL",
    "assets/waist_roll_link_rev_1_0.STL",
    "assets/torso_link_rev_1_0.STL",
    "assets/logo_link.STL",
    "assets/head_link.STL",
    "assets/left_shoulder_pitch_link.STL",
    "assets/left_shoulder_roll_link.STL",
    "assets/left_shoulder_yaw_link.STL",
    "assets/left_elbow_link.STL",
    "assets/left_wrist_roll_link.STL",
    "assets/left_wrist_pitch_link.STL",
    "assets/left_wrist_yaw_link.STL",
    "assets/left_rubber_hand.STL",
    "assets/right_shoulder_pitch_link.STL",
    "assets/right_shoulder_roll_link.STL",
    "assets/right_shoulder_yaw_link.STL",
    "assets/right_elbow_link.STL",
    "assets/right_wrist_roll_link.STL",
    "assets/right_wrist_pitch_link.STL",
    "assets/right_wrist_yaw_link.STL",
    "assets/right_rubber_hand.STL",
]


def download_g1_model(max_retries: int = 3) -> Path:
    """Download G1 XML + mesh assets from mujoco_menagerie (idempotent)."""
    model_dir = CACHE_DIR / "unitree_g1"
    assets_dir = model_dir / "assets"
    assets_dir.mkdir(parents=True, exist_ok=True)

    for fname in _G1_FILES:
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
    """Load the G1 scene and reset to standing pose (home keyframe).

    NOTE: The caller should set model.opt.timestep if a non-default
    timestep is desired (see SIM_DT in mpc.py).
    """
    model_dir = CACHE_DIR / "unitree_g1"
    scene_path = model_dir / "scene.xml"
    if not scene_path.exists():
        download_g1_model()

    model = mujoco.MjModel.from_xml_path(str(scene_path))
    data = mujoco.MjData(model)

    # Reset to home keyframe (standing pose)
    if model.nkey > 0:
        mujoco.mj_resetDataKeyframe(model, data, 0)
    mujoco.mj_forward(model, data)

    return model, data


def compute_grav_comp(model: mujoco.MjModel) -> np.ndarray:
    """Compute gravity compensation generalized forces for the home keyframe.

    Returns (nu,) array of generalized forces that counteract gravity when
    the robot is at the home keyframe with zero velocity and acceleration.

    For position-controlled actuators (like G1), the ctrl offset needed is:
        ctrl_offset = grav_comp / kp
    where kp is the actuator's position gain.
    """
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, 0)
    data.qvel[:] = 0
    data.qacc[:] = 0
    mujoco.mj_inverse(model, data)

    torques = np.zeros(model.nu)
    for i in range(model.nu):
        jnt_id = model.actuator_trnid[i, 0]
        dof = model.joint(jnt_id).dofadr[0]
        torques[i] = data.qfrc_inverse[dof]
    return torques


def get_home_positions(model: mujoco.MjModel) -> np.ndarray:
    """Return the home keyframe joint positions for actuated joints (nu,).

    For position-controlled actuators, these are the natural nominal
    ctrl targets (commanding home positions holds the standing pose).
    """
    home = np.zeros(model.nu)
    key_qpos = model.key_qpos[0]
    for i in range(model.nu):
        jnt_id = model.actuator_trnid[i, 0]
        qpos_adr = model.joint(jnt_id).qposadr[0]
        home[i] = key_qpos[qpos_adr]
    return home


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
        nstep: Number of physics steps to simulate per trajectory.
            The control_substeps is inferred as nstep // horizon.

    Returns:
        State trajectories, shape (nbatch, nstep, nstate).
    """
    nbatch = control_seq.shape[0]
    horizon = control_seq.shape[1]
    control_substeps = nstep // horizon

    # Repeat each control for control_substeps physics steps
    ctrl_expanded = np.repeat(control_seq, control_substeps, axis=1)

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

def evaluate_speed(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    n_steps: int | None = None,
) -> float:
    """Run MPC for n_steps control steps, return average forward speed (m/s).

    Imports timing constants and get_action from mpc at call time so the
    agent's latest code is always used.

    Args:
        model: MjModel instance.
        data: MjData instance.
        n_steps: Number of control steps. Defaults to N_CONTROL_STEPS from mpc.
            Use a smaller value (e.g., 250 = 5s) for fast screening.
    """
    from mpc import get_action, CONTROL_DT, CONTROL_SUBSTEPS, N_CONTROL_STEPS

    if n_steps is None:
        n_steps = N_CONTROL_STEPS

    # Reset to home keyframe
    if model.nkey > 0:
        mujoco.mj_resetDataKeyframe(model, data, 0)
    mujoco.mj_forward(model, data)

    initial_x = data.qpos[0]

    for step in range(n_steps):
        action = get_action(model, data)
        data.ctrl[:] = action
        for _ in range(CONTROL_SUBSTEPS):
            mujoco.mj_step(model, data)

    final_x = data.qpos[0]
    elapsed = n_steps * CONTROL_DT
    avg_speed = (final_x - initial_x) / elapsed

    return avg_speed


# ── Main ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Downloading Unitree G1 model...")
    model_dir = download_g1_model()
    print(f"G1 model ready at: {model_dir}")

    model, data = make_sim()
    print(f"Model loaded: {model.nq} qpos, {model.nv} qvel, {model.nu} actuators")

    # Show utility info
    gc = compute_grav_comp(model)
    home = get_home_positions(model)
    print(f"\nGravity comp forces: {np.round(gc, 1)}")
    print(f"Home positions: {np.round(home, 2)}")
    print(f"\nActuator info:")
    for i in range(model.nu):
        name = model.actuator(i).name
        jnt_id = model.actuator_trnid[i, 0]
        lo, hi = model.jnt_range[jnt_id]
        print(f"  {name:35s}  ctrl=[{model.actuator_ctrlrange[i,0]:7.2f}, {model.actuator_ctrlrange[i,1]:7.2f}]  "
              f"joint=[{lo:6.3f}, {hi:6.3f}]  home={home[i]:6.3f}  gc={gc[i]:7.2f}")

    print("\nReady for autoresearch.")
