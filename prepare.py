"""Fixed infrastructure for the robotics autoresearch loop.

Downloads the Unitree G1 humanoid model, provides simulation helpers,
and evaluates MPC controllers. This file is READ-ONLY to the agent.

Uses mujoco_warp for parallel rollouts (GPU-accelerated on Linux/NVIDIA,
CPU fallback on macOS).
"""

import time
import urllib.request
import urllib.error
from pathlib import Path

import mujoco
import mujoco_warp as mjw
import numpy as np
import warp as wp

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

    # Restrict contacts to foot-floor only. This disables body/limb contacts
    # which dramatically reduces constraint count (njmax 161→~32) and prevents
    # the robot from exploiting body-floor sliding for forward motion.
    _foot_bodies = {"left_ankle_roll_link", "right_ankle_roll_link"}
    for i in range(model.ngeom):
        geom = model.geom(i)
        if geom.name == "floor":
            continue
        bodyid = int(geom.bodyid[0])
        body_name = model.body(bodyid).name
        if body_name not in _foot_bodies:
            model.geom_contype[i] = 0
            model.geom_conaffinity[i] = 0

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
    """Extract full physics state (time, qpos, qvel, act).

    State layout: [time(1), qpos(nq), qvel(nv), act(na)]
    """
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


# ── Parallel rollout (mujoco_warp) ────────────────────────────────────────────


class WarpRollout:
    """Manages mujoco_warp resources for batched parallel rollouts.

    Lazily initializes device model, data, and trajectory buffers on first
    use. Reuses allocations across calls; only reallocates when batch size
    or horizon changes.

    On GPU: uses CUDA graph capture to eliminate per-step kernel launch
    overhead, and keeps the full control sequence on device to avoid
    host→device transfers in the inner loop.
    """

    def __init__(self):
        self._model = None
        self._data = None
        self._nworld = 0
        self._qpos_buf = None
        self._qvel_buf = None
        self._ctrl_dev = None  # full control seq on device (horizon, nbatch, nu)
        self._horizon = 0
        self._step_graph = None  # CUDA graph for mjw.step
        self._use_graphs = False
        self._control_substeps = 0

    def _ensure_model(self, model: mujoco.MjModel) -> None:
        if self._model is None:
            self._model = mjw.put_model(model)

    def _ensure_data(self, model: mujoco.MjModel, nbatch: int) -> None:
        if self._data is None or self._nworld != nbatch:
            self._data = mjw.make_data(model, nworld=nbatch, njmax=80, nconmax=16)
            self._nworld = nbatch
            self._step_graph = None  # invalidate graph on data change

    def _ensure_buffers(
        self, horizon: int, nbatch: int, nq: int, nv: int, nu: int,
    ) -> None:
        if (self._qpos_buf is None
                or self._horizon != horizon
                or self._qpos_buf.shape[1] != nbatch):
            self._qpos_buf = wp.zeros((horizon, nbatch, nq), dtype=wp.float32)
            self._qvel_buf = wp.zeros((horizon, nbatch, nv), dtype=wp.float32)
            self._ctrl_dev = wp.zeros((horizon, nbatch, nu), dtype=wp.float32)
            self._horizon = horizon

    def _ensure_graph(self, control_substeps: int) -> None:
        """Capture a CUDA graph for `control_substeps` calls to mjw.step."""
        if not wp.is_cuda_available():
            return
        if (self._step_graph is not None
                and self._control_substeps == control_substeps):
            return
        # Capture the inner physics loop as a CUDA graph
        with wp.ScopedCapture() as capture:
            for _ in range(control_substeps):
                mjw.step(self._model, self._data)
        self._step_graph = capture.graph
        self._control_substeps = control_substeps
        self._use_graphs = True

    def rollout(
        self,
        model: mujoco.MjModel,
        initial_state: np.ndarray,
        control_seq: np.ndarray,
        nstep: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Roll out sampled trajectories in parallel.

        Args:
            model: MjModel instance (CPU).
            initial_state: Full physics state from get_initial_state.
                Layout: [time(1), qpos(nq), qvel(nv), act(na)].
            control_seq: Control sequences, shape (nbatch, horizon, nu).
            nstep: Total physics steps (= horizon * control_substeps).

        Returns:
            qpos_traj: (nbatch, horizon, nq) positions at control rate.
            qvel_traj: (nbatch, horizon, nv) velocities at control rate.
        """
        nbatch, horizon, nu = control_seq.shape
        control_substeps = nstep // horizon
        nq, nv = model.nq, model.nv

        self._ensure_model(model)
        self._ensure_data(model, nbatch)
        self._ensure_buffers(horizon, nbatch, nq, nv, nu)
        self._ensure_graph(control_substeps)

        # Set initial state in-place (no allocation)
        qpos0 = initial_state[1:1 + nq].astype(np.float32)
        qvel0 = initial_state[1 + nq:1 + nq + nv].astype(np.float32)
        qpos_init = np.tile(qpos0, (nbatch, 1))
        qvel_init = np.tile(qvel0, (nbatch, 1))

        self._data.qpos.assign(qpos_init)
        self._data.qvel.assign(qvel_init)
        self._data.time.zero_()
        self._data.qacc.zero_()
        self._data.ctrl.zero_()
        mjw.forward(self._model, self._data)

        # Upload full control sequence to device in one transfer
        # Layout: (horizon, nbatch, nu) — each [t] slice is contiguous
        ctrl_f32 = np.ascontiguousarray(
            control_seq.transpose(1, 0, 2), dtype=np.float32
        )
        self._ctrl_dev.assign(ctrl_f32)

        # Rollout: step through horizon, accumulate on device
        for t in range(horizon):
            # Device-to-device ctrl copy (no host involvement)
            wp.copy(self._data.ctrl, self._ctrl_dev,
                    src_offset=t * nbatch * nu, count=nbatch * nu)
            if self._use_graphs:
                wp.capture_launch(self._step_graph)
            else:
                for _ in range(control_substeps):
                    mjw.step(self._model, self._data)
            # Record trajectory on device (no host sync)
            wp.copy(self._qpos_buf, self._data.qpos,
                    dest_offset=t * nbatch * nq, count=nbatch * nq)
            wp.copy(self._qvel_buf, self._data.qvel,
                    dest_offset=t * nbatch * nv, count=nbatch * nv)

        # Single device→host transfer
        qpos_raw = self._qpos_buf.numpy().reshape(horizon, nbatch, nq)
        qvel_raw = self._qvel_buf.numpy().reshape(horizon, nbatch, nv)
        return (qpos_raw.transpose(1, 0, 2).copy(),
                qvel_raw.transpose(1, 0, 2).copy())


_warp_rollout = WarpRollout()


def parallel_rollout(
    model: mujoco.MjModel,
    initial_state: np.ndarray,
    control_seq: np.ndarray,
    nstep: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Roll out sampled trajectories in parallel using mujoco_warp.

    Uses GPU acceleration on Linux/NVIDIA, CPU backend on macOS.
    """
    return _warp_rollout.rollout(model, initial_state, control_seq, nstep)


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
    print(f"Warp device: {wp.get_preferred_device()}")

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
