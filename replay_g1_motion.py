#!/usr/bin/env python3
"""
replay_g1_motion.py

Replay a G1 + Inspire Hand motion CSV in MuJoCo.

Usage:
    python3 replay_g1_motion.py path/to/motion.csv
    python3 replay_g1_motion.py path/to/motion.csv --model path/to/g1_29dof_inspire_hand.xml
    python3 replay_g1_motion.py path/to/motion.csv --fps 50 --loop

The CSV must have a header row with actuator names matching JOINT_INFO.md.
All values are in radians.

Requirements:
    pip install mujoco numpy
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

import mujoco
import mujoco.viewer
import numpy as np

# Default model path (relative to this script)
_DEFAULT_MODEL = (
    Path(__file__).resolve().parent.parent
    / "Downloads/unitree-g1-mujoco/assets/g1_29dof_inspire_hand.xml"
)

# Body actuator → MuJoCo joint name (appends "_joint")
_BODY_ACTUATORS = {
    "left_hip_pitch", "left_hip_roll", "left_hip_yaw",
    "left_knee", "left_ankle_pitch", "left_ankle_roll",
    "right_hip_pitch", "right_hip_roll", "right_hip_yaw",
    "right_knee", "right_ankle_pitch", "right_ankle_roll",
    "waist_yaw", "waist_roll", "waist_pitch",
    "left_shoulder_pitch", "left_shoulder_roll", "left_shoulder_yaw",
    "left_elbow", "left_wrist_roll", "left_wrist_pitch", "left_wrist_yaw",
    "right_shoulder_pitch", "right_shoulder_roll", "right_shoulder_yaw",
    "right_elbow", "right_wrist_roll", "right_wrist_pitch", "right_wrist_yaw",
}

# Hand actuator → MuJoCo joint name (same name)
_HAND_ACTUATORS = {
    "l_ih_thumb_proximal_yaw", "l_ih_thumb_proximal_pitch",
    "l_ih_index_proximal", "l_ih_middle_proximal",
    "l_ih_ring_proximal", "l_ih_pinky_proximal",
    "r_ih_thumb_proximal_yaw", "r_ih_thumb_proximal_pitch",
    "r_ih_index_proximal", "r_ih_middle_proximal",
    "r_ih_ring_proximal", "r_ih_pinky_proximal",
}


def _actuator_to_joint(name: str) -> str:
    """Return the MuJoCo joint name for an actuator name."""
    if name in _BODY_ACTUATORS:
        return name + "_joint"
    return name  # hand joints share the same name


def build_qpos_map(model: mujoco.MjModel, col_names: List[str]) -> Dict[str, int]:
    """
    Map CSV column names → qpos array indices.
    Returns only the columns that could be resolved.
    """
    mapping: Dict[str, int] = {}
    for col in col_names:
        joint_name = _actuator_to_joint(col)
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
        if jid >= 0:
            qpos_idx = model.jnt_qposadr[jid]
            mapping[col] = qpos_idx
        else:
            print(f"  [warn] Joint '{joint_name}' not found in model — column '{col}' skipped.")
    return mapping


def build_mimic_table(
    model: mujoco.MjModel,
) -> List[tuple]:
    """
    Read all mjEQ_JOINT equality constraints and return a list of
    (master_qpos_idx, slave_qpos_idx, poly0, poly1) tuples.

    During kinematic replay MuJoCo does NOT auto-apply equality constraints
    (they are only enforced via constraint forces during mj_step).  We must
    manually compute slave qpos = poly0 + poly1 * master_qpos every frame so
    that intermediate/distal finger phalanges actually move.
    """
    EQ_JOINT = int(mujoco.mjtEq.mjEQ_JOINT)
    table = []
    for i in range(model.neq):
        if int(model.eq_type[i]) != EQ_JOINT:
            continue
        j1 = int(model.eq_obj1id[i])
        j2 = int(model.eq_obj2id[i])
        poly = model.eq_data[i]          # shape (11,); only [0] and [1] used
        table.append((
            int(model.jnt_qposadr[j1]),  # master qpos index
            int(model.jnt_qposadr[j2]),  # slave  qpos index
            float(poly[0]),               # poly offset
            float(poly[1]),               # poly linear coefficient
        ))
    return table


def load_csv(path: Path) -> tuple[List[str], np.ndarray]:
    """Load CSV → (header_list, float_array).  First row is header."""
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader)
        rows = [list(map(float, row)) for row in reader if row]
    return header, np.array(rows, dtype=np.float64)


def replay(
    csv_path: Path,
    model_path: Path,
    fps: int = 50,
    loop: bool = False,
    speed: float = 1.0,
) -> None:
    print(f"\n{'='*60}")
    print(f"  Motion:  {csv_path.name}")
    print(f"  Model:   {model_path.name}")
    print(f"  FPS:     {fps}  |  Speed: {speed}x  |  Loop: {loop}")
    print(f"{'='*60}\n")

    # Load model
    model = mujoco.MjModel.from_xml_path(str(model_path))
    data  = mujoco.MjData(model)

    # Fix floating-base quaternion to identity (standing upright)
    data.qpos[3:7] = [1.0, 0.0, 0.0, 0.0]
    mujoco.mj_forward(model, data)

    # Load CSV
    header, frames = load_csv(csv_path)
    qpos_map = build_qpos_map(model, header)
    mimic_table = build_mimic_table(model)   # [(master_idx, slave_idx, p0, p1), ...]
    n_frames = len(frames)
    col_indices = {col: i for i, col in enumerate(header)}

    print(f"  Frames: {n_frames}  ({n_frames/fps:.2f}s at {fps} Hz)")
    print(f"  Joints resolved: {len(qpos_map)} / {len(header)}")
    print(f"  Mimic constraints applied per frame: {len(mimic_table)}\n")
    print("  Playing... (close viewer window or press Ctrl+C to stop)\n")

    frame_dt = 1.0 / (fps * speed)

    with mujoco.viewer.launch_passive(
        model, data,
        show_left_ui=False,
        show_right_ui=False,
    ) as viewer:
        viewer.cam.type      = mujoco.mjtCamera.mjCAMERA_FREE
        viewer.cam.azimuth   = 135
        viewer.cam.elevation = -20
        viewer.cam.distance  = 2.5
        viewer.cam.lookat[:] = [0.0, 0.0, 0.9]

        try:
            while True:
                for frame_idx in range(n_frames):
                    if not viewer.is_running():
                        return

                    t_frame_start = time.perf_counter()

                    # Set all resolved actuated joints from this frame
                    row = frames[frame_idx]
                    for col, qpos_idx in qpos_map.items():
                        csv_col = col_indices[col]
                        data.qpos[qpos_idx] = row[csv_col]

                    # Apply mimic (equality) constraints: slave = p0 + p1 * master
                    # This drives intermediate/distal finger phalanges that are
                    # not directly actuated but are coupled to their proximal joint.
                    for master_idx, slave_idx, p0, p1 in mimic_table:
                        data.qpos[slave_idx] = p0 + p1 * data.qpos[master_idx]

                    mujoco.mj_kinematics(model, data)
                    mujoco.mj_comPos(model, data)
                    viewer.sync()

                    # Timing: sleep to maintain requested fps
                    elapsed = time.perf_counter() - t_frame_start
                    sleep_t = frame_dt - elapsed
                    if sleep_t > 0:
                        time.sleep(sleep_t)

                if not loop:
                    # Hold last frame
                    print("  Playback complete. Close viewer to exit.")
                    while viewer.is_running():
                        time.sleep(0.05)
                    break

        except KeyboardInterrupt:
            print("\n  Interrupted.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Replay a G1 + Inspire Hand motion CSV in MuJoCo viewer"
    )
    parser.add_argument("csv", type=Path, help="Path to motion CSV file")
    parser.add_argument(
        "--model", type=Path, default=_DEFAULT_MODEL,
        help=f"Path to MuJoCo XML model (default: {_DEFAULT_MODEL})",
    )
    parser.add_argument(
        "--fps", type=int, default=50,
        help="Playback frame rate in Hz (default: 50)",
    )
    parser.add_argument(
        "--speed", type=float, default=1.0,
        help="Playback speed multiplier, e.g. 0.5 for half speed (default: 1.0)",
    )
    parser.add_argument(
        "--loop", action="store_true",
        help="Loop the animation until viewer is closed",
    )
    args = parser.parse_args()

    csv_path   = args.csv.expanduser().resolve()
    model_path = args.model.expanduser().resolve()

    if not csv_path.exists():
        print(f"ERROR: CSV file not found: {csv_path}")
        sys.exit(1)
    if not model_path.exists():
        print(f"ERROR: Model file not found: {model_path}")
        print(f"  Specify the model with --model path/to/g1_29dof_inspire_hand.xml")
        sys.exit(1)

    replay(csv_path, model_path, fps=args.fps, loop=args.loop, speed=args.speed)


if __name__ == "__main__":
    main()
