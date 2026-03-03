#!/usr/bin/env python3
"""
generate_g1_inspire_datasets.py

Generates synthetic motion CSV datasets for the Unitree G1 + Inspire Hand.

CSV format:
  - Row 0: header row with actuator names (41 columns)
  - Rows 1+: one frame per row at 50 Hz
  - Values in radians, clamped to joint limits
  - Column order matches actuator order in g1_29dof_inspire_hand.xml

Actuator order (41 total):
  Cols  0-11: Left/Right leg (hip pitch/roll/yaw, knee, ankle pitch/roll)
  Cols 12-14: Waist (yaw, roll, pitch)
  Cols 15-21: Left arm (shoulder pitch/roll/yaw, elbow, wrist roll/pitch/yaw)
  Cols 22-27: Left Inspire Hand (thumb yaw/pitch, index/middle/ring/pinky proximal)
  Cols 28-34: Right arm (shoulder pitch/roll/yaw, elbow, wrist roll/pitch/yaw)
  Cols 35-40: Right Inspire Hand (thumb yaw/pitch, index/middle/ring/pinky proximal)

Usage:
    python3 generate_g1_inspire_datasets.py
    python3 generate_g1_inspire_datasets.py --task wave_hello_right
    python3 generate_g1_inspire_datasets.py --list
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Joint definitions
# ---------------------------------------------------------------------------

ACTUATOR_NAMES: List[str] = [
    # Left leg (0-5)
    "left_hip_pitch", "left_hip_roll", "left_hip_yaw",
    "left_knee", "left_ankle_pitch", "left_ankle_roll",
    # Right leg (6-11)
    "right_hip_pitch", "right_hip_roll", "right_hip_yaw",
    "right_knee", "right_ankle_pitch", "right_ankle_roll",
    # Waist (12-14)
    "waist_yaw", "waist_roll", "waist_pitch",
    # Left arm (15-21)
    "left_shoulder_pitch", "left_shoulder_roll", "left_shoulder_yaw",
    "left_elbow", "left_wrist_roll", "left_wrist_pitch", "left_wrist_yaw",
    # Left Inspire Hand (22-27)
    "l_ih_thumb_proximal_yaw", "l_ih_thumb_proximal_pitch",
    "l_ih_index_proximal", "l_ih_middle_proximal",
    "l_ih_ring_proximal", "l_ih_pinky_proximal",
    # Right arm (28-34)
    "right_shoulder_pitch", "right_shoulder_roll", "right_shoulder_yaw",
    "right_elbow", "right_wrist_roll", "right_wrist_pitch", "right_wrist_yaw",
    # Right Inspire Hand (35-40)
    "r_ih_thumb_proximal_yaw", "r_ih_thumb_proximal_pitch",
    "r_ih_index_proximal", "r_ih_middle_proximal",
    "r_ih_ring_proximal", "r_ih_pinky_proximal",
]

# Joint limits in radians [min, max] — from JOINT_INFO.md
JOINT_LIMITS: Dict[str, Tuple[float, float]] = {
    "left_hip_pitch":          (-2.531,  2.880),
    "left_hip_roll":           (-0.524,  2.967),
    "left_hip_yaw":            (-2.757,  2.757),
    "left_knee":               (-0.087,  2.880),
    "left_ankle_pitch":        (-0.873,  0.524),
    "left_ankle_roll":         (-0.262,  0.262),
    "right_hip_pitch":         (-2.531,  2.880),
    "right_hip_roll":          (-2.967,  0.524),
    "right_hip_yaw":           (-2.757,  2.757),
    "right_knee":              (-0.087,  2.880),
    "right_ankle_pitch":       (-0.873,  0.524),
    "right_ankle_roll":        (-0.262,  0.262),
    "waist_yaw":               (-2.618,  2.618),
    "waist_roll":              (-0.520,  0.520),
    "waist_pitch":             (-0.520,  0.520),
    "left_shoulder_pitch":     (-3.089,  2.670),
    "left_shoulder_roll":      (-1.589,  2.251),
    "left_shoulder_yaw":       (-2.618,  2.618),
    "left_elbow":              (-1.047,  2.094),
    "left_wrist_roll":         (-1.972,  1.972),
    "left_wrist_pitch":        (-1.614,  1.614),
    "left_wrist_yaw":          (-1.614,  1.614),
    "l_ih_thumb_proximal_yaw": (0.0,     1.308),
    "l_ih_thumb_proximal_pitch":(0.0,    0.600),
    "l_ih_index_proximal":     (0.0,     1.470),
    "l_ih_middle_proximal":    (0.0,     1.470),
    "l_ih_ring_proximal":      (0.0,     1.470),
    "l_ih_pinky_proximal":     (0.0,     1.470),
    "right_shoulder_pitch":    (-3.089,  2.670),
    "right_shoulder_roll":     (-2.251,  1.589),
    "right_shoulder_yaw":      (-2.618,  2.618),
    "right_elbow":             (-1.047,  2.094),
    "right_wrist_roll":        (-1.972,  1.972),
    "right_wrist_pitch":       (-1.614,  1.614),
    "right_wrist_yaw":         (-1.614,  1.614),
    "r_ih_thumb_proximal_yaw": (0.0,     1.308),
    "r_ih_thumb_proximal_pitch":(0.0,    0.600),
    "r_ih_index_proximal":     (0.0,     1.470),
    "r_ih_middle_proximal":    (0.0,     1.470),
    "r_ih_ring_proximal":      (0.0,     1.470),
    "r_ih_pinky_proximal":     (0.0,     1.470),
}

# Shorthand helpers — open/closed finger values (radians)
FINGERS_OPEN  = {"l_ih_index_proximal": 0.0, "l_ih_middle_proximal": 0.0,
                 "l_ih_ring_proximal": 0.0,  "l_ih_pinky_proximal": 0.0}
FINGERS_CLOSE = {"l_ih_index_proximal": 1.2, "l_ih_middle_proximal": 1.2,
                 "l_ih_ring_proximal": 1.2,  "l_ih_pinky_proximal": 1.2}
R_FINGERS_OPEN  = {"r_ih_index_proximal": 0.0, "r_ih_middle_proximal": 0.0,
                   "r_ih_ring_proximal": 0.0,  "r_ih_pinky_proximal": 0.0}
R_FINGERS_CLOSE = {"r_ih_index_proximal": 1.2, "r_ih_middle_proximal": 1.2,
                   "r_ih_ring_proximal": 1.2,  "r_ih_pinky_proximal": 1.2}

# ---------------------------------------------------------------------------
# Keyframe task definitions
# ---------------------------------------------------------------------------
# Each task is a list of (time_seconds, {joint_name: value_rad}) tuples.
# Time = 0 is the neutral standing pose.  All unspecified joints stay at 0.
# The generator fills missing joints via cubic interpolation from their
# last known value; joints never mentioned default to 0 throughout.

_NEUTRAL: Dict[str, float] = {}   # all zeros implicitly


def _d(deg: float) -> float:
    """Degrees → radians."""
    return math.radians(deg)


TASKS: Dict[str, List[Tuple[float, Dict[str, float]]]] = {

    # -----------------------------------------------------------------------
    # SOCIAL / EXPRESSIVE
    # -----------------------------------------------------------------------

    "wave_hello_right": [
        (0.0,  {}),
        (0.5,  {"right_shoulder_pitch": _d(-80), "right_shoulder_roll": _d(-20),
                "right_elbow": _d(80)}),
        (0.9,  {"right_wrist_yaw": _d(45)}),
        (1.3,  {"right_wrist_yaw": _d(-45)}),
        (1.7,  {"right_wrist_yaw": _d(45)}),
        (2.1,  {"right_wrist_yaw": _d(-45)}),
        (2.5,  {"right_wrist_yaw": _d(45)}),
        (3.0,  {"right_shoulder_pitch": 0, "right_shoulder_roll": 0,
                "right_elbow": 0, "right_wrist_yaw": 0}),
    ],

    "wave_hello_left": [
        (0.0,  {}),
        (0.5,  {"left_shoulder_pitch": _d(-80), "left_shoulder_roll": _d(20),
                "left_elbow": _d(80)}),
        (0.9,  {"left_wrist_yaw": _d(-45)}),
        (1.3,  {"left_wrist_yaw": _d(45)}),
        (1.7,  {"left_wrist_yaw": _d(-45)}),
        (2.1,  {"left_wrist_yaw": _d(45)}),
        (2.5,  {"left_wrist_yaw": _d(-45)}),
        (3.0,  {"left_shoulder_pitch": 0, "left_shoulder_roll": 0,
                "left_elbow": 0, "left_wrist_yaw": 0}),
    ],

    "clap_hands": [
        (0.0,  {}),
        (0.6,  {"left_shoulder_pitch":  _d(-60), "left_shoulder_roll":  _d(30),
                "left_elbow":  _d(90),
                "right_shoulder_pitch": _d(-60), "right_shoulder_roll": _d(-30),
                "right_elbow": _d(90)}),
        (1.1,  {"left_shoulder_roll": _d(5), "right_shoulder_roll": _d(-5)}),   # clap
        (1.5,  {"left_shoulder_roll": _d(30), "right_shoulder_roll": _d(-30)}), # apart
        (1.9,  {"left_shoulder_roll": _d(5), "right_shoulder_roll": _d(-5)}),   # clap
        (2.3,  {"left_shoulder_roll": _d(30), "right_shoulder_roll": _d(-30)}), # apart
        (2.7,  {"left_shoulder_roll": _d(5), "right_shoulder_roll": _d(-5)}),   # clap
        (3.2,  {"left_shoulder_pitch": 0, "left_shoulder_roll": 0,
                "left_elbow": 0,
                "right_shoulder_pitch": 0, "right_shoulder_roll": 0,
                "right_elbow": 0}),
    ],

    "thumbs_up_right": [
        (0.0,  {}),
        (0.6,  {"right_shoulder_pitch": _d(-50), "right_shoulder_roll": _d(-25),
                "right_elbow": _d(90), "right_wrist_pitch": _d(-30)}),
        (0.9,  {**R_FINGERS_CLOSE,
                "r_ih_thumb_proximal_yaw": 0.0,
                "r_ih_thumb_proximal_pitch": 0.0}),  # thumb stays extended
        (2.0,  {}),   # hold
        (2.8,  {**R_FINGERS_OPEN,
                "right_shoulder_pitch": 0, "right_shoulder_roll": 0,
                "right_elbow": 0, "right_wrist_pitch": 0}),
    ],

    "point_at_object_right": [
        (0.0,  {}),
        (0.5,  {"right_shoulder_pitch": _d(-75), "right_shoulder_roll": _d(-15),
                "right_elbow": _d(10)}),
        (0.8,  {"r_ih_middle_proximal": 1.3, "r_ih_ring_proximal": 1.3,
                "r_ih_pinky_proximal": 1.3,
                "r_ih_thumb_proximal_yaw": 0.8, "r_ih_thumb_proximal_pitch": 0.4,
                "r_ih_index_proximal": 0.0}),    # index stays straight
        (2.5,  {}),   # hold pointing
        (3.2,  {**R_FINGERS_OPEN,
                "r_ih_thumb_proximal_yaw": 0, "r_ih_thumb_proximal_pitch": 0,
                "right_shoulder_pitch": 0, "right_shoulder_roll": 0,
                "right_elbow": 0}),
    ],

    "handshake_right": [
        (0.0,  {}),
        (0.7,  {"right_shoulder_pitch": _d(-55), "right_shoulder_roll": _d(-15),
                "right_elbow": _d(50), "right_wrist_pitch": _d(20)}),
        (1.1,  {**R_FINGERS_CLOSE,
                "r_ih_thumb_proximal_yaw": 0.5, "r_ih_thumb_proximal_pitch": 0.3}),
        (1.5,  {**R_FINGERS_OPEN}),
        (1.9,  {**R_FINGERS_CLOSE,
                "r_ih_thumb_proximal_yaw": 0.5, "r_ih_thumb_proximal_pitch": 0.3}),
        (2.6,  {**R_FINGERS_OPEN,
                "right_shoulder_pitch": 0, "right_shoulder_roll": 0,
                "right_elbow": 0, "right_wrist_pitch": 0,
                "r_ih_thumb_proximal_yaw": 0, "r_ih_thumb_proximal_pitch": 0}),
    ],

    # -----------------------------------------------------------------------
    # MANIPULATION
    # -----------------------------------------------------------------------

    "reach_and_grasp_right": [
        (0.0,  {}),
        (0.8,  {"right_shoulder_pitch": _d(-65), "right_shoulder_roll": _d(-10),
                "right_elbow": _d(60), "right_wrist_pitch": _d(-15)}),
        (1.3,  {**R_FINGERS_CLOSE,
                "r_ih_thumb_proximal_yaw": 0.7, "r_ih_thumb_proximal_pitch": 0.5}),
        (2.5,  {}),  # hold
        (3.3,  {**R_FINGERS_OPEN,
                "r_ih_thumb_proximal_yaw": 0, "r_ih_thumb_proximal_pitch": 0,
                "right_shoulder_pitch": 0, "right_shoulder_roll": 0,
                "right_elbow": 0, "right_wrist_pitch": 0}),
    ],

    "reach_and_grasp_left": [
        (0.0,  {}),
        (0.8,  {"left_shoulder_pitch": _d(-65), "left_shoulder_roll": _d(10),
                "left_elbow": _d(60), "left_wrist_pitch": _d(-15)}),
        (1.3,  {**FINGERS_CLOSE,
                "l_ih_thumb_proximal_yaw": 0.7, "l_ih_thumb_proximal_pitch": 0.5}),
        (2.5,  {}),  # hold
        (3.3,  {**FINGERS_OPEN,
                "l_ih_thumb_proximal_yaw": 0, "l_ih_thumb_proximal_pitch": 0,
                "left_shoulder_pitch": 0, "left_shoulder_roll": 0,
                "left_elbow": 0, "left_wrist_pitch": 0}),
    ],

    "bimanual_box_grasp": [
        (0.0,  {}),
        (0.8,  {"left_shoulder_pitch": _d(-55), "left_shoulder_roll": _d(35),
                "left_elbow": _d(85),
                "right_shoulder_pitch": _d(-55), "right_shoulder_roll": _d(-35),
                "right_elbow": _d(85)}),
        (1.4,  {"left_shoulder_roll":  _d(10),
                "right_shoulder_roll": _d(-10)}),  # hands approach box
        (2.0,  {**FINGERS_CLOSE, "l_ih_thumb_proximal_yaw": 0.6,
                "l_ih_thumb_proximal_pitch": 0.4,
                **R_FINGERS_CLOSE, "r_ih_thumb_proximal_yaw": 0.6,
                "r_ih_thumb_proximal_pitch": 0.4}),
        (2.8,  {"left_shoulder_pitch": _d(-75), "right_shoulder_pitch": _d(-75)}),  # lift
        (4.0,  {}),  # hold lifted
        (5.0,  {**FINGERS_OPEN, "l_ih_thumb_proximal_yaw": 0,
                "l_ih_thumb_proximal_pitch": 0,
                **R_FINGERS_OPEN, "r_ih_thumb_proximal_yaw": 0,
                "r_ih_thumb_proximal_pitch": 0,
                "left_shoulder_pitch": 0, "left_shoulder_roll": 0,
                "left_elbow": 0,
                "right_shoulder_pitch": 0, "right_shoulder_roll": 0,
                "right_elbow": 0}),
    ],

    "pick_and_place_right": [
        # Phase 1: reach down and grasp
        (0.0,  {}),
        (0.8,  {"right_shoulder_pitch": _d(-30), "right_shoulder_roll": _d(-10),
                "right_elbow": _d(100), "right_wrist_pitch": _d(30)}),  # reach low
        (1.3,  {**R_FINGERS_CLOSE,
                "r_ih_thumb_proximal_yaw": 0.7, "r_ih_thumb_proximal_pitch": 0.5}),
        # Phase 2: lift
        (2.0,  {"right_shoulder_pitch": _d(-90), "right_elbow": _d(70),
                "right_wrist_pitch": _d(-10)}),
        # Phase 3: move to place
        (2.8,  {"right_shoulder_pitch": _d(-70), "right_shoulder_roll": _d(-40),
                "right_elbow": _d(80)}),
        # Phase 4: lower and release
        (3.5,  {"right_shoulder_pitch": _d(-40), "right_elbow": _d(95),
                "right_wrist_pitch": _d(25)}),
        (3.9,  {**R_FINGERS_OPEN,
                "r_ih_thumb_proximal_yaw": 0, "r_ih_thumb_proximal_pitch": 0}),
        (4.6,  {"right_shoulder_pitch": 0, "right_shoulder_roll": 0,
                "right_elbow": 0, "right_wrist_pitch": 0}),
    ],

    "press_button_right_index": [
        (0.0,  {}),
        (0.6,  {"right_shoulder_pitch": _d(-60), "right_shoulder_roll": _d(-10),
                "right_elbow": _d(55), "right_wrist_pitch": _d(15)}),
        (0.9,  {"r_ih_middle_proximal": 1.3, "r_ih_ring_proximal": 1.3,
                "r_ih_pinky_proximal": 1.3,
                "r_ih_thumb_proximal_yaw": 0.7,
                "r_ih_index_proximal": 0.0}),  # index ready to press
        (1.2,  {"right_elbow": _d(75), "r_ih_index_proximal": 0.9}),  # press
        (1.6,  {"right_elbow": _d(55), "r_ih_index_proximal": 0.0}),  # release
        (1.9,  {"right_elbow": _d(75), "r_ih_index_proximal": 0.9}),  # press 2
        (2.3,  {"right_elbow": _d(55), "r_ih_index_proximal": 0.0}),  # release 2
        (2.8,  {**R_FINGERS_OPEN,
                "r_ih_thumb_proximal_yaw": 0,
                "right_shoulder_pitch": 0, "right_shoulder_roll": 0,
                "right_elbow": 0, "right_wrist_pitch": 0}),
    ],

    "pour_liquid_right": [
        (0.0,  {**R_FINGERS_CLOSE,
                "r_ih_thumb_proximal_yaw": 0.6,
                "r_ih_thumb_proximal_pitch": 0.4}),  # holding container
        (0.8,  {"right_shoulder_pitch": _d(-80), "right_shoulder_roll": _d(-30),
                "right_elbow": _d(70)}),  # raise arm
        (1.5,  {"right_wrist_roll": _d(-100)}),  # tilt to pour
        (2.8,  {}),  # hold pouring
        (3.4,  {"right_wrist_roll": 0}),  # upright
        (4.0,  {**R_FINGERS_OPEN,
                "r_ih_thumb_proximal_yaw": 0, "r_ih_thumb_proximal_pitch": 0,
                "right_shoulder_pitch": 0, "right_shoulder_roll": 0,
                "right_elbow": 0}),
    ],

    "open_and_close_right_hand": [
        (0.0,  {**R_FINGERS_OPEN, "r_ih_thumb_proximal_yaw": 0,
                "r_ih_thumb_proximal_pitch": 0}),
        (0.5,  {**R_FINGERS_CLOSE,
                "r_ih_thumb_proximal_yaw": 0.8, "r_ih_thumb_proximal_pitch": 0.5}),
        (1.0,  {**R_FINGERS_OPEN, "r_ih_thumb_proximal_yaw": 0,
                "r_ih_thumb_proximal_pitch": 0}),
        (1.5,  {**R_FINGERS_CLOSE,
                "r_ih_thumb_proximal_yaw": 0.8, "r_ih_thumb_proximal_pitch": 0.5}),
        (2.0,  {**R_FINGERS_OPEN, "r_ih_thumb_proximal_yaw": 0,
                "r_ih_thumb_proximal_pitch": 0}),
    ],

    "open_and_close_left_hand": [
        (0.0,  {**FINGERS_OPEN, "l_ih_thumb_proximal_yaw": 0,
                "l_ih_thumb_proximal_pitch": 0}),
        (0.5,  {**FINGERS_CLOSE,
                "l_ih_thumb_proximal_yaw": 0.8, "l_ih_thumb_proximal_pitch": 0.5}),
        (1.0,  {**FINGERS_OPEN, "l_ih_thumb_proximal_yaw": 0,
                "l_ih_thumb_proximal_pitch": 0}),
        (1.5,  {**FINGERS_CLOSE,
                "l_ih_thumb_proximal_yaw": 0.8, "l_ih_thumb_proximal_pitch": 0.5}),
        (2.0,  {**FINGERS_OPEN, "l_ih_thumb_proximal_yaw": 0,
                "l_ih_thumb_proximal_pitch": 0}),
    ],

    "open_and_close_both_hands": [
        (0.0,  {}),
        (0.5,  {**FINGERS_CLOSE, "l_ih_thumb_proximal_yaw": 0.8,
                "l_ih_thumb_proximal_pitch": 0.5,
                **R_FINGERS_CLOSE, "r_ih_thumb_proximal_yaw": 0.8,
                "r_ih_thumb_proximal_pitch": 0.5}),
        (1.0,  {**FINGERS_OPEN, "l_ih_thumb_proximal_yaw": 0,
                "l_ih_thumb_proximal_pitch": 0,
                **R_FINGERS_OPEN, "r_ih_thumb_proximal_yaw": 0,
                "r_ih_thumb_proximal_pitch": 0}),
        (1.5,  {**FINGERS_CLOSE, "l_ih_thumb_proximal_yaw": 0.8,
                "l_ih_thumb_proximal_pitch": 0.5,
                **R_FINGERS_CLOSE, "r_ih_thumb_proximal_yaw": 0.8,
                "r_ih_thumb_proximal_pitch": 0.5}),
        (2.0,  {**FINGERS_OPEN, "l_ih_thumb_proximal_yaw": 0,
                "l_ih_thumb_proximal_pitch": 0,
                **R_FINGERS_OPEN, "r_ih_thumb_proximal_yaw": 0,
                "r_ih_thumb_proximal_pitch": 0}),
    ],

    # -----------------------------------------------------------------------
    # TOOLS / OCCUPATIONAL
    # -----------------------------------------------------------------------

    "type_keyboard_both_hands": [
        # Starting: both arms in typing position
        (0.0,  {"left_shoulder_pitch":  _d(-50), "left_shoulder_roll":  _d(10),
                "left_elbow":  _d(75), "left_wrist_pitch": _d(20),
                "right_shoulder_pitch": _d(-50), "right_shoulder_roll": _d(-10),
                "right_elbow": _d(75), "right_wrist_pitch": _d(20)}),
        # Finger presses: left index
        (0.25, {"l_ih_index_proximal": 0.8}),
        (0.45, {"l_ih_index_proximal": 0.0}),
        # Right middle
        (0.55, {"r_ih_middle_proximal": 0.8}),
        (0.75, {"r_ih_middle_proximal": 0.0}),
        # Left middle
        (0.9,  {"l_ih_middle_proximal": 0.8}),
        (1.1,  {"l_ih_middle_proximal": 0.0}),
        # Right index
        (1.2,  {"r_ih_index_proximal": 0.8}),
        (1.4,  {"r_ih_index_proximal": 0.0}),
        # Left ring
        (1.55, {"l_ih_ring_proximal": 0.7}),
        (1.75, {"l_ih_ring_proximal": 0.0}),
        # Right ring
        (1.9,  {"r_ih_ring_proximal": 0.7}),
        (2.1,  {"r_ih_ring_proximal": 0.0}),
        # Left pinky
        (2.25, {"l_ih_pinky_proximal": 0.6}),
        (2.45, {"l_ih_pinky_proximal": 0.0}),
        # Right pinky
        (2.6,  {"r_ih_pinky_proximal": 0.6}),
        (2.8,  {"r_ih_pinky_proximal": 0.0}),
        # Return
        (3.4,  {"left_shoulder_pitch": 0, "left_shoulder_roll": 0,
                "left_elbow": 0, "left_wrist_pitch": 0,
                "right_shoulder_pitch": 0, "right_shoulder_roll": 0,
                "right_elbow": 0, "right_wrist_pitch": 0}),
    ],

    "hold_phone_to_ear_right": [
        (0.0,  {}),
        (0.7,  {"right_shoulder_pitch": _d(-115), "right_shoulder_roll": _d(-40),
                "right_elbow": _d(120), "right_wrist_pitch": _d(-20),
                "right_wrist_yaw": _d(20)}),
        (1.0,  {"r_ih_index_proximal": 0.6, "r_ih_middle_proximal": 0.6,
                "r_ih_ring_proximal": 0.6, "r_ih_pinky_proximal": 0.6,
                "r_ih_thumb_proximal_yaw": 0.3}),  # phone grip
        (3.5,  {}),  # talking
        (4.3,  {"right_shoulder_pitch": 0, "right_shoulder_roll": 0,
                "right_elbow": 0, "right_wrist_pitch": 0, "right_wrist_yaw": 0,
                **R_FINGERS_OPEN, "r_ih_thumb_proximal_yaw": 0}),
    ],

    "use_screwdriver_right": [
        (0.0,  {}),
        (0.7,  {"right_shoulder_pitch": _d(-65), "right_shoulder_roll": _d(-15),
                "right_elbow": _d(80), "right_wrist_pitch": _d(10)}),
        (1.0,  {"r_ih_index_proximal": 0.9, "r_ih_middle_proximal": 1.0,
                "r_ih_ring_proximal": 1.0, "r_ih_pinky_proximal": 0.8,
                "r_ih_thumb_proximal_yaw": 0.7, "r_ih_thumb_proximal_pitch": 0.5}),
        # Clockwise rotations of wrist
        (1.3,  {"right_wrist_roll": _d(90)}),
        (1.6,  {"right_wrist_roll": _d(0)}),
        (1.9,  {"right_wrist_roll": _d(90)}),
        (2.2,  {"right_wrist_roll": _d(0)}),
        (2.5,  {"right_wrist_roll": _d(90)}),
        (2.8,  {"right_wrist_roll": _d(0)}),
        (3.4,  {**R_FINGERS_OPEN,
                "r_ih_thumb_proximal_yaw": 0, "r_ih_thumb_proximal_pitch": 0,
                "right_shoulder_pitch": 0, "right_shoulder_roll": 0,
                "right_elbow": 0, "right_wrist_pitch": 0,
                "right_wrist_roll": 0}),
    ],

    "give_object_to_person_right": [
        (0.0,  {**R_FINGERS_CLOSE,
                "r_ih_thumb_proximal_yaw": 0.6,
                "r_ih_thumb_proximal_pitch": 0.4}),  # holding object
        (0.8,  {"right_shoulder_pitch": _d(-70), "right_shoulder_roll": _d(-20),
                "right_elbow": _d(60)}),  # extend arm to offer
        (1.8,  {}),  # hold/wait
        (2.4,  {**R_FINGERS_OPEN,
                "r_ih_thumb_proximal_yaw": 0,
                "r_ih_thumb_proximal_pitch": 0}),  # release
        (3.2,  {"right_shoulder_pitch": 0, "right_shoulder_roll": 0,
                "right_elbow": 0}),
    ],

    # -----------------------------------------------------------------------
    # STANDING NEUTRAL (useful as a reference/reset pose)
    # -----------------------------------------------------------------------

    "standing_neutral": [
        (0.0, {}),
        (2.0, {}),
    ],
}

# ---------------------------------------------------------------------------
# Dataset categories → folder names
# ---------------------------------------------------------------------------

TASK_CATEGORIES: Dict[str, str] = {
    "wave_hello_right":           "g1_social_expressive",
    "wave_hello_left":            "g1_social_expressive",
    "clap_hands":                 "g1_social_expressive",
    "thumbs_up_right":            "g1_social_expressive",
    "point_at_object_right":      "g1_social_expressive",
    "handshake_right":            "g1_social_expressive",
    "reach_and_grasp_right":      "g1_manipulation",
    "reach_and_grasp_left":       "g1_manipulation",
    "bimanual_box_grasp":         "g1_manipulation",
    "pick_and_place_right":       "g1_manipulation",
    "press_button_right_index":   "g1_manipulation",
    "pour_liquid_right":          "g1_manipulation",
    "open_and_close_right_hand":  "g1_hand_gestures",
    "open_and_close_left_hand":   "g1_hand_gestures",
    "open_and_close_both_hands":  "g1_hand_gestures",
    "type_keyboard_both_hands":   "g1_tool_use",
    "hold_phone_to_ear_right":    "g1_tool_use",
    "use_screwdriver_right":      "g1_tool_use",
    "give_object_to_person_right":"g1_manipulation",
    "standing_neutral":           "g1_reference",
}

# ---------------------------------------------------------------------------
# Motion generation
# ---------------------------------------------------------------------------

FPS = 50  # frames per second


def _cubic_hermite(t: np.ndarray, t0: float, t1: float,
                   v0: float, v1: float) -> np.ndarray:
    """Smooth cubic Hermite interpolation between v0→v1 over [t0,t1]."""
    if abs(t1 - t0) < 1e-9:
        return np.full_like(t, v1)
    s = (t - t0) / (t1 - t0)
    s = np.clip(s, 0, 1)
    # smooth-step: 3s²−2s³
    blend = s * s * (3 - 2 * s)
    return v0 + (v1 - v0) * blend


def generate_motion(
    keyframes: List[Tuple[float, Dict[str, float]]],
    fps: int = FPS,
) -> np.ndarray:
    """
    Interpolate keyframes into a dense (N_frames × 41) array of joint angles.
    Joints not mentioned in any keyframe stay at 0.
    """
    if not keyframes:
        raise ValueError("No keyframes provided")

    t_end = keyframes[-1][0]
    n_frames = max(2, int(round(t_end * fps)) + 1)
    times = np.linspace(0, t_end, n_frames)

    # Build per-joint time-value pairs from keyframes
    joint_timeseries: Dict[str, List[Tuple[float, float]]] = {
        name: [] for name in ACTUATOR_NAMES
    }

    for t, jdict in keyframes:
        for name in ACTUATOR_NAMES:
            if name in jdict:
                joint_timeseries[name].append((t, jdict[name]))

    # For any joint that has no keyframes at all, fix at 0
    data = np.zeros((n_frames, len(ACTUATOR_NAMES)))

    for col_idx, name in enumerate(ACTUATOR_NAMES):
        series = joint_timeseries[name]
        if not series:
            # stays 0
            continue

        # Ensure we start and end at known values
        # Prepend 0 at t=0 if no keyframe at t=0
        if series[0][0] > 1e-6:
            series = [(0.0, 0.0)] + series
        # Append last value at t_end if needed
        if series[-1][0] < t_end - 1e-6:
            series = series + [(t_end, series[-1][1])]

        # Piecewise cubic Hermite interpolation between segments
        signal = np.zeros(n_frames)
        for seg in range(len(series) - 1):
            t0, v0 = series[seg]
            t1, v1 = series[seg + 1]
            mask = (times >= t0) & (times <= t1)
            signal[mask] = _cubic_hermite(times[mask], t0, t1, v0, v1)
        # fill before and after with endpoint values
        signal[times < series[0][0]] = series[0][1]
        signal[times > series[-1][0]] = series[-1][1]

        # Clamp to joint limits
        lo, hi = JOINT_LIMITS[name]
        signal = np.clip(signal, lo, hi)
        data[:, col_idx] = signal

    return data


def write_csv(path: Path, data: np.ndarray) -> None:
    """Write (N × 41) array to CSV with header row."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(ACTUATOR_NAMES)
        for row in data:
            writer.writerow([f"{v:.6f}" for v in row])


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate G1 + Inspire Hand motion datasets"
    )
    parser.add_argument(
        "--task", default=None,
        help="Single task name to generate (default: all tasks)",
    )
    parser.add_argument(
        "--list", action="store_true",
        help="List all available tasks and exit",
    )
    parser.add_argument(
        "--out_dir", default=None,
        help="Output root directory (default: same folder as this script)",
    )
    args = parser.parse_args()

    if args.list:
        print(f"{'Task':<40} {'Category':<30} {'~Frames'}")
        print("-" * 85)
        for task_name, kfs in TASKS.items():
            t_total = kfs[-1][0]
            approx_frames = int(round(t_total * FPS))
            cat = TASK_CATEGORIES.get(task_name, "uncategorized")
            print(f"{task_name:<40} {cat:<30} ~{approx_frames}")
        return

    out_root = Path(args.out_dir) if args.out_dir else Path(__file__).parent

    tasks_to_run = (
        {args.task: TASKS[args.task]} if args.task else TASKS
    )
    if args.task and args.task not in TASKS:
        print(f"ERROR: Unknown task '{args.task}'. Use --list to see options.")
        return

    generated = []
    for task_name, keyframes in tasks_to_run.items():
        category = TASK_CATEGORIES.get(task_name, "uncategorized")
        csv_name = task_name.replace(" ", "_") + ".csv"
        out_path = out_root / category / csv_name

        data = generate_motion(keyframes)
        write_csv(out_path, data)
        generated.append((task_name, out_path, len(data)))
        print(f"  ✓ {task_name:<42} {len(data):>5} frames → {out_path.relative_to(out_root)}")

    print(f"\nGenerated {len(generated)} dataset(s).")
    print(f"Output root: {out_root}")
    print(f"\nCSV format: {len(ACTUATOR_NAMES)} columns (joint names in row 0), values in radians.")


if __name__ == "__main__":
    main()
