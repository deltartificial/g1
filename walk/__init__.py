"""
G1 Bipedal Walking Module

A clean implementation of bipedal locomotion for the Unitree G1 humanoid robot,
adapted from the Genesis Go2 quadruped locomotion example.

Files:
    g1_env.py       - Genesis-style environment with modular rewards
    g1_train.py     - Training script (supports rsl_rl and stable-baselines3)
    g1_eval.py      - Evaluation/visualization script
    g1_env_sb3.py   - SB3 VecEnv wrapper for g1_env

Usage:
    # Training
    python walk/g1_train.py -B 4096 --max_iter 1000

    # Evaluation
    python walk/g1_eval.py --ckpt 500 --cmd_vel 0.5 0.0 0.0

Key Features:
    - Modular reward functions (easy to tune)
    - Gait phase signal for periodic walking
    - Projected gravity observation (critical for balance)
    - Support for both rsl_rl and stable-baselines3

Differences from Go2 (quadruped):
    - Higher center of mass (0.79m vs 0.42m)
    - 12 leg DOFs (vs 12 for Go2, but different kinematics)
    - Stricter termination conditions (bipeds are less stable)
    - Symmetry rewards for natural gait
    - Phase-based hip motion for alternating stance/swing
"""

from .g1_env import G1Env

__all__ = ["G1Env"]
