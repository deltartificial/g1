"""
G1 Bipedal Walking Training Script

Uses rsl_rl PPO implementation with Genesis physics engine.
Falls back to stable-baselines3 if rsl_rl is not available.

Adapted from Genesis examples/locomotion/go2_train.py for bipedal locomotion.

Usage:
    python walk/g1_train.py                          # Default 4096 envs, 1000 iterations
    python walk/g1_train.py -B 2048 --max_iter 500   # Custom settings
    python walk/g1_train.py -e my_experiment         # Custom experiment name
    python walk/g1_train.py --backend sb3            # Force stable-baselines3
"""

import argparse
import os
import pickle
import shutil
import sys

# Add parent directory for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Try rsl_rl first, fallback to stable-baselines3
RSL_RL_AVAILABLE = False
try:
    from rsl_rl.runners import OnPolicyRunner
    RSL_RL_AVAILABLE = True
except ImportError:
    pass

import torch
import genesis as gs


def get_train_cfg(exp_name, max_iterations):
    """PPO training configuration optimized for bipedal locomotion (rsl_rl v3.x format)."""
    train_cfg_dict = {
        # Algorithm config (PPO)
        "algorithm": {
            "class_name": "PPO",
            "clip_param": 0.2,
            "desired_kl": 0.01,
            "entropy_coef": 0.01,
            "gamma": 0.99,
            "lam": 0.95,
            "learning_rate": 3e-4,
            "max_grad_norm": 1.0,
            "num_learning_epochs": 5,
            "num_mini_batches": 4,
            "schedule": "adaptive",
            "use_clipped_value_loss": True,
            "value_loss_coef": 1.0,
        },
        # Policy config (ActorCritic)
        "policy": {
            "class_name": "ActorCritic",
            "activation": "elu",
            "actor_hidden_dims": [512, 256, 128],
            "critic_hidden_dims": [512, 256, 128],
            "init_noise_std": 1.0,
        },
        # Observation groups (required by rsl_rl v3.x)
        "obs_groups": {},
        # Runner config (root level for v3.x)
        "seed": 1,
        "num_steps_per_env": 24,
        "save_interval": 100,
        "max_iterations": max_iterations,
        "experiment_name": exp_name,
        "log_interval": 1,
    }

    return train_cfg_dict


def get_cfgs():
    """
    Environment and reward configuration for G1 bipedal walking.

    Key considerations for bipeds vs quadrupeds:
    - Higher base height (0.75m vs 0.3m for Go2)
    - Tighter termination thresholds (less stable)
    - Emphasis on orientation rewards (critical for balance)
    - Gait phase observation for walking pattern
    """
    # Path to G1 robot MJCF
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    robot_path = os.path.join(base_dir, "..", "unitree_mujoco", "unitree_robots", "g1", "g1_29dof.xml")

    env_cfg = {
        "num_actions": 12,  # Leg joints only (upper body PD-locked)
        "robot_path": robot_path,

        # PD gains for legs (RL-controlled)
        "kp": 150.0,
        "kd": 15.0,

        # PD gains for upper body (locked for balance)
        "upper_body_kp": 300.0,  # Higher stiffness to prevent torso collapse
        "upper_body_kd": 30.0,

        # Termination conditions - TIGHTER for faster learning
        "termination_if_roll_greater_than": 0.3,   # ~17 degrees (was 0.5)
        "termination_if_pitch_greater_than": 0.3,  # ~17 degrees (was 0.5)
        "termination_if_height_lower_than": 0.5,   # Higher threshold (was 0.4)

        # Base initialization - lower with bent knees (KNEES_BENT_KEYFRAME)
        "base_init_pos": [0.0, 0.0, 0.76],  # Lower for stability (was 0.793)
        "base_init_quat": [1.0, 0.0, 0.0, 0.0],  # wxyz format

        # Episode settings
        "episode_length_s": 20.0,
        "resampling_time_s": 8.0,  # Longer for learning stable gait

        # Action scaling (conservative for bipeds)
        "action_scale": 0.25,
        "simulate_action_latency": True,
        "clip_actions": 10.0,

        # Simulation
        "dt": 0.02,
        "substeps": 4,
    }

    obs_cfg = {
        # 3 + 3 + 3 + 12 + 12 + 12 + 2 = 47
        "num_obs": 47,
        "obs_scales": {
            "lin_vel": 2.0,
            "ang_vel": 0.25,
            "dof_pos": 1.0,
            "dof_vel": 0.05,
        },
    }

    reward_cfg = {
        "tracking_sigma": 0.25,  # Tighter tracking (was 0.5)
        "base_height_target": 0.72,  # Lower target for bent-knee stance

        "reward_scales": {
            # Primary: Velocity tracking (now that we have balance)
            "tracking_lin_vel": 1.5,   # Main reward source
            "tracking_ang_vel": 0.5,

            # Balance rewards
            "alive": 1.0,              # Survival bonus
            "orientation": -10.0,      # Tilt penalty (reduced, upper body helps)
            "base_height": -30.0,      # Height penalty
            "ang_vel_xy": -0.1,        # Reduce wobbling
            "lin_vel_z": -2.0,         # Reduce bouncing

            # Gait quality
            "feet_air_time": 0.5,      # Encourage alternating feet
            "symmetry": 0.2,           # Symmetric leg motion

            # Regularization
            "action_rate": -0.01,      # Smooth actions
            "similar_to_default": -0.1, # Don't drift too far
            "dof_vel": -0.0001,
            "dof_acc": -1e-7,
            "torques": -0.0001,
        },
    }

    command_cfg = {
        "num_commands": 3,
        # Walking velocity commands
        "lin_vel_x_range": [0.2, 0.5],  # Forward walking
        "lin_vel_y_range": [-0.1, 0.1], # Slight lateral
        "ang_vel_range": [-0.2, 0.2],   # Slight turning
    }

    return env_cfg, obs_cfg, reward_cfg, command_cfg


def train_with_rsl_rl(env, train_cfg, log_dir, device, max_iter):
    """Train using rsl_rl OnPolicyRunner (Genesis style)."""
    runner = OnPolicyRunner(env, train_cfg, log_dir, device=device)
    runner.learn(num_learning_iterations=max_iter, init_at_random_ep_len=True)


def train_with_sb3(env, log_dir, max_iter, device):
    """Train using stable-baselines3 PPO (fallback)."""
    from stable_baselines3 import PPO
    from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback

    # Create SB3-compatible wrapper
    from g1_env_sb3 import G1EnvSB3Wrapper
    wrapped_env = G1EnvSB3Wrapper(env)

    model = PPO(
        "MlpPolicy",
        wrapped_env,
        learning_rate=3e-4,
        n_steps=24,
        batch_size=256,
        n_epochs=5,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        vf_coef=1.0,
        max_grad_norm=1.0,
        policy_kwargs=dict(
            net_arch=dict(pi=[512, 256, 128], vf=[512, 256, 128]),
            activation_fn=torch.nn.ELU,
        ),
        verbose=1,
        device=device,
    )

    # Save every ~500k timesteps (save_freq is per env)
    # 500k / num_envs = steps per env
    save_freq_steps = max(500_000 // wrapped_env.num_envs, 100)
    checkpoint_callback = CheckpointCallback(
        save_freq=save_freq_steps,
        save_path=log_dir,
        name_prefix="model",
    )

    total_timesteps = max_iter * 24 * wrapped_env.num_envs
    model.learn(total_timesteps=total_timesteps, callback=checkpoint_callback)
    model.save(os.path.join(log_dir, "model_final"))


def main():
    parser = argparse.ArgumentParser(description="Train G1 bipedal walking policy")
    parser.add_argument("-e", "--exp_name", type=str, default="g1-walking",
                        help="Experiment name for logging")
    parser.add_argument("-B", "--num_envs", type=int, default=4096,
                        help="Number of parallel environments")
    parser.add_argument("--max_iter", type=int, default=1000,
                        help="Maximum training iterations")
    parser.add_argument("--device", type=str, default="mps",
                        help="Device (cuda, mps, cpu)")
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to checkpoint to resume from")
    parser.add_argument("--backend", type=str, choices=["rsl_rl", "sb3", "auto"], default="auto",
                        help="Training backend: rsl_rl, sb3, or auto (default)")
    args = parser.parse_args()

    # Determine backend
    use_rsl_rl = False
    if args.backend == "rsl_rl":
        if not RSL_RL_AVAILABLE:
            print("Error: rsl_rl not installed. Install with:")
            print("  pip install git+https://github.com/leggedrobotics/rsl_rl.git")
            return
        use_rsl_rl = True
    elif args.backend == "sb3":
        use_rsl_rl = False
    else:  # auto
        use_rsl_rl = RSL_RL_AVAILABLE

    gs.init(logging_level="warning")

    log_dir = os.path.join(os.path.dirname(__file__), "checkpoints")
    env_cfg, obs_cfg, reward_cfg, command_cfg = get_cfgs()
    train_cfg = get_train_cfg(args.exp_name, args.max_iter)

    # Handle resume
    if args.resume:
        train_cfg["runner"]["resume"] = True
        train_cfg["runner"]["resume_path"] = args.resume
    else:
        if os.path.exists(log_dir):
            shutil.rmtree(log_dir)
        os.makedirs(log_dir, exist_ok=True)

    # Create environment
    from g1_env import G1Env
    env = G1Env(
        num_envs=args.num_envs,
        env_cfg=env_cfg,
        obs_cfg=obs_cfg,
        reward_cfg=reward_cfg,
        command_cfg=command_cfg,
        device=args.device,
    )

    # Save configs
    pickle.dump(
        [env_cfg, obs_cfg, reward_cfg, command_cfg, train_cfg],
        open(os.path.join(log_dir, "cfgs.pkl"), "wb"),
    )

    backend_name = "rsl_rl" if use_rsl_rl else "stable-baselines3"
    print(f"\n{'='*60}")
    print(f"G1 Bipedal Walking Training")
    print(f"{'='*60}")
    print(f"  Experiment: {args.exp_name}")
    print(f"  Backend: {backend_name}")
    print(f"  Environments: {args.num_envs}")
    print(f"  Max iterations: {args.max_iter}")
    print(f"  Device: {args.device}")
    print(f"  Log directory: {log_dir}")
    print(f"{'='*60}\n")

    # Train
    if use_rsl_rl:
        train_with_rsl_rl(env, train_cfg, log_dir, args.device, args.max_iter)
    else:
        train_with_sb3(env, log_dir, args.max_iter, args.device)


if __name__ == "__main__":
    main()
