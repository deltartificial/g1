"""
G1 Bipedal Walking Evaluation Script

Loads a trained policy and runs it in visualization mode.
Supports both rsl_rl and stable-baselines3 checkpoints.

Adapted from Genesis examples/locomotion/go2_eval.py.

Usage:
    python walk/g1_eval.py                           # Default: latest checkpoint
    python walk/g1_eval.py -e g1-walking --ckpt 500  # Specific checkpoint
    python walk/g1_eval.py --cmd_vel 0.5 0.0 0.0     # Custom velocity command
    python walk/g1_eval.py --debug                   # Debug mode with joint info
"""

import argparse
import os
import pickle
import sys

import torch
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Joint names for G1 legs (12 DOFs)
LEG_JOINT_NAMES = [
    "L_hip_pitch", "L_hip_roll", "L_hip_yaw", "L_knee", "L_ankle_pitch", "L_ankle_roll",
    "R_hip_pitch", "R_hip_roll", "R_hip_yaw", "R_knee", "R_ankle_pitch", "R_ankle_roll",
]


def print_debug_state(env, actions, step):
    """Print detailed robot state for debugging."""
    pos = env.base_pos[0].cpu().numpy()
    quat = env.base_quat[0].cpu().numpy()
    lin_vel = env.base_lin_vel[0].cpu().numpy()
    ang_vel = env.base_ang_vel[0].cpu().numpy()
    dof_pos = env.dof_pos[0].cpu().numpy()
    dof_vel = env.dof_vel[0].cpu().numpy()
    proj_grav = env.projected_gravity[0].cpu().numpy()

    actions_np = actions[0].cpu().numpy() if isinstance(actions, torch.Tensor) else actions[0]

    print(f"\n{'─' * 70}")
    print(f"Step {step:4d} | Pos=({pos[0]:+.2f}, {pos[1]:+.2f}, {pos[2]:.2f})")
    print(f"         | LinVel=({lin_vel[0]:+.2f}, {lin_vel[1]:+.2f}, {lin_vel[2]:+.2f}) m/s")
    print(f"         | AngVel=({ang_vel[0]:+.2f}, {ang_vel[1]:+.2f}, {ang_vel[2]:+.2f}) rad/s")
    print(f"         | ProjGrav=({proj_grav[0]:+.2f}, {proj_grav[1]:+.2f}, {proj_grav[2]:+.2f})")

    # Joint info
    action_mag = np.sqrt(np.sum(actions_np ** 2))
    max_idx = np.argmax(np.abs(actions_np))
    max_name = LEG_JOINT_NAMES[max_idx] if max_idx < len(LEG_JOINT_NAMES) else f"j{max_idx}"

    print(f"         | Action mag={action_mag:.3f}, max={max_name}({actions_np[max_idx]:+.3f})")

    # Moving joints
    moving = np.where(np.abs(dof_vel) > 0.1)[0]
    if len(moving) > 0:
        names = [LEG_JOINT_NAMES[i] if i < len(LEG_JOINT_NAMES) else f"j{i}" for i in moving[:6]]
        print(f"         | Moving: {', '.join(names)}" + ("..." if len(moving) > 6 else ""))
    else:
        print(f"         | Moving: NONE (robot statique!)")

    # Left vs Right leg activity
    left_vel = np.mean(np.abs(dof_vel[:6]))
    right_vel = np.mean(np.abs(dof_vel[6:]))
    print(f"         | L_leg={left_vel:.2f} R_leg={right_vel:.2f} rad/s")

# Try rsl_rl first
RSL_RL_AVAILABLE = False
try:
    from rsl_rl.runners import OnPolicyRunner
    RSL_RL_AVAILABLE = True
except ImportError:
    pass

import genesis as gs


def find_latest_checkpoint(log_dir):
    """Find the highest numbered checkpoint in log directory."""
    checkpoints = []
    for f in os.listdir(log_dir):
        # rsl_rl format: model_100.pt
        if f.startswith("model_") and f.endswith(".pt"):
            try:
                num = int(f.replace("model_", "").replace(".pt", ""))
                checkpoints.append(("rsl_rl", num, f))
            except ValueError:
                continue
        # SB3 format: model_499712_steps.zip
        elif f.startswith("model_") and f.endswith("_steps.zip"):
            try:
                num = int(f.replace("model_", "").replace("_steps.zip", ""))
                checkpoints.append(("sb3", num, f))
            except ValueError:
                continue
    if not checkpoints:
        return None
    # Return the highest numbered checkpoint
    latest = max(checkpoints, key=lambda x: x[1])
    return latest[1]  # Return the number


def detect_checkpoint_type(log_dir, ckpt):
    """Detect if checkpoint is rsl_rl or sb3 format."""
    # rsl_rl format
    rsl_path = os.path.join(log_dir, f"model_{ckpt}.pt")
    if os.path.exists(rsl_path):
        return "rsl_rl", rsl_path

    # SB3 format: model_{steps}_steps.zip
    sb3_path = os.path.join(log_dir, f"model_{ckpt}_steps.zip")
    if os.path.exists(sb3_path):
        return "sb3", sb3_path

    # SB3 final model
    final_path = os.path.join(log_dir, "model_final.zip")
    if os.path.exists(final_path):
        return "sb3", final_path

    # Search for any matching SB3 checkpoint
    for f in os.listdir(log_dir):
        if f.startswith("model_") and f.endswith("_steps.zip"):
            return "sb3", os.path.join(log_dir, f)

    return None, None


def main():
    parser = argparse.ArgumentParser(description="Evaluate G1 walking policy")
    parser.add_argument("-e", "--exp_name", type=str, default="g1-walking",
                        help="Experiment name")
    parser.add_argument("--ckpt", type=int, default=None,
                        help="Checkpoint iteration (default: latest)")
    parser.add_argument("--device", type=str, default="mps",
                        help="Device (cuda, mps, cpu)")
    parser.add_argument("--cmd_vel", type=float, nargs=3, default=None,
                        help="Override velocity command [vx, vy, vyaw]")
    parser.add_argument("--num_steps", type=int, default=None,
                        help="Number of steps to run (default: infinite)")
    parser.add_argument("--debug", "-d", action="store_true",
                        help="Debug mode: print joint positions/velocities")
    args = parser.parse_args()

    gs.init()

    log_dir = os.path.join(os.path.dirname(__file__), "logs", args.exp_name)

    if not os.path.exists(log_dir):
        print(f"Error: Log directory not found: {log_dir}")
        print("Have you trained a model first? Run: python walk/g1_train.py")
        return

    # Find checkpoint
    ckpt = args.ckpt
    if ckpt is None:
        ckpt = find_latest_checkpoint(log_dir)
        if ckpt is None:
            # Check for sb3 final model
            if os.path.exists(os.path.join(log_dir, "model_final.zip")):
                ckpt = "final"
            else:
                print(f"Error: No checkpoints found in {log_dir}")
                return
        print(f"Using checkpoint: {ckpt}")

    # Load configs
    cfg_path = os.path.join(log_dir, "cfgs.pkl")
    if not os.path.exists(cfg_path):
        print(f"Error: Config file not found: {cfg_path}")
        return

    env_cfg, obs_cfg, reward_cfg, command_cfg, train_cfg = pickle.load(open(cfg_path, "rb"))

    # Override command if specified
    if args.cmd_vel is not None:
        command_cfg["lin_vel_x_range"] = [args.cmd_vel[0], args.cmd_vel[0]]
        command_cfg["lin_vel_y_range"] = [args.cmd_vel[1], args.cmd_vel[1]]
        command_cfg["ang_vel_range"] = [args.cmd_vel[2], args.cmd_vel[2]]

    # Disable reward computation for faster eval
    reward_cfg["reward_scales"] = {}

    # Create environment with viewer
    from g1_env import G1Env
    env = G1Env(
        num_envs=1,
        env_cfg=env_cfg,
        obs_cfg=obs_cfg,
        reward_cfg=reward_cfg,
        command_cfg=command_cfg,
        show_viewer=True,
        device=args.device,
    )

    # Detect checkpoint type and load policy
    ckpt_type, ckpt_path = detect_checkpoint_type(log_dir, ckpt)

    if ckpt_type is None:
        # Try direct path
        direct_path = os.path.join(log_dir, f"model_{ckpt}.pt")
        if os.path.exists(direct_path):
            ckpt_type, ckpt_path = "rsl_rl", direct_path
        else:
            print(f"Error: Checkpoint not found for iteration {ckpt}")
            return

    print(f"\n{'='*60}")
    print(f"G1 Bipedal Walking Evaluation")
    print(f"{'='*60}")
    print(f"  Experiment: {args.exp_name}")
    print(f"  Checkpoint: {ckpt} ({ckpt_type})")
    print(f"  Command: vx={command_cfg['lin_vel_x_range'][0]:.2f}, "
          f"vy={command_cfg['lin_vel_y_range'][0]:.2f}, "
          f"vyaw={command_cfg['ang_vel_range'][0]:.2f}")
    print(f"{'='*60}\n")

    # Load policy based on type
    if ckpt_type == "rsl_rl":
        if not RSL_RL_AVAILABLE:
            print("Error: rsl_rl not installed but checkpoint is rsl_rl format")
            return
        runner = OnPolicyRunner(env, train_cfg, log_dir, device=args.device)
        runner.load(ckpt_path)
        policy = runner.get_inference_policy(device=args.device)

        # Run with rsl_rl policy
        obs, _ = env.reset()
        step_count = 0
        total_reward = 0

        with torch.no_grad():
            while True:
                actions = policy(obs)
                obs, rews, dones, infos = env.step(actions)
                total_reward += rews.sum().item()
                step_count += 1

                if args.debug and step_count % 10 == 0:
                    print_debug_state(env, actions, step_count)

                if dones.any():
                    print(f"\n{'=' * 40}")
                    print(f"Episode done - Total reward: {total_reward:.1f}")
                    print(f"{'=' * 40}")
                    obs, _ = env.reset()
                    total_reward = 0

                if args.num_steps and step_count >= args.num_steps:
                    break
    else:  # sb3
        from stable_baselines3 import PPO
        from g1_env_sb3 import G1EnvSB3Wrapper

        wrapped_env = G1EnvSB3Wrapper(env)
        model = PPO.load(ckpt_path, env=wrapped_env, device=args.device)

        # Run with sb3 policy
        obs = wrapped_env.reset()
        step_count = 0
        total_reward = 0

        while True:
            actions, _ = model.predict(obs, deterministic=True)
            obs, rewards, dones, infos = wrapped_env.step(actions)
            total_reward += rewards.sum()
            step_count += 1

            if args.debug and step_count % 10 == 0:
                print_debug_state(env, actions, step_count)

            if dones.any():
                print(f"\n{'=' * 40}")
                print(f"Episode done - Total reward: {total_reward:.1f}")
                print(f"{'=' * 40}")
                obs = wrapped_env.reset()
                total_reward = 0

            if args.num_steps and step_count >= args.num_steps:
                break


if __name__ == "__main__":
    main()
