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

# Full joint names matching G1Env.ALL_DOF_NAMES order
ALL_JOINT_NAMES = [
    # Legs (12)
    "left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint",
    "left_knee_joint", "left_ankle_pitch_joint", "left_ankle_roll_joint",
    "right_hip_pitch_joint", "right_hip_roll_joint", "right_hip_yaw_joint",
    "right_knee_joint", "right_ankle_pitch_joint", "right_ankle_roll_joint",
    # Waist (3)
    "waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint",
    # Left arm (7)
    "left_shoulder_pitch_joint", "left_shoulder_roll_joint", "left_shoulder_yaw_joint",
    "left_elbow_joint", "left_wrist_roll_joint", "left_wrist_pitch_joint", "left_wrist_yaw_joint",
    # Right arm (7)
    "right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint",
    "right_elbow_joint", "right_wrist_roll_joint", "right_wrist_pitch_joint", "right_wrist_yaw_joint",
]

# Short names for compact display
SHORT_NAMES = {
    "left_hip_pitch_joint": "L_hip_p", "left_hip_roll_joint": "L_hip_r", "left_hip_yaw_joint": "L_hip_y",
    "left_knee_joint": "L_knee", "left_ankle_pitch_joint": "L_ank_p", "left_ankle_roll_joint": "L_ank_r",
    "right_hip_pitch_joint": "R_hip_p", "right_hip_roll_joint": "R_hip_r", "right_hip_yaw_joint": "R_hip_y",
    "right_knee_joint": "R_knee", "right_ankle_pitch_joint": "R_ank_p", "right_ankle_roll_joint": "R_ank_r",
    "waist_yaw_joint": "waist_y", "waist_roll_joint": "waist_r", "waist_pitch_joint": "waist_p",
    "left_shoulder_pitch_joint": "L_sh_p", "left_shoulder_roll_joint": "L_sh_r", "left_shoulder_yaw_joint": "L_sh_y",
    "left_elbow_joint": "L_elbow", "left_wrist_roll_joint": "L_wr_r", "left_wrist_pitch_joint": "L_wr_p", "left_wrist_yaw_joint": "L_wr_y",
    "right_shoulder_pitch_joint": "R_sh_p", "right_shoulder_roll_joint": "R_sh_r", "right_shoulder_yaw_joint": "R_sh_y",
    "right_elbow_joint": "R_elbow", "right_wrist_roll_joint": "R_wr_r", "right_wrist_pitch_joint": "R_wr_p", "right_wrist_yaw_joint": "R_wr_y",
}


def get_link_positions(env):
    """Get XYZ positions for key links (end effectors and key body parts)."""
    link_positions = {}
    key_links = [
        "left_ankle_roll_link", "right_ankle_roll_link",  # Feet
        "left_wrist_yaw_link", "right_wrist_yaw_link",    # Hands
        "head_link", "pelvis",                             # Head & pelvis
    ]
    for link_name in key_links:
        try:
            link = env.robot.get_link(link_name)
            pos = link.get_pos()[0].cpu().numpy()
            link_positions[link_name] = pos
        except Exception:
            pass
    return link_positions


def print_debug_state(env, actions, step):
    """Print detailed robot state with joint XYZ positions, names, and velocities."""
    pos = env.base_pos[0].cpu().numpy()
    lin_vel = env.base_lin_vel[0].cpu().numpy()
    ang_vel = env.base_ang_vel[0].cpu().numpy()
    dof_pos = env.dof_pos[0].cpu().numpy()
    dof_vel = env.dof_vel[0].cpu().numpy()
    proj_grav = env.projected_gravity[0].cpu().numpy()

    actions_np = actions[0].cpu().numpy() if isinstance(actions, torch.Tensor) else actions[0]

    print(f"\n{'═' * 90}")
    print(f"  STEP {step:5d}")
    print(f"{'═' * 90}")

    # Base state
    print(f"  BASE: pos=({pos[0]:+6.3f}, {pos[1]:+6.3f}, {pos[2]:6.3f}) m")
    print(f"        lin_vel=({lin_vel[0]:+5.2f}, {lin_vel[1]:+5.2f}, {lin_vel[2]:+5.2f}) m/s")
    print(f"        ang_vel=({ang_vel[0]:+5.2f}, {ang_vel[1]:+5.2f}, {ang_vel[2]:+5.2f}) rad/s")
    print(f"        proj_grav=({proj_grav[0]:+5.2f}, {proj_grav[1]:+5.2f}, {proj_grav[2]:+5.2f})")

    # Link XYZ positions
    link_positions = get_link_positions(env)
    if link_positions:
        print(f"\n  {'─' * 86}")
        print(f"  LINK POSITIONS (XYZ):")
        for link_name, xyz in link_positions.items():
            short = link_name.replace("_link", "").replace("_roll", "").replace("_yaw", "")
            print(f"    {short:20s}: ({xyz[0]:+6.3f}, {xyz[1]:+6.3f}, {xyz[2]:6.3f}) m")

    # Joint states table
    print(f"\n  {'─' * 86}")
    print(f"  JOINT STATES:")
    print(f"  {'Joint':<25} {'Pos (rad)':>12} {'Vel (rad/s)':>12} {'Action':>12}")
    print(f"  {'-'*25} {'-'*12} {'-'*12} {'-'*12}")

    num_joints = min(len(dof_pos), len(ALL_JOINT_NAMES))
    for i in range(num_joints):
        joint_name = ALL_JOINT_NAMES[i]
        short_name = SHORT_NAMES.get(joint_name, joint_name[:20])
        act = actions_np[i] if i < len(actions_np) else 0.0
        vel_marker = "█" if abs(dof_vel[i]) > 0.5 else "▓" if abs(dof_vel[i]) > 0.1 else " "
        print(f"  {short_name:<25} {dof_pos[i]:+12.4f} {dof_vel[i]:+12.4f} {act:+12.4f} {vel_marker}")

    # Summary stats
    print(f"\n  {'─' * 86}")
    action_mag = np.sqrt(np.sum(actions_np ** 2))
    max_act_idx = np.argmax(np.abs(actions_np))
    max_vel_idx = np.argmax(np.abs(dof_vel))

    max_act_name = SHORT_NAMES.get(ALL_JOINT_NAMES[max_act_idx], f"j{max_act_idx}") if max_act_idx < len(ALL_JOINT_NAMES) else f"j{max_act_idx}"
    max_vel_name = SHORT_NAMES.get(ALL_JOINT_NAMES[max_vel_idx], f"j{max_vel_idx}") if max_vel_idx < len(ALL_JOINT_NAMES) else f"j{max_vel_idx}"

    print(f"  SUMMARY:")
    print(f"    Action magnitude: {action_mag:.4f}")
    print(f"    Max action: {max_act_name} = {actions_np[max_act_idx]:+.4f}")
    print(f"    Max velocity: {max_vel_name} = {dof_vel[max_vel_idx]:+.4f} rad/s")

    # Leg comparison
    left_leg_vel = np.mean(np.abs(dof_vel[:6]))
    right_leg_vel = np.mean(np.abs(dof_vel[6:12]))
    print(f"    Leg activity: L={left_leg_vel:.3f} R={right_leg_vel:.3f} rad/s")

    # Moving joints
    moving = np.where(np.abs(dof_vel) > 0.1)[0]
    if len(moving) > 0:
        moving_names = [SHORT_NAMES.get(ALL_JOINT_NAMES[i], f"j{i}") for i in moving if i < len(ALL_JOINT_NAMES)]
        print(f"    Active joints ({len(moving)}): {', '.join(moving_names[:8])}" + ("..." if len(moving) > 8 else ""))
    else:
        print(f"    Active joints: NONE (robot static!)")

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

    log_dir = os.path.join(os.path.dirname(__file__), "checkpoints")

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

                if step_count % 10 == 0:
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

            if step_count % 10 == 0:
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
