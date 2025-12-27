"""
G1 Bipedal Walking Environment

Adapted from Genesis Go2 locomotion example for Unitree G1 humanoid robot.
Key differences from quadruped:
- 29 DOFs (12 leg + 3 waist + 14 arm) vs 12 DOFs
- Higher CoM requiring more precise balance control
- Bipedal gait with alternating stance/swing phases
- Critical need for projected gravity observation

Reference: Genesis examples/locomotion/go2_env.py
"""

import torch
import math
import genesis as gs
from genesis.utils.geom import quat_to_xyz, transform_by_quat, inv_quat, transform_quat_by_quat


def gs_rand_float(lower, upper, shape, device):
    return (upper - lower) * torch.rand(size=shape, device=device) + lower


class ObsDict(dict):
    """Dict wrapper that supports .to(device) for rsl_rl v3.x compatibility."""
    def to(self, device):
        return ObsDict({k: v.to(device) if hasattr(v, 'to') else v for k, v in self.items()})


class G1Env:
    """
    Genesis-style environment for G1 bipedal locomotion.

    Observation space (48 dim):
        - base_ang_vel (3): Angular velocity in body frame
        - projected_gravity (3): Gravity vector in body frame (critical for balance)
        - commands (3): [vx, vy, vyaw] velocity commands
        - dof_pos (12): Leg joint positions relative to default
        - dof_vel (12): Leg joint velocities
        - actions (12): Previous actions (for smoothness)
        - phase (2): Gait phase [sin, cos] for periodic walking

    Action space (12 dim):
        - Target positions for leg joints (scaled delta from default pose)
    """

    # Joint organization for G1
    LEG_DOF_NAMES = [
        # Left leg (6 joints)
        "left_hip_pitch_joint",
        "left_hip_roll_joint",
        "left_hip_yaw_joint",
        "left_knee_joint",
        "left_ankle_pitch_joint",
        "left_ankle_roll_joint",
        # Right leg (6 joints)
        "right_hip_pitch_joint",
        "right_hip_roll_joint",
        "right_hip_yaw_joint",
        "right_knee_joint",
        "right_ankle_pitch_joint",
        "right_ankle_roll_joint",
    ]

    # Upper body joints - MUST be actively controlled for balance
    UPPER_BODY_DOF_NAMES = [
        # Waist (3 joints) - CRITICAL for balance
        "waist_yaw_joint",
        "waist_roll_joint",
        "waist_pitch_joint",
        # Left arm (7 joints)
        "left_shoulder_pitch_joint",
        "left_shoulder_roll_joint",
        "left_shoulder_yaw_joint",
        "left_elbow_joint",
        "left_wrist_roll_joint",
        "left_wrist_pitch_joint",
        "left_wrist_yaw_joint",
        # Right arm (7 joints)
        "right_shoulder_pitch_joint",
        "right_shoulder_roll_joint",
        "right_shoulder_yaw_joint",
        "right_elbow_joint",
        "right_wrist_roll_joint",
        "right_wrist_pitch_joint",
        "right_wrist_yaw_joint",
    ]

    # Default standing pose - UPRIGHT VERSION (jambes quasi-droites, plus stable)
    DEFAULT_LEG_ANGLES = {
        "left_hip_pitch_joint": -0.1,     # Était -0.312
        "left_hip_roll_joint": 0.0,
        "left_hip_yaw_joint": 0.0,
        "left_knee_joint": 0.3,           # Était 0.669 (squat) -> micro-flexion
        "left_ankle_pitch_joint": -0.2,   # Était -0.363
        "left_ankle_roll_joint": 0.0,
        "right_hip_pitch_joint": -0.1,
        "right_hip_roll_joint": 0.0,
        "right_hip_yaw_joint": 0.0,
        "right_knee_joint": 0.3,
        "right_ankle_pitch_joint": -0.2,
        "right_ankle_roll_joint": 0.0,
    }

    # Upper body default pose - arms tucked, waist neutral
    DEFAULT_UPPER_BODY_ANGLES = {
        "waist_yaw_joint": 0.0,
        "waist_roll_joint": 0.0,
        "waist_pitch_joint": 0.0,
        "left_shoulder_pitch_joint": 0.2,
        "left_shoulder_roll_joint": 0.2,
        "left_shoulder_yaw_joint": 0.0,
        "left_elbow_joint": 0.6,
        "left_wrist_roll_joint": 0.0,
        "left_wrist_pitch_joint": 0.0,
        "left_wrist_yaw_joint": 0.0,
        "right_shoulder_pitch_joint": 0.2,
        "right_shoulder_roll_joint": -0.2,
        "right_shoulder_yaw_joint": 0.0,
        "right_elbow_joint": 0.6,
        "right_wrist_roll_joint": 0.0,
        "right_wrist_pitch_joint": 0.0,
        "right_wrist_yaw_joint": 0.0,
    }

    def __init__(self, num_envs, env_cfg, obs_cfg, reward_cfg, command_cfg, show_viewer=False, device="mps"):
        self.device = torch.device(device)

        self.num_envs = num_envs
        self.num_obs = obs_cfg["num_obs"]
        self.num_privileged_obs = None
        self.num_actions = env_cfg["num_actions"]  # 12 for legs only
        self.num_commands = command_cfg["num_commands"]

        self.simulate_action_latency = env_cfg.get("simulate_action_latency", True)
        self.dt = env_cfg.get("dt", 0.02)  # 50Hz control
        self.max_episode_length = math.ceil(env_cfg["episode_length_s"] / self.dt)

        self.env_cfg = env_cfg
        self.obs_cfg = obs_cfg
        self.reward_cfg = reward_cfg
        self.command_cfg = command_cfg

        # rsl_rl v3.x expects env.cfg attribute
        self.cfg = env_cfg

        self.obs_scales = obs_cfg["obs_scales"]
        self.reward_scales = reward_cfg["reward_scales"]

        # Build Genesis scene
        self.scene = gs.Scene(
            sim_options=gs.options.SimOptions(
                dt=self.dt,
                substeps=env_cfg.get("substeps", 4),
                gravity=(0, 0, -9.81),
            ),
            viewer_options=gs.options.ViewerOptions(
                max_FPS=int(0.5 / self.dt),
                camera_pos=(3.0, 0.0, 1.5),
                camera_lookat=(0.0, 0.0, 0.8),
                camera_fov=40,
            ),
            vis_options=gs.options.VisOptions(n_rendered_envs=1),
            rigid_options=gs.options.RigidOptions(
                dt=self.dt,
                constraint_solver=gs.constraint_solver.Newton,
                enable_collision=True,
                enable_joint_limit=True,
            ),
            show_viewer=show_viewer,
        )

        # Ground plane
        self.scene.add_entity(gs.morphs.Plane())

        # G1 Robot
        self.base_init_pos = torch.tensor(env_cfg["base_init_pos"], device=self.device)
        self.base_init_quat = torch.tensor(env_cfg["base_init_quat"], device=self.device)
        self.inv_base_init_quat = inv_quat(self.base_init_quat)

        self.robot = self.scene.add_entity(
            gs.morphs.MJCF(
                file=env_cfg["robot_path"],
                pos=self.base_init_pos.cpu().numpy(),
            ),
        )

        # Build scene with parallel envs
        self.scene.build(n_envs=num_envs)

        # Full body control: merge all DOFs
        self.ALL_DOF_NAMES = self.LEG_DOF_NAMES + self.UPPER_BODY_DOF_NAMES

        self.motor_dofs = []
        for name in self.ALL_DOF_NAMES:
            idx = self.robot.get_joint(name).dofs_idx_local
            if isinstance(idx, (list, tuple)):
                self.motor_dofs.extend(idx)
            else:
                self.motor_dofs.append(idx)

        # No separate upper body control - everything is RL-controlled now
        self.upper_body_dofs = []

        # Differentiated PD gains: legs rigid, waist rigid, arms soft
        kp_list = []
        kd_list = []
        for name in self.ALL_DOF_NAMES:
            if name in self.LEG_DOF_NAMES:
                kp_list.append(env_cfg.get("leg_kp", 150.0))
                kd_list.append(env_cfg.get("leg_kd", 15.0))
            elif "waist" in name:
                kp_list.append(env_cfg.get("waist_kp", 200.0))
                kd_list.append(env_cfg.get("waist_kd", 20.0))
            else:  # Arms
                kp_list.append(env_cfg.get("arm_kp", 50.0))
                kd_list.append(env_cfg.get("arm_kd", 5.0))

        print(f"[INFO] Full Body Control: {len(self.motor_dofs)} DOFs")
        self.robot.set_dofs_kp(kp_list, self.motor_dofs)
        self.robot.set_dofs_kv(kd_list, self.motor_dofs)

        # Reward functions
        self.reward_functions, self.episode_sums = dict(), dict()
        for name in self.reward_scales.keys():
            self.reward_scales[name] *= self.dt
            self.reward_functions[name] = getattr(self, "_reward_" + name)
            self.episode_sums[name] = torch.zeros((self.num_envs,), device=self.device, dtype=gs.tc_float)

        # Initialize buffers
        self._init_buffers()

    def _init_buffers(self):
        """Initialize all state buffers."""
        # Base state
        self.base_lin_vel = torch.zeros((self.num_envs, 3), device=self.device, dtype=gs.tc_float)
        self.base_ang_vel = torch.zeros((self.num_envs, 3), device=self.device, dtype=gs.tc_float)
        self.projected_gravity = torch.zeros((self.num_envs, 3), device=self.device, dtype=gs.tc_float)
        self.global_gravity = torch.tensor([0.0, 0.0, -1.0], device=self.device, dtype=gs.tc_float).repeat(
            self.num_envs, 1
        )

        # Observation and reward buffers
        self.obs_buf = torch.zeros((self.num_envs, self.num_obs), device=self.device, dtype=gs.tc_float)
        self.rew_buf = torch.zeros((self.num_envs,), device=self.device, dtype=gs.tc_float)
        self.reset_buf = torch.ones((self.num_envs,), device=self.device, dtype=gs.tc_int)
        self.episode_length_buf = torch.zeros((self.num_envs,), device=self.device, dtype=gs.tc_int)

        # Commands
        self.commands = torch.zeros((self.num_envs, self.num_commands), device=self.device, dtype=gs.tc_float)
        self.commands_scale = torch.tensor(
            [self.obs_scales["lin_vel"], self.obs_scales["lin_vel"], self.obs_scales["ang_vel"]],
            device=self.device,
            dtype=gs.tc_float,
        )

        # Actions and DOF state
        self.actions = torch.zeros((self.num_envs, self.num_actions), device=self.device, dtype=gs.tc_float)
        self.last_actions = torch.zeros_like(self.actions)
        self.dof_pos = torch.zeros_like(self.actions)
        self.dof_vel = torch.zeros_like(self.actions)
        self.last_dof_vel = torch.zeros_like(self.actions)

        # Base pose
        self.base_pos = torch.zeros((self.num_envs, 3), device=self.device, dtype=gs.tc_float)
        self.base_quat = torch.zeros((self.num_envs, 4), device=self.device, dtype=gs.tc_float)

        # Default joint positions for full body (29 DOFs)
        default_pos_list = []
        for name in self.ALL_DOF_NAMES:
            if name in self.DEFAULT_LEG_ANGLES:
                default_pos_list.append(self.DEFAULT_LEG_ANGLES[name])
            elif name in self.DEFAULT_UPPER_BODY_ANGLES:
                default_pos_list.append(self.DEFAULT_UPPER_BODY_ANGLES[name])
            else:
                default_pos_list.append(0.0)
        self.default_dof_pos = torch.tensor(default_pos_list, device=self.device, dtype=gs.tc_float)

        # Gait phase for periodic walking pattern
        self.phase = torch.zeros((self.num_envs, 1), device=self.device, dtype=gs.tc_float)
        self.phase_freq = 1.5  # Hz - walking frequency

        self.extras = dict()

    def _resample_commands(self, envs_idx):
        """Sample new velocity commands for specified environments."""
        self.commands[envs_idx, 0] = gs_rand_float(*self.command_cfg["lin_vel_x_range"], (len(envs_idx),), self.device)
        self.commands[envs_idx, 1] = gs_rand_float(*self.command_cfg["lin_vel_y_range"], (len(envs_idx),), self.device)
        self.commands[envs_idx, 2] = gs_rand_float(*self.command_cfg["ang_vel_range"], (len(envs_idx),), self.device)

    def step(self, actions):
        """Execute one environment step."""
        self.actions = torch.clip(actions, -self.env_cfg["clip_actions"], self.env_cfg["clip_actions"])
        exec_actions = self.last_actions if self.simulate_action_latency else self.actions
        target_dof_pos = exec_actions * self.env_cfg["action_scale"] + self.default_dof_pos
        self.robot.control_dofs_position(target_dof_pos, self.motor_dofs)

        self.scene.step()

        # Update episode counter and phase
        self.episode_length_buf += 1
        self.phase += self.dt * self.phase_freq * 2 * math.pi
        self.phase = torch.fmod(self.phase, 2 * math.pi)

        # Update state buffers
        self.base_pos[:] = self.robot.get_pos()
        self.base_quat[:] = self.robot.get_quat()
        self.base_euler = quat_to_xyz(
            transform_quat_by_quat(torch.ones_like(self.base_quat) * self.inv_base_init_quat, self.base_quat)
        )
        inv_base_quat = inv_quat(self.base_quat)
        self.base_lin_vel[:] = transform_by_quat(self.robot.get_vel(), inv_base_quat)
        self.base_ang_vel[:] = transform_by_quat(self.robot.get_ang(), inv_base_quat)
        self.projected_gravity = transform_by_quat(self.global_gravity, inv_base_quat)
        self.dof_pos[:] = self.robot.get_dofs_position(self.motor_dofs)
        self.dof_vel[:] = self.robot.get_dofs_velocity(self.motor_dofs)

        # Resample commands periodically
        envs_idx = (
            (self.episode_length_buf % int(self.env_cfg["resampling_time_s"] / self.dt) == 0)
            .nonzero(as_tuple=False)
            .flatten()
        )
        self._resample_commands(envs_idx)

        # Check termination
        self.reset_buf = self.episode_length_buf > self.max_episode_length
        self.reset_buf |= torch.abs(self.base_euler[:, 1]) > self.env_cfg["termination_if_pitch_greater_than"]
        self.reset_buf |= torch.abs(self.base_euler[:, 0]) > self.env_cfg["termination_if_roll_greater_than"]
        self.reset_buf |= self.base_pos[:, 2] < self.env_cfg["termination_if_height_lower_than"]

        time_out_idx = (self.episode_length_buf > self.max_episode_length).nonzero(as_tuple=False).flatten()
        self.extras["time_outs"] = torch.zeros_like(self.reset_buf, device=self.device, dtype=gs.tc_float)
        self.extras["time_outs"][time_out_idx] = 1.0

        self.reset_idx(self.reset_buf.nonzero(as_tuple=False).flatten())

        # Compute reward
        self.rew_buf[:] = 0.0
        for name, reward_func in self.reward_functions.items():
            rew = reward_func() * self.reward_scales[name]
            self.rew_buf += rew
            self.episode_sums[name] += rew

        # Build observation
        self.obs_buf = torch.cat(
            [
                self.base_ang_vel * self.obs_scales["ang_vel"],           # 3
                self.projected_gravity,                                     # 3
                self.commands * self.commands_scale,                        # 3
                (self.dof_pos - self.default_dof_pos) * self.obs_scales["dof_pos"],  # 12
                self.dof_vel * self.obs_scales["dof_vel"],                 # 12
                self.actions,                                               # 12
                torch.sin(self.phase),                                      # 1
                torch.cos(self.phase),                                      # 1
            ],
            axis=-1,
        )

        self.last_actions[:] = self.actions[:]
        self.last_dof_vel[:] = self.dof_vel[:]

        # rsl_rl v3.x expects: obs, rewards, dones, extras (4 values)
        return ObsDict({"policy": self.obs_buf}), self.rew_buf, self.reset_buf, self.extras

    def get_observations(self):
        return ObsDict({"policy": self.obs_buf})

    def get_privileged_observations(self):
        return None

    def reset_idx(self, envs_idx):
        """Reset specific environments."""
        if len(envs_idx) == 0:
            return

        # Reset leg DOFs to default
        self.dof_pos[envs_idx] = self.default_dof_pos
        self.dof_vel[envs_idx] = 0.0
        self.robot.set_dofs_position(
            position=self.dof_pos[envs_idx],
            dofs_idx_local=self.motor_dofs,
            zero_velocity=True,
            envs_idx=envs_idx,
        )

        # Reset upper body DOFs to default
        if self.upper_body_dofs:
            self.robot.set_dofs_position(
                position=self.default_upper_body_pos.unsqueeze(0).expand(len(envs_idx), -1),
                dofs_idx_local=self.upper_body_dofs,
                zero_velocity=True,
                envs_idx=envs_idx,
            )

        # Reset base pose
        self.base_pos[envs_idx] = self.base_init_pos
        self.base_quat[envs_idx] = self.base_init_quat.reshape(1, -1)
        self.robot.set_pos(self.base_pos[envs_idx], zero_velocity=False, envs_idx=envs_idx)
        self.robot.set_quat(self.base_quat[envs_idx], zero_velocity=False, envs_idx=envs_idx)
        self.base_lin_vel[envs_idx] = 0
        self.base_ang_vel[envs_idx] = 0
        self.robot.zero_all_dofs_velocity(envs_idx)

        # Reset buffers
        self.last_actions[envs_idx] = 0.0
        self.last_dof_vel[envs_idx] = 0.0
        self.episode_length_buf[envs_idx] = 0
        self.reset_buf[envs_idx] = True
        self.phase[envs_idx] = 0.0

        # Log episode stats
        self.extras["episode"] = {}
        for key in self.episode_sums.keys():
            self.extras["episode"]["rew_" + key] = (
                torch.mean(self.episode_sums[key][envs_idx]).item() / self.env_cfg["episode_length_s"]
            )
            self.episode_sums[key][envs_idx] = 0.0

        self._resample_commands(envs_idx)

    def reset(self):
        """Reset all environments."""
        self.reset_buf[:] = True
        self.reset_idx(torch.arange(self.num_envs, device=self.device))
        return ObsDict({"policy": self.obs_buf}), None

    # ===================== REWARD FUNCTIONS =====================
    # Modular reward design following Genesis/rsl_rl conventions

    def _reward_tracking_lin_vel(self):
        """Track commanded linear velocity (xy plane)."""
        lin_vel_error = torch.sum(torch.square(self.commands[:, :2] - self.base_lin_vel[:, :2]), dim=1)
        return torch.exp(-lin_vel_error / self.reward_cfg["tracking_sigma"])

    def _reward_tracking_ang_vel(self):
        """Track commanded angular velocity (yaw)."""
        ang_vel_error = torch.square(self.commands[:, 2] - self.base_ang_vel[:, 2])
        return torch.exp(-ang_vel_error / self.reward_cfg["tracking_sigma"])

    def _reward_lin_vel_z(self):
        """Penalize vertical base velocity (bouncing)."""
        return torch.square(self.base_lin_vel[:, 2])

    def _reward_ang_vel_xy(self):
        """Penalize roll/pitch angular velocities."""
        return torch.sum(torch.square(self.base_ang_vel[:, :2]), dim=1)

    def _reward_orientation(self):
        """Penalize non-upright orientation (critical for bipeds)."""
        return torch.sum(torch.square(self.projected_gravity[:, :2]), dim=1)

    def _reward_base_height(self):
        """Penalize deviation from target height."""
        return torch.square(self.base_pos[:, 2] - self.reward_cfg["base_height_target"])

    def _reward_action_rate(self):
        """Penalize rapid action changes for smoothness."""
        return torch.sum(torch.square(self.last_actions - self.actions), dim=1)

    def _reward_similar_to_default(self):
        """Penalize deviation from default standing pose."""
        return torch.sum(torch.abs(self.dof_pos - self.default_dof_pos), dim=1)

    def _reward_dof_vel(self):
        """Penalize high joint velocities."""
        return torch.sum(torch.square(self.dof_vel), dim=1)

    def _reward_dof_acc(self):
        """Penalize joint accelerations."""
        return torch.sum(torch.square(self.dof_vel - self.last_dof_vel), dim=1)

    def _reward_torques(self):
        """Penalize high torques (approximated by action magnitude)."""
        return torch.sum(torch.square(self.actions), dim=1)

    def _reward_feet_air_time(self):
        """
        Reward alternating feet contact for walking gait.
        Uses phase signal to encourage left/right foot coordination.
        """
        # Left leg joints: 0-5, Right leg joints: 6-11
        left_hip_pitch = self.dof_pos[:, 0]
        right_hip_pitch = self.dof_pos[:, 6]

        # Encourage anti-phase hip motion (walking pattern)
        phase_signal = torch.sin(self.phase.squeeze(-1))
        left_target = phase_signal * 0.3
        right_target = -phase_signal * 0.3

        left_error = torch.square(left_hip_pitch - left_target)
        right_error = torch.square(right_hip_pitch - right_target)

        return torch.exp(-(left_error + right_error) / 0.1)

    def _reward_symmetry(self):
        """Encourage symmetric leg motion."""
        left_dof = self.dof_pos[:, :6]
        right_dof = self.dof_pos[:, 6:]

        # Mirror mapping: same joint on opposite side should have opposite sign for some
        # Hip roll and yaw should be mirrored, others same
        mirror_mask = torch.tensor([1, -1, -1, 1, 1, -1], device=self.device, dtype=gs.tc_float)

        symmetry_error = torch.sum(torch.square(left_dof - right_dof * mirror_mask), dim=1)
        return torch.exp(-symmetry_error / 0.5)

    def _reward_feet_spacing(self):
        """
        Gaussian stance reward: target narrow human-like foot width.
        Penalizes both too wide (cowboy) and too narrow (scissoring).
        """
        left_hip_roll = self.dof_pos[:, 1]
        right_hip_roll = self.dof_pos[:, 7]

        spacing = torch.abs(left_hip_roll - right_hip_roll)
        target = 0.08  # Much narrower for human-like gait (was 0.15)

        # Tight sigma for strict enforcement
        return torch.exp(-torch.square(spacing - target) / 0.01)

    def _reward_quiet_wrists(self):
        """
        Anti-hack: penalize frantic wrist/elbow movement.
        Forces agent to use torso for balance, not hands.
        """
        # Wrist indices in arm chains
        # L_Arm: 15-21 (Wrist ~ 19-21), R_Arm: 22-28 (Wrist ~ 26-28)
        wrist_indices = [19, 20, 21, 26, 27, 28]

        wrist_vel = self.dof_vel[:, wrist_indices]
        return torch.sum(torch.square(wrist_vel), dim=1)

    def _reward_alive(self):
        """Small survival bonus."""
        return torch.ones(self.num_envs, device=self.device, dtype=gs.tc_float)

    def _reward_arm_swing(self):
        """
        Encourage contralateral arm-leg synchronization.
        Left arm swings with right leg, right arm swings with left leg.
        Uses phase signal to guide sinusoidal shoulder motion.

        DOF order: Legs(12) -> Waist(3) -> L_Arm(7) -> R_Arm(7)
        L_Shoulder_Pitch = index 15, R_Shoulder_Pitch = index 22
        """
        l_sh_pitch = self.dof_pos[:, 15]
        r_sh_pitch = self.dof_pos[:, 22]

        # Phase drives leg swing. Arms should be opposite.
        phase_signal = torch.sin(self.phase.squeeze(-1))

        target_l = -0.6 * phase_signal  # Amplitude 0.6 rad
        target_r = 0.6 * phase_signal

        err_l = torch.square(l_sh_pitch - target_l)
        err_r = torch.square(r_sh_pitch - target_r)

        return torch.exp(-(err_l + err_r) / 0.2)

    def _reward_arm_close_to_body(self):
        """
        Penalize T-pose (arms spread out) via Shoulder Roll.
        L_Shoulder_Roll = index 16, R_Shoulder_Roll = index 23
        """
        l_sh_roll = self.dof_pos[:, 16]
        r_sh_roll = self.dof_pos[:, 23]

        # We want them close to 0 (arms along body)
        return torch.exp(-(torch.square(l_sh_roll) + torch.square(r_sh_roll)) / 0.1)

    def _reward_track_pitch(self):
        """
        Force upright torso with ASYMMETRIC penalty.
        Penalize backward lean (pitch < 0) 3x more than forward lean.
        """
        pitch = self.base_euler[:, 1]
        # Asymmetric: backward lean (negative pitch) is punished 3x harder
        penalty = torch.where(pitch < 0, 3.0 * torch.square(pitch), torch.square(pitch))
        return torch.exp(-penalty / 0.05)

    def _reward_waist_pitch(self):
        """
        Penalize waist_pitch joint deviation from 0.
        This is the REAL torso lean - waist_p at index 14.
        Asymmetric: backward lean (negative) punished 3x harder.
        """
        waist_p = self.dof_pos[:, 14]  # waist_pitch joint
        # Asymmetric: negative = backward lean = 3x penalty
        penalty = torch.where(waist_p < 0, 3.0 * torch.square(waist_p), torch.square(waist_p))
        return torch.exp(-penalty / 0.02)  # Tight sigma

    def _reward_arm_actions(self):
        """
        Penalize extreme actions on arm joints.
        Prevents the robot from using arms as counterweights.
        Arm indices: 15-28 (L_arm: 15-21, R_arm: 22-28)
        """
        arm_actions = self.actions[:, 15:29]  # All arm joints
        return torch.sum(torch.square(arm_actions), dim=1)
