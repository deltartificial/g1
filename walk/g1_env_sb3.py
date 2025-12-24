"""
Stable-Baselines3 Wrapper for G1Env

Converts the Genesis-style G1Env to SB3 VecEnv interface.
"""

import numpy as np
import torch
from stable_baselines3.common.vec_env.base_vec_env import VecEnv
from gymnasium import spaces


class G1EnvSB3Wrapper(VecEnv):
    """
    Wraps G1Env (Genesis style) to be compatible with stable-baselines3 VecEnv.
    """

    def __init__(self, genesis_env):
        self.env = genesis_env
        self.num_envs = genesis_env.num_envs
        self.device = genesis_env.device

        # Define spaces
        observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(genesis_env.num_obs,),
            dtype=np.float32
        )
        action_space = spaces.Box(
            low=-genesis_env.env_cfg["clip_actions"],
            high=genesis_env.env_cfg["clip_actions"],
            shape=(genesis_env.num_actions,),
            dtype=np.float32
        )

        super().__init__(self.num_envs, observation_space, action_space)

        # Buffers for async step
        self._actions = None

        # Episode tracking for SB3
        self._episode_rewards = np.zeros(self.num_envs, dtype=np.float32)
        self._episode_lengths = np.zeros(self.num_envs, dtype=np.int32)

    def reset(self):
        obs, _ = self.env.reset()
        self._episode_rewards[:] = 0
        self._episode_lengths[:] = 0
        return self._to_numpy(obs)

    def step_async(self, actions):
        self._actions = self._to_tensor(actions)

    def step_wait(self):
        obs, _, rews, dones, extras = self.env.step(self._actions)

        obs_np = self._to_numpy(obs)
        rews_np = self._to_numpy(rews)
        dones_np = self._to_numpy(dones).astype(bool)

        # Track episode stats
        self._episode_rewards += rews_np
        self._episode_lengths += 1

        # Convert infos to SB3 format
        infos = []
        for i in range(self.num_envs):
            info = {}
            if dones_np[i]:
                # SB3 expects 'episode' dict with 'r' (reward) and 'l' (length)
                info["episode"] = {
                    "r": float(self._episode_rewards[i]),
                    "l": int(self._episode_lengths[i]),
                }
                # Reset tracking for this env
                self._episode_rewards[i] = 0
                self._episode_lengths[i] = 0

            if "time_outs" in extras:
                timeout_val = extras["time_outs"]
                if isinstance(timeout_val, torch.Tensor):
                    timeout_val = timeout_val.cpu().numpy()
                if timeout_val[i]:
                    info["TimeLimit.truncated"] = True
            infos.append(info)

        return obs_np, rews_np, dones_np, infos

    def close(self):
        pass

    def env_is_wrapped(self, wrapper_class, indices=None):
        return [False] * self.num_envs

    def env_method(self, method_name, *method_args, indices=None, **method_kwargs):
        pass

    def get_attr(self, attr_name, indices=None):
        return [getattr(self.env, attr_name)] * self.num_envs

    def set_attr(self, attr_name, value, indices=None):
        setattr(self.env, attr_name, value)

    def seed(self, seed=None):
        pass

    def render(self, mode="human"):
        pass

    def _to_numpy(self, tensor):
        """Convert torch tensor to numpy array."""
        if isinstance(tensor, torch.Tensor):
            return tensor.cpu().numpy().astype(np.float32)
        return np.asarray(tensor, dtype=np.float32)

    def _to_tensor(self, array):
        """Convert numpy array to torch tensor."""
        if isinstance(array, np.ndarray):
            return torch.from_numpy(array).to(self.device).float()
        return array
