# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import gymnasium as gym

from . import agents





# Specialized policy on SOFT dynamics
gym.register(
    id="Isaac-Go2-Climb-Soft-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.env_cfg:Go2ClimbSoftEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:PPORunnerFieldSoftCfg",
        "play_env_cfg_entry_point": f"{__name__}.rsl_rl_ppo_cfg:PPORunnerFieldSoftCfg",

    },
)
gym.register(
    id="Isaac-Go2-Tilt-Soft-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.env_cfg:Go2TiltSoftEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:PPORunnerFieldSoftCfg",
        "play_env_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:PPORunnerFieldSoftCfg",
    },
)

# Specialized policy on HARD dynamics
gym.register(
    id="Isaac-Go2-Climb-Hard-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.env_cfg:Go2ClimbHardEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:PPORunnerFieldHardCfg",
        "play_env_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:PPORunnerFieldHardCfg",
    }
)
gym.register(
    id="Isaac-Go2-Tilt-Hard-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.env_cfg:Go2TiltHardEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:PPORunnerFieldHardCfg",
        "play_env_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:PPORunnerFieldHardCfg",
        "rsl_rl_distillation_cfg_entry_point": (
            f"{agents.__name__}.rsl_rl_distill_cfg:DistillationRunnerCfg"
        ),
    }
)



# =============================== PLAY ====================================

gym.register(
    id="Isaac-Go2-Climb-Soft-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.env_cfg:Go2ClimbSoftEnvCfg_Play",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:PPORunnerFieldSoftCfg",
    }
)

gym.register(
    id="Isaac-Go2-Climb-Hard-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.env_cfg:Go2ClimbHardEnvCfg_Play",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:PPORunnerFieldHardCfg",
    }
)

gym.register(
    id="Isaac-Go2-Tilt-Hard-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.env_cfg:Go2TiltHardEnvCfg_Play",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:PPORunnerFieldHardCfg",
    }
)