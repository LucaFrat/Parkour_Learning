from __future__ import annotations

import torch
from typing import TYPE_CHECKING
import numpy as np

from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import euler_xyz_from_quat, quat_apply_inverse

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv



def roll_pitch(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    ) -> torch.Tensor:

    robot = env.scene[asset_cfg.name]
    roll, pitch, _ = euler_xyz_from_quat(robot.data.root_quat_w[:])
    roll[roll > np.pi] -= np.pi * 2 # to range (-pi, pi)
    pitch[pitch > np.pi] -= np.pi * 2 # to range (-pi, pi)

    return torch.stack((roll, pitch), dim=-1)


def center_of_mass(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
    ) -> torch.Tensor:

    robot = env.scene[asset_cfg.name]
    base_pos_w = robot.data.root_pos_w
    base_quat_w = robot.data.root_quat_w

    com_pos_w = robot.data.root_com_pos_w
    com_offset_w = com_pos_w - base_pos_w
    com_offset_b = quat_apply_inverse(base_quat_w, com_offset_w)

    return com_offset_b


def motor_strength(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
    ) -> torch.Tensor:

    robot = env.scene[asset_cfg.name]
    return robot.actuators["base_legs"].effort_limit