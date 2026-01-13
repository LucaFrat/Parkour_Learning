# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import wrap_to_pi

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def joint_pos_target_l2(env: ManagerBasedRLEnv, target: float, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Penalize joint position deviation from a target value."""
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    # wrap the joint positions to (-pi, pi)
    joint_pos = wrap_to_pi(asset.data.joint_pos[:, asset_cfg.joint_ids])
    # compute the reward
    return torch.sum(torch.square(joint_pos - target), dim=1)


def forward_velocity(
    env: ManagerBasedRLEnv,
    command_name: str = "forward_velocity",
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
    ) -> torch.Tensor:

    robot = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)

    command_x = command[:, 0]
    robot_vel_x = robot.data.root_lin_vel_w[:, 0]

    return torch.norm(command_x - robot_vel_x)


def lateral_velocity(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
    ) -> torch.Tensor:

    robot = env.scene[asset_cfg.name]
    robot_vel_y = robot.data.root_lin_vel_w[:, 1]

    return torch.norm(robot_vel_y) ** 2


def yaw_rate(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
    ) -> torch.Tensor:

    robot = env.scene[asset_cfg.name]
    robot_yaw_rate = robot.data.root_ang_vel_w[:, 2]
    robot_yaw_rate_norm = torch.norm(robot_yaw_rate)

    return torch.exp(-robot_yaw_rate_norm)


def energy_usage(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    ) -> torch.Tensor:

    robot = env.scene[asset_cfg.name]

    torques = robot.data.applied_torque[:, :]
    velocities = robot.data.joint_vel[:, :]

    value_per_joing = torch.abs(torques*velocities) **2
    return torch.sum(value_per_joing, dim=1)