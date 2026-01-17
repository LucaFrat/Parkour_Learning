# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
from typing import TYPE_CHECKING
import numpy as np

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import wrap_to_pi

from Robot_Parkour.tasks.manager_based.robot_parkour.config.go2.config import SimConfig
from Robot_Parkour.tasks.manager_based.robot_parkour.utils.penetration_points import PenetrationManager

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

PENETRATION_MANAGER = None


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


def obstacle_penetration(
    env: ManagerBasedRLEnv,
    weight_violation: float,
    weight_depth: float,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    obstacle_cfg: SceneEntityCfg = SceneEntityCfg("obstacle"),
    debug_vis: bool = False
    ) -> torch.Tensor:

    robot = env.scene[robot_cfg.name]
    obstacle = env.scene[obstacle_cfg.name]

    robot_vel_x = robot.data.root_vel_w[:, 0].squeeze().clone()
    obstacle_center = obstacle.data.root_pos_w.clone() # (num_envs, 3)
    obstacle_size = obstacle.cfg.spawn.size # (3)

    # fix dimensions
    # (num_envs, 3) -> (num_envs, 1, 3)
    obstacle_center = obstacle_center.unsqueeze(1)
    obstacle_size = torch.tensor(obstacle_size, device=env.device).view(1, 1, 3)

    half_size = obstacle_size / 2

    global PENETRATION_MANAGER

    if PENETRATION_MANAGER is None:
        points_cfg = SimConfig.body_measure_points
        PENETRATION_MANAGER = PenetrationManager(env, points_cfg)
        PENETRATION_MANAGER.visualizer.set_visibility(True)

    # (num_envs, num_points, 3)
    points = PENETRATION_MANAGER.compute_world_points().view(env.scene.num_envs, -1, 3)

    # DEPTH
    dist_from_center = torch.abs(points - obstacle_center)
    # Positive = Inside, Negative = Outside
    dist_to_face = half_size - dist_from_center
    min_dist_to_face = torch.min(dist_to_face, dim=-1).values
    depths = torch.clamp(min_dist_to_face, min=0.0)
    total_depth = torch.sum(depths, dim=-1)

    # VIOLATIONS
    is_point_inside = depths > 0.0
    num_penetrating_points = torch.sum(is_point_inside, dim=-1).float()

    # TOTAL
    penalty = (weight_violation * num_penetrating_points + weight_depth * total_depth) * torch.abs(robot_vel_x)

    if debug_vis:
        PENETRATION_MANAGER.visualize(is_inside_list=is_point_inside, env_ids=[0])

    return penalty