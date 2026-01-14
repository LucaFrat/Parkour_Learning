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



def penetration_penalty(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    obstacle_cfg: SceneEntityCfg = SceneEntityCfg("obstacle"),
    ):

    obstacle = env.scene[obstacle_cfg.name]

    obstacle_pos = obstacle.data.root_pos_w[:, :]
    print(obstacle_pos.shape)
    obstacle_size = obstacle.spawn.size
    print(obstacle_size.shape)

    # Shape: (Num_Envs, Total_Points, 3)
    points_w = env.penetration_manager.compute_world_points(torch.arange(0, env.scene.num_envs-1, device=env.device, dtype=int))


    # 3. Transform Points to Obstacle Frame
    # Since obstacle is at (0,0,0) relative to Env Origin,
    # and env.scene.env_origins contains the world position of that origin:

    # Obstacle World Pos = Env Origin + (0, 0, H/2) -> Center of box
    # We want points relative to the box CENTER or CORNER?
    # The paper usually calculates relative to the box geometric center.

    # Expand env origins to match points: (Num_Envs, 1, 3)
    env_origins = env.scene.env_origins[env.env_ids].unsqueeze(1)

    # Box Center in World Frame
    box_center_z = box_size[2] / 2.0
    box_pos_w = env_origins.clone()
    box_pos_w[..., 2] += box_center_z

    # Points in Box Frame
    # Shape: (Num_Envs, Total_Points, 3)
    points_b = points_w.view(len(env.env_ids), -1, 3) - box_pos_w

    # 4. Check Intersection (AABB)
    # Box frame is aligned with world frame (assuming no rotation)
    # Condition: |x| < L/2  AND  |y| < W/2  AND  |z| < H/2

    half_l = box_size[0] / 2.0
    half_w = box_size[1] / 2.0
    half_h = box_size[2] / 2.0

    in_x = torch.abs(points_b[..., 0]) < half_l
    in_y = torch.abs(points_b[..., 1]) < half_w
    in_z = torch.abs(points_b[..., 2]) < half_h

    is_inside = in_x & in_y & in_z

    # 5. Calculate Penalty
    # Count how many points are inside
    num_inside = torch.sum(is_inside, dim=1).float()

    return num_inside # The function wrapper applies the negative weight