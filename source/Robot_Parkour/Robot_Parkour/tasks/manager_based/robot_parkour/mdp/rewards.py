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
from isaaclab.utils.math import wrap_to_pi, quat_apply_inverse, yaw_quat
from isaaclab.sensors import ContactSensor

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

def track_lin_vel_xy_exp(
    env: ManagerBasedRLEnv,
    std: float,
    command_name: str,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
    ) -> torch.Tensor:

    robot = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)[:, :2]
    robot_vel = quat_apply_inverse(robot.data.root_quat_w, robot.data.root_lin_vel_w[:, :3])
    lin_vel_error = torch.sum(torch.square(command - robot_vel[:, :2]), dim=-1)
    return torch.exp(-lin_vel_error / std**2)

def track_ang_vel_z_world_exp(
    env, command_name: str, std: float, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Reward tracking of angular velocity commands (yaw) in world frame using exponential kernel."""
    # extract the used quantities (to enable type-hinting)
    asset = env.scene[asset_cfg.name]
    ang_vel_error = torch.square(env.command_manager.get_command(command_name)[:, 2] - asset.data.root_ang_vel_w[:, 2])
    return torch.exp(-ang_vel_error / std**2)


def feet_air_time(
    env: ManagerBasedRLEnv, command_name: str, sensor_cfg: SceneEntityCfg, threshold: float
) -> torch.Tensor:
    """Reward long steps taken by the feet using L2-kernel.

    This function rewards the agent for taking steps that are longer than a threshold. This helps ensure
    that the robot lifts its feet off the ground and takes steps. The reward is computed as the sum of
    the time for which the feet are in the air.

    If the commands are small (i.e. the agent is not supposed to take a step), then the reward is zero.
    """
    # extract the used quantities (to enable type-hinting)
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    # compute the reward
    first_contact = contact_sensor.compute_first_contact(env.step_dt)[:, sensor_cfg.body_ids]
    last_air_time = contact_sensor.data.last_air_time[:, sensor_cfg.body_ids]
    reward = torch.sum((last_air_time - threshold) * first_contact, dim=1)
    # no reward for zero command
    reward *= torch.norm(env.command_manager.get_command(command_name)[:, :2], dim=1) > 0.1
    return reward


def forward_velocity(
    env: ManagerBasedRLEnv,
    command_name: str = "forward_velocity",
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
    ) -> torch.Tensor:

    robot = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)

    command_vel_x = command[:, 0]
    robot_vel_b = quat_apply_inverse(robot.data.root_quat_w, robot.data.root_lin_vel_w[:, :3])

    error = torch.square(command_vel_x - robot_vel_b[:, 0])
    return torch.exp(-error / 0.25**2)


def lateral_velocity(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
    ) -> torch.Tensor:

    robot = env.scene[asset_cfg.name]
    robot_vel = quat_apply_inverse(robot.data.root_quat_w, robot.data.root_lin_vel_w[:, :3])

    return torch.square(robot_vel[:, 1])


def yaw_rate(
    env: ManagerBasedRLEnv,
    command_name: str = "forward_velocity",
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
    ) -> torch.Tensor:

    robot = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)
    robot_ang_vel = quat_apply_inverse(robot.data.root_quat_w, robot.data.root_ang_vel_w[:, :3])

    robot_yaw_rate = wrap_to_pi(robot_ang_vel[:, 2])

    return torch.exp(-torch.square(robot_yaw_rate - command[:, 2]) / 0.25**2)


def move(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    command_name: str = "forward_velocity"
    ) -> torch.Tensor:

    robot = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)

    robot_vel = quat_apply_inverse(robot.data.root_quat_w, robot.data.root_lin_vel_w[:, :3])
    robot_vel_x = torch.abs(robot_vel[:, 0])
    command_x = torch.abs(command[:, 0])

    # if slower than 90% of the command -> not moving
    is_still = robot_vel_x < (0.9 * command_x)

    return command_x * is_still


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

    robot_vel_x = torch.abs(robot.data.root_lin_vel_b[:, 0].squeeze().clone())
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
    num_penetrating_points = torch.count_nonzero(depths, dim=-1).float()

    # TOTAL
    penalty = (weight_violation * num_penetrating_points + weight_depth * total_depth) * robot_vel_x

    if debug_vis:
        is_point_inside = depths > 0.0
        PENETRATION_MANAGER.visualize(is_inside_list=is_point_inside, env_ids=[0])

    return penalty