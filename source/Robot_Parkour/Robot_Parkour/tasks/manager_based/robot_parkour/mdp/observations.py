from __future__ import annotations

import torch
from typing import TYPE_CHECKING
import numpy as np
import torch.nn.functional as F

from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import euler_xyz_from_quat, quat_apply_inverse
from isaaclab_assets.robots.unitree import UNITREE_GO2_CFG


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

    default_strength = UNITREE_GO2_CFG.actuators["base_legs"].effort_limit
    current_strength = robot.actuators["base_legs"].effort_limit

    scale = current_strength / default_strength

    return scale


def terrain_friction(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    ) -> torch.Tensor:

    robot = env.scene[asset_cfg.name]
    friction = robot.root_physx_view.get_material_properties()[:, :, :2]

    return friction[:, 0, :].to(env.device)


def distance_from_obstacle(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    obstacle_cfg: SceneEntityCfg = SceneEntityCfg("obstacle")
    ) -> torch.Tensor:

    robot = env.scene[asset_cfg.name]
    obstacle = env.scene[obstacle_cfg.name]
    env_origin_x = env.scene.env_origins[:, 0]

    robot_pos_x = robot.data.root_pos_w[:, 0]
    obstacle_pos_x = obstacle.data.root_pos_w[:, 0]
    obstacle_size_x = obstacle.cfg.spawn.size[0]

    robot_pos = robot_pos_x - env_origin_x
    obstacle_pos = obstacle_pos_x - env_origin_x
    obstacle_front_x = obstacle_pos - (obstacle_size_x / 2.0)

    distance_x = obstacle_front_x - robot_pos

    return distance_x.unsqueeze(1)


def height_obstacle(
    env: ManagerBasedRLEnv,
    obstacle_cfg: SceneEntityCfg = SceneEntityCfg("obstacle")
    ) -> torch.Tensor:
    """ Here the logic comes from the positioning of the obstacles in
    the scene. In the event reset_pos_obstacles are the details
    """

    obstacle = env.scene[obstacle_cfg.name]
    env_origins_z = env.scene.env_origins[:, 2]

    obstacle_pos_z = obstacle.data.root_pos_w[:, 2]
    obstacle_height = obstacle.cfg.spawn.size[2]

    obstacle_pos_z_local = obstacle_pos_z - env_origins_z

    height = obstacle_pos_z_local + (obstacle_height / 2.0)

    return height.unsqueeze(1)


def width_obstacle(
    env: ManagerBasedRLEnv,
    is_tilt: bool = False,
    obstacle_cfg: SceneEntityCfg = SceneEntityCfg("obstacle")
    ) -> torch.Tensor:

    obstacle = env.scene[obstacle_cfg.name]
    if not hasattr(env, "gap_width"):
        env.gap_width = torch.zeros(env.scene.num_envs, device=env.device, dtype=torch.float)

    if is_tilt:
        return env.gap_width.unsqueeze(1)
    else:
        obstacle_width = obstacle.cfg.spawn.size[1]
        return obstacle_width * torch.ones(env.scene.num_envs, device=env.device).unsqueeze(1)


def one_hot_category(
    env: ManagerBasedRLEnv,
    category_id: int = 0,
    num_categories: int = 4
    ) -> torch.Tensor:

    indices = torch.full((env.num_envs,), category_id, device=env.device, dtype=torch.long)
    one_hot = F.one_hot(indices, num_classes=num_categories)

    return one_hot.float()