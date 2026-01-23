# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Common functions that can be used to create curriculum for the learning environment.

The functions can be passed to the :class:`isaaclab.managers.CurriculumTermCfg` object to enable
the curriculum introduced by the function.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from collections.abc import Sequence
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg
from isaaclab.terrains import TerrainImporter

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def terrain_levels(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    obstacle_cfg: SceneEntityCfg = SceneEntityCfg("obstacle")
    ) -> torch.Tensor:

    """
    Condition to move up with the difficulty:
    2. robot is closer than 1.0 meters to the obstacle

    Condition to move down:
    5. robot is closer than 1.0 to the env_origin
    """

    robot = env.scene[asset_cfg.name]
    obstacle = env.scene[obstacle_cfg.name]
    terrain = env.scene.terrain

    robot_pos_w = robot.data.root_pos_w[env_ids, :2]
    obstacle_pos_w = obstacle.data.root_pos_w[env_ids, :2]
    env_origin_w = env.scene.env_origins[env_ids, :2]

    robot_pos = robot_pos_w - env_origin_w
    obstacle_pos = obstacle_pos_w - env_origin_w
    obstacle_pos[:, 0] += env.cfg.commands.forward_velocity.goal_distance_behind_obstacle

    obstacle_robot_distance = torch.norm(obstacle_pos - robot_pos, dim=-1)

    # CONDITIONS
    # cosine similarity
    cos_similarity = F.cosine_similarity(robot_pos, obstacle_pos, dim=-1)
    condition_1 = cos_similarity > 0.
    # robot close to obstacle
    condition_2 = torch.norm(obstacle_robot_distance, dim=-1) < 1.
    # robot after obstacle
    condition_3 = torch.norm(robot_pos, dim=-1) > torch.norm(obstacle_pos, dim=-1)
    condition_4 = torch.logical_not(condition_1)
    # robot is closer than 1.0 meters to the origin
    condition_5 = torch.norm(robot_pos, dim=-1) < 1.

    # CHANGE DIFFICULTY
    # move_up = condition_1 & condition_2 & condition_3
    move_up = condition_2
    move_down = condition_5
    # move_down = condition_4 & condition_5

    move_down *= ~move_up
    terrain.update_env_origins(env_ids, move_up, move_down)

    return torch.mean(terrain.terrain_levels.float())
