from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from Robot_Parkour.tasks.manager_based.robot_parkour.config.go2.config import SimConfig
from Robot_Parkour.tasks.manager_based.robot_parkour.utils.penetration_points import PenetrationManager

from isaaclab.managers import SceneEntityCfg
from isaaclab_assets.robots.unitree import UNITREE_GO2_CFG
from isaaclab.assets import RigidObject



if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv

PENETRATION_MANAGER = None


def randomize_motor_strenght(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor | None,
    range: tuple[float, float],
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
    ):

    if range[0] > range[1]:
        raise ValueError(f"First value in tuple is bigger than second.")

    robot = env.scene[asset_cfg.name]

    default_strenght = UNITREE_GO2_CFG.actuators["base_legs"].effort_limit

    scale = torch.rand(len(env_ids), device="cuda")
    scale = scale * ( range[1] - range[0] ) + range[0]

    new_strenghts = default_strenght * scale
    new_strenghts = new_strenghts.unsqueeze(1).repeat(1,12)

    robot.actuators["base_legs"].effort_limit[env_ids] = new_strenghts



def debug_visualize_body_points(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor | None,
    ):

    global PENETRATION_MANAGER

    if PENETRATION_MANAGER is None:
        points_cfg = SimConfig.body_measure_points
        PENETRATION_MANAGER = PenetrationManager(env, points_cfg)
        PENETRATION_MANAGER.visualizer.set_visibility(True)

    # Update and visualize
    PENETRATION_MANAGER.visualize(env_ids=torch.tensor([0], device=env.device))


def reset_pos_obstacles_climb(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor | None,
    pos_xy: tuple[float, float],
    range_z: tuple[float, float],
    obstacle_cfg: SceneEntityCfg = SceneEntityCfg("obstacle"),
    ):

    obstacle: RigidObject = env.scene[obstacle_cfg.name]
    root_state = obstacle.data.default_root_state[env_ids].clone()

    height_of_obstacle = obstacle.cfg.spawn.size[2]

    pos_x, pos_y = pos_xy
    min_z, max_z = range_z

    x = torch.ones(len(env_ids), device=env.device) * pos_x
    y = torch.ones(len(env_ids), device=env.device) * pos_y

    num_rows = env.scene.terrain.terrain_origins.shape[0]

    terrain_rows = env.scene.terrain.terrain_levels[env_ids].float()
    difficulty = terrain_rows / (num_rows-1)

    z = min_z + difficulty * (max_z - min_z)

    env_origins = env.scene.env_origins[env_ids]
    root_state[:, 0] = env_origins[:, 0] + x
    root_state[:, 1] = env_origins[:, 1] + y
    root_state[:, 2] = env_origins[:, 2] + z - height_of_obstacle/2

    obstacle.write_root_pose_to_sim(root_state[:, :7], env_ids=env_ids)
    obstacle.write_root_velocity_to_sim(root_state[:, 7:], env_ids=env_ids)


def reset_pos_obstacles_tilt(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor | None,
    pos_x: float,
    range_gap: tuple[float, float],
    obstacle_cfg: SceneEntityCfg = SceneEntityCfg("obstacle"),
    ):

    if not hasattr(env, "gap_width"):
        env.gap_width = torch.zeros(env.num_envs, device=env.device, dtype=torch.float)

    obstacle: RigidObject = env.scene[obstacle_cfg.name]
    root_state = obstacle.data.default_root_state[env_ids].clone()

    gap_min, gap_max = range_gap
    width_obstacle = obstacle.cfg.spawn.size[1]
    height_of_obstacle = obstacle.cfg.spawn.size[2]

    num_rows = env.scene.terrain.terrain_origins.shape[0]

    terrain_rows = env.scene.terrain.terrain_levels[env_ids].float()
    difficulty = terrain_rows / (num_rows-1)

    gap_width = gap_max - (gap_max - gap_min) * difficulty
    y = width_obstacle/2.0 + gap_width

    x = torch.ones(len(env_ids), device=env.device) * pos_x
    # y = torch.ones(len(env_ids), device=env.device) * pos_y

    env_origins = env.scene.env_origins[env_ids]
    root_state[:, 0] = env_origins[:, 0] + x
    root_state[:, 1] = env_origins[:, 1] - y
    root_state[:, 2] = env_origins[:, 2] + height_of_obstacle/2

    env.gap_width[env_ids] = gap_width

    obstacle.write_root_pose_to_sim(root_state[:, :7], env_ids=env_ids)
    obstacle.write_root_velocity_to_sim(root_state[:, 7:], env_ids=env_ids)