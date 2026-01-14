from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from Robot_Parkour.tasks.manager_based.robot_parkour.config.go2.config import SimConfig
from Robot_Parkour.tasks.manager_based.robot_parkour.utils.penetration_points import PenetrationManager

from isaaclab.managers import SceneEntityCfg
from isaaclab_assets.robots.unitree import UNITREE_GO2_CFG


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