from __future__ import annotations

import torch
from typing import TYPE_CHECKING, Literal

from isaaclab.managers import SceneEntityCfg
from isaaclab_assets.robots.unitree import UNITREE_GO2_CFG


if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv



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