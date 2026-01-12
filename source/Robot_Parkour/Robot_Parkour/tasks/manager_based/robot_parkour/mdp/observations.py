from __future__ import annotations

import torch
from typing import TYPE_CHECKING
import numpy as np

from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import euler_xyz_from_quat

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
