from isaaclab.utils import configclass

from Robot_Parkour.tasks.manager_based.robot_parkour.robot_parkour_env_cfg import RobotParkourEnvCfg
from isaaclab_assets.robots.unitree import UNITREE_GO2_CFG  # isort: skip


@configclass
class Go2FlatEnvCfg(RobotParkourEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.robot = UNITREE_GO2_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

