from isaaclab.utils import configclass

from Robot_Parkour.tasks.manager_based.robot_parkour.robot_parkour_env_cfg import RobotParkourEnvCfg
from isaaclab_assets.robots.unitree import UNITREE_GO2_CFG  # isort: skip


@configclass
class Go2FieldEnvCfg(RobotParkourEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.robot = UNITREE_GO2_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.scene.robot.actuators["base_legs"].stiffness = 50.0
        self.scene.robot.actuators["base_legs"].damping = 1.0


@configclass
class Go2FieldEnvCfg_Play(Go2FieldEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        self.scene.num_envs = 50
        self.observations.policy.enable_corruption = False
        self.scene.terrain.max_init_terrain_level = None

        if self.scene.terrain.terrain_generator is not None:
            self.scene.terrain.terrain_generator.num_rows = 5
            self.scene.terrain.terrain_generator.num_cols = 5
            self.scene.terrain.terrain_generator.curriculum = False


