from isaaclab.utils import configclass
import isaaclab.sim as sim_utils


from Robot_Parkour.tasks.manager_based.robot_parkour.robot_parkour_env_cfg import RobotParkourEnvCfg
from isaaclab_assets.robots.unitree import UNITREE_GO2_CFG  # isort: skip


# SOFT
@configclass
class Go2FieldSoftEnvCfg(RobotParkourEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.robot = UNITREE_GO2_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.scene.robot.actuators["base_legs"].stiffness = 50.0
        self.scene.robot.actuators["base_legs"].damping = 1.0

# HARD
@configclass
class Go2FieldHardEnvCfg(Go2FieldSoftEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.obstacle.spawn.physics_material = sim_utils.RigidBodyMaterialCfg(
                static_friction=1.0,
                dynamic_friction=1.0,
                restitution=0.0,
            )
        self.scene.obstacle.spawn.collision_props.collision_enabled = True
        self.rewards.penetration.weight = 0.0




# PLAY --------------------------------------
@configclass
class Go2FieldSoftEnvCfg_Play(Go2FieldSoftEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        self.scene.num_envs = 50
        self.observations.policy.enable_corruption = False
        self.scene.terrain.max_init_terrain_level = None

        if self.scene.terrain.terrain_generator is not None:
            self.scene.terrain.terrain_generator.num_rows = 5
            self.scene.terrain.terrain_generator.num_cols = 5
            self.scene.terrain.terrain_generator.curriculum = False

@configclass
class Go2FieldHardEnvCfg_Play(Go2FieldHardEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        self.scene.num_envs = 50
        self.observations.policy.enable_corruption = False
        self.scene.terrain.max_init_terrain_level = None

        if self.scene.terrain.terrain_generator is not None:
            self.scene.terrain.terrain_generator.num_rows = 5
            self.scene.terrain.terrain_generator.num_cols = 5
            self.scene.terrain.terrain_generator.curriculum = False





