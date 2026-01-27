from isaaclab.utils import configclass
import isaaclab.sim as sim_utils


from Robot_Parkour.tasks.manager_based.robot_parkour.robot_parkour_env_cfg import RobotParkourEnvCfg
from isaaclab_assets.robots.unitree import UNITREE_GO2_CFG  # isort: skip

from Robot_Parkour.tasks.manager_based.robot_parkour import mdp

# ============================== SOFT ==============================
@configclass
class Go2ClimbSoftEnvCfg(RobotParkourEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.terrain.terrain_generator = mdp.TERRAIN_CFG_CLIMB_SOFT
        self.events.reset_obstacle_tilt = None
        self.scene.robot = UNITREE_GO2_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.scene.robot.actuators["base_legs"].stiffness = 40.0
        self.scene.robot.actuators["base_legs"].damping = 1.0


@configclass
class Go2TiltSoftEnvCfg(RobotParkourEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.terrain.terrain_generator = mdp.TERRAIN_CFG_TILT_SOFT
        self.scene.robot = UNITREE_GO2_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.scene.robot.actuators["base_legs"].stiffness = 40.0
        self.scene.robot.actuators["base_legs"].damping = 1.0
        self.events.reset_obstacle_climb = None
        self.scene.obstacle.spawn.size = (0.6, 2.0, 0.9)
        self.commands.forward_velocity.goal_pos_for_tilt = True
        self.observations.visual.width_obstacle.params["is_tilt"] = True






# ============================== HARD ==============================




@configclass
class Go2ClimbHardEnvCfg(Go2ClimbSoftEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.terrain.terrain_generator = mdp.TERRAIN_CFG_CLIMB_HARD
        self.scene.obstacle.spawn.physics_material = sim_utils.RigidBodyMaterialCfg(
                static_friction=1.0,
                dynamic_friction=1.0,
                restitution=0.0,
            )
        self.rewards.penetration.weight = 0.0
        self.scene.terrain.max_init_terrain_level = 5

@configclass
class Go2TiltHardEnvCfg(Go2TiltSoftEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.terrain.terrain_generator = mdp.TERRAIN_CFG_TILT_HARD
        self.scene.obstacle.spawn.physics_material = sim_utils.RigidBodyMaterialCfg(
                static_friction=1.0,
                dynamic_friction=1.0,
                restitution=0.0,
            )
        self.rewards.penetration.weight = 0.0
        self.scene.terrain.max_init_terrain_level = 5





# PLAY --------------------------------------


@configclass
class Go2ClimbSoftEnvCfg_Play(Go2ClimbSoftEnvCfg):
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
class Go2ClimbHardEnvCfg_Play(Go2ClimbHardEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        self.scene.num_envs = 50
        self.observations.policy.enable_corruption = False
        self.scene.terrain.max_init_terrain_level = None

        if self.scene.terrain.terrain_generator is not None:
            self.scene.terrain.terrain_generator.num_rows = 5
            self.scene.terrain.terrain_generator.num_cols = 5
            self.scene.terrain.terrain_generator.curriculum = True





