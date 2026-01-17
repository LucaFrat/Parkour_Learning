# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import math
from dataclasses import MISSING

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils import configclass
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils.assets import ISAACLAB_NUCLEUS_DIR, ISAAC_NUCLEUS_DIR
from isaaclab.sensors import TiledCameraCfg, ContactSensorCfg
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise



from . import mdp


@configclass
class RobotParkourSceneCfg(InteractiveSceneCfg):

    robot: ArticulationCfg = MISSING

    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="generator",
        terrain_generator=mdp.TERRAIN_CFG,
        max_init_terrain_level=5,
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
        ),
        visual_material=sim_utils.MdlFileCfg(
            mdl_path=f"{ISAACLAB_NUCLEUS_DIR}/Materials/TilesMarbleSpiderWhiteBrickBondHoned/TilesMarbleSpiderWhiteBrickBondHoned.mdl",
            project_uvw=True,
            texture_scale=(0.25, 0.25),
        ),
        debug_vis=False,
    )

    contact_forces = ContactSensorCfg(prim_path="{ENV_REGEX_NS}/Robot/.*", history_length=3, track_air_time=True)

    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(
            intensity=750.0,
            texture_file=f"{ISAAC_NUCLEUS_DIR}/Materials/Textures/Skies/PolyHaven/kloofendal_43d_clear_puresky_4k.hdr",
        ),
    )

    obstacle = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Obstacle",
        spawn=sim_utils.CuboidCfg(
            size=(1.0, 4.0, 1.0),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.2, 0.2), opacity=1.0),
            # Penetrable (Visual Only)
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=False),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                kinematic_enabled=True,
                disable_gravity=True,
            ),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0)),
    )

    # depth_camera = TiledCameraCfg(
    #     prim_path="{ENV_REGEX_NS}/Robot/base/camera",
    #     update_period=5, # 10Hz
    #     height=64,
    #     width=80,
    #     debug_vis=False,
    #     data_types=["depth"],
    #     spawn=sim_utils.PinholeCameraCfg(
    #         focal_length=24.0, focus_distance=400.0, horizontal_aperture=20.955, clipping_range=(0.1, 8.0)
    #     ),
    #     offset=CameraCfg.OffsetCfg(pos=(-0.4, 0.0, 0.1), rot=(0.5, -0.5, -0.5, 0.5), convention="ros"),
    # )


@configclass
class CommandsCfg:

    forward_velocity = mdp.UniformVelocityCommandCfg(
        asset_name="robot",
        resampling_time_range=(10.0, 10.0),
        rel_standing_envs=0.02,
        rel_heading_envs=1.0,
        heading_command=True,
        heading_control_stiffness=0.5,
        debug_vis=True,
        ranges=mdp.UniformVelocityCommandCfg.Ranges(
            lin_vel_x=(1.2, 1.2), lin_vel_y=(0.0, 0.0), ang_vel_z=(0.0, 0.0), heading=(0.0, 0.0)
        ),
    )


@configclass
class ActionsCfg:
    """Action specifications for the MDP."""

    joint_pos_hips = mdp.JointPositionActionCfg(asset_name="robot", joint_names=[".*hip_joint"], scale=0.4, use_default_offset=True)
    joint_pos_thigh_knee = mdp.JointPositionActionCfg(asset_name="robot", joint_names=[".*thigh_joint", ".*calf_joint"], scale=0.6, use_default_offset=True)


@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""

        # proprioceptive R29
        roll_pitch = ObsTerm(func=mdp.roll_pitch)
        joint_pos_rel = ObsTerm(func=mdp.joint_pos_rel)
        joint_vel_rel = ObsTerm(func=mdp.joint_vel_rel)
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, noise=Unoise(n_min=-0.2, n_max=0.2))

        actions = ObsTerm(func=mdp.last_action)

        def __post_init__(self) -> None:
            self.enable_corruption = False
            self.concatenate_terms = True

    @configclass
    class Privileged_Physical(ObsGroup):
        """Privileged Physical Information"""

        center_of_mass = ObsTerm(func=mdp.center_of_mass)
        motor_strength = ObsTerm(func=mdp.motor_strength)
        terrain_friction = ObsTerm(func=mdp.terrain_friction)

        def __post_init__(self) -> None:
            self.enable_corruption = False
            self.concatenate_terms = True

    @configclass
    class Privileged_Visual(ObsGroup):
        """Privileged Visual Information"""
        def __post_init__(self) -> None:
            self.enable_corruption = False
            self.concatenate_terms = True


    # observation groups
    policy: PolicyCfg = PolicyCfg()
    physics: Privileged_Physical = Privileged_Physical()
    # visual: Privileged_Visual = Privileged_Visual()


@configclass
class EventCfg:
    """Configuration for events."""

    reset_obstacle = EventTerm(
        func=mdp.reset_pos_obstacles,
        mode="reset",
        params={
            "obstacle_cfg": SceneEntityCfg("obstacle"),
            "pos_xy": (3.0, 0.0),
            "range_z": (0.2, 0.45)
        }
    )

    motor_strengh = EventTerm(
        func=mdp.randomize_motor_strenght,
        mode="reset",
        params={
            "range": (0.9, 1.1)
        }
    )

    physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": (0.7, 1.0),
            "dynamic_friction_range": (0.5, 0.7),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 64,
        },
    )

    base_com = EventTerm(
        func=mdp.randomize_rigid_body_com,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="base"),
            "com_range": {"x": (-0.05, 0.15), "y": (-0.1, 0.1), "z": (-0.05, 0.05)},
        },
    )

    add_base_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="base"),
            "mass_distribution_params": (1.0, 3.0),
            "operation": "add",
        },
    )

    reset_base = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": (-0.3, 0.3), "y": (-0.3, 0.3), "yaw": (0.0, 0.0)},
            "velocity_range": {
                "x": (-0.0, 0.0),
                "y": (-0.0, 0.0),
                "z": (-0.0, 0.0),
                "roll": (-0.0, 0.0),
                "pitch": (-0.0, 0.0),
                "yaw": (-0.0, 0.0),
            },
        },
    )

    reset_robot_joints = EventTerm(
        func=mdp.reset_joints_by_scale,
        mode="reset",
        params={
            "position_range": (0.8, 1.2),
            "velocity_range": (0.0, 0.0),
        },
    )



@configclass
class RewardsCfg:
    """Reward terms for the MDP."""

    # FORWARD
    forward_velocity = RewTerm(
        func=mdp.forward_velocity,
        weight= -1.0,
        params={"command_name": "forward_velocity"}
    )
    lateral_velocity = RewTerm(
        func=mdp.lateral_velocity,
        weight= -1.0
    )
    yaw_rate = RewTerm(
        func=mdp.yaw_rate,
        weight= 0.1
    )

    # ENERGY
    energy_usage = RewTerm(
        func=mdp.energy_usage,
        weight= 2e-6
    )

    # ALIVE
    alive = RewTerm(func=mdp.is_alive, weight=2.0)

    # PENETRATE
    penetration = RewTerm(
        func=mdp.obstacle_penetration,
        weight= -1.0,
        params={
            "weight_violation": 1e-2,
            "weight_depth": 1e-2,
            "debug_vis": False,
        }
    )


@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    base_contact = DoneTerm(
        func=mdp.illegal_contact,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names="base"), "threshold": 1.0},
    )




@configclass
class RobotParkourEnvCfg(ManagerBasedRLEnvCfg):
    # Scene settings
    scene: RobotParkourSceneCfg = RobotParkourSceneCfg(num_envs=2048, env_spacing=4)
    # Basic settings
    observations: ObservationsCfg = ObservationsCfg()
    commands: CommandsCfg = CommandsCfg()
    actions: ActionsCfg = ActionsCfg()
    events: EventCfg = EventCfg()
    # MDP settings
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()

    # Post initialization
    def __post_init__(self) -> None:
        """Post initialization."""
        # general settings
        self.decimation = 4
        self.episode_length_s = 10
        # viewer settings
        self.viewer.eye = (8.0, 0.0, 5.0)
        # simulation settings
        self.sim.dt = 1 / 120
        self.sim.render_interval = self.decimation