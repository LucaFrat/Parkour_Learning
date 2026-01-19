
from __future__ import annotations
import torch
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from isaaclab.envs.mdp.commands.velocity_command import UniformVelocityCommand
from isaaclab.envs.mdp.commands.commands_cfg import UniformVelocityCommandCfg
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import quat_apply_inverse, wrap_to_pi
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
import isaaclab.sim as sim_utils


if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv



class GoalBasedVelocityCommand(UniformVelocityCommand):

    cfg: GoalBasedVelocityCommandCfg

    def __init__(self, cfg: GoalBasedVelocityCommandCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self.robot = env.scene[cfg.asset_cfg.name]
        self.obstacle = env.scene[cfg.obstacle_cfg.name]
        self.env = env

        marker_cfg = VisualizationMarkersCfg(
                prim_path="/Visuals/Goal_pos",
                markers={
                    "marker": sim_utils.SphereCfg(
                        radius=0.5,
                        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 1.0, 0.0)),
                    )})
        self.visualizer = VisualizationMarkers(marker_cfg)

    def _resample_command(self, env_ids: torch.Tensor):
        robot_pos_w = self.robot.data.root_pos_w[env_ids, :]
        robot_quat_w = self.robot.data.root_quat_w[env_ids, :]
        env_origins = self.env.scene.env_origins[env_ids, :]
        obstacle_pos_w = self.obstacle.data.root_pos_w[env_ids, :]

        min_vel_x, max_vel_x = self.cfg.ranges.lin_vel_x
        min_vel_y, max_vel_y = self.cfg.ranges.lin_vel_y
        min_ang_vel_z, max_ang_vel_z = self.cfg.ranges.ang_vel_z

        goal_pos = obstacle_pos_w - env_origins
        goal_pos[:, 0] += self.cfg.goal_distance_behind_obstacle

        if self.cfg.debug_goal_vis:
            goal_pos_w = goal_pos + env_origins
            self.visualizer.set_visibility(True)
            self.visualizer.visualize(translations=goal_pos_w)

        robot_pos = robot_pos_w - env_origins
        distance_b = goal_pos - robot_pos

        # Rotate into robot body frame
        target_vec_b = quat_apply_inverse(robot_quat_w, distance_b)

        cmd_x = torch.clip(target_vec_b[:, 0], min=min_vel_x, max=max_vel_x)
        cmd_y = torch.clip(target_vec_b[:, 1], min=min_vel_y, max=max_vel_y)
        target_yaw = torch.atan2(target_vec_b[:, 1], target_vec_b[:, 0])
        cmd_yaw = torch.clip(target_yaw, min=min_ang_vel_z, max=max_ang_vel_z)

        self.vel_command_b[env_ids, 0] = cmd_x
        self.vel_command_b[env_ids, 1] = cmd_y
        self.vel_command_b[env_ids, 2] = cmd_yaw



@dataclass
class GoalBasedVelocityCommandCfg(UniformVelocityCommandCfg):
    class_type: type = GoalBasedVelocityCommand

    asset_cfg: SceneEntityCfg = field(default_factory=lambda: SceneEntityCfg("robot"))
    obstacle_cfg: SceneEntityCfg = field(default_factory=lambda: SceneEntityCfg("obstacle"))

    goal_distance_behind_obstacle: float = 1.0
    debug_goal_vis: bool = False


