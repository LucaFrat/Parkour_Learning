
import torch
from isaaclab.utils.math import quat_apply_inverse, wrap_to_pi


def update_commands_goal_based(env):
    """
    Computes velocity commands pointing towards the goal.
    Should be called in pre_physics_step or as a command generator.
    """

    robot = env.scene["robot"]
    root_pos_w = robot.data.root_pos_w
    root_quat_w = robot.data.root_quat_w


    env_origins = env.scene.env_origins

    # (Assuming track aligns with world X)
    goal_pos_w = env_origins.clone()
    # Look ahead distance (e.g., 4.0m or end of track)
    goal_pos_w[:, 0] += root_pos_w[:, 0] + 4.0
    goal_pos_w[:, 1] = env_origins[:, 1] #(y=0 relative to origin)


    target_vec_w = goal_pos_w - root_pos_w
    # Rotate into robot body frame
    target_vec_b = quat_apply_inverse(root_quat_w, target_vec_w)


    cmd_x = torch.clip(target_vec_b[:, 0], min=0.0, max=1.5)

    cmd_y = torch.clip(target_vec_b[:, 1], min=-0.5, max=0.5)

    target_yaw = torch.atan2(target_vec_b[:, 1], target_vec_b[:, 0])
    cmd_yaw = torch.clip(target_yaw, min=-1.0, max=1.0)