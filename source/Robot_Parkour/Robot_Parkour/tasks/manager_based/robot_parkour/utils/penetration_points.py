import torch
import itertools
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
from isaaclab.markers.config import SPHERE_MARKER_CFG
from isaaclab.utils.math import quat_from_euler_xyz, quat_apply
import isaaclab.sim as sim_utils



class PenetrationManager:
    def __init__(self, env, measure_points_cfg):
        self.env = env
        self.robot = env.scene["robot"]
        self.device = self.env.device
        self.cfg = measure_points_cfg

        # --- Visualization Setup ---
        # marker_cfg = SPHERE_MARKER_CFG.copy()
        # marker_cfg.markers["sphere"].radius = 0.015
        # marker_cfg.prim_path = "/Visuals/BodyPoints"

        marker_cfg = VisualizationMarkersCfg(
                prim_path="/Visuals/BodyPoints",
                markers={
                    "marker_red": sim_utils.SphereCfg(
                        radius=0.01,
                        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 1.0, 0.0)),
                    ),
                    "marker_green": sim_utils.SphereCfg(
                        radius=0.01,
                        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0)),
                    ),
                }
            )
        self.visualizer = VisualizationMarkers(marker_cfg)

        self._init_body_points_vectorized()


    def _init_body_points_vectorized(self):
        body_names = self.robot.data.body_names

        all_local_points = []
        all_body_indices = []

        print(f"[PenetrationManager] initializing points...")

        for body_idx, body_name in enumerate(body_names):
            matched_key = next((key for key in self.cfg.keys() if key in body_name.lower()), None)

            if matched_key:
                data = self.cfg[matched_key]
                xs, ys, zs = data.get('x', [0.]), data.get('y', [0.]), data.get('z', [0.])
                transform = data.get('transform', [0.]*6)

                # Create Grid
                points_list = list(itertools.product(xs, ys, zs))
                local_pts = torch.tensor(points_list, device=self.device, dtype=torch.float32)

                # Apply Static Rotation (from config)
                roll, pitch, yaw = transform[3], transform[4], transform[5]
                if roll != 0 or pitch != 0 or yaw != 0:
                    q_rot = quat_from_euler_xyz(
                        torch.tensor([roll], device=self.device),
                        torch.tensor([pitch], device=self.device),
                        torch.tensor([yaw], device=self.device)
                    )
                    local_pts = quat_apply(q_rot.expand(len(local_pts), -1), local_pts)

                # Apply Static Translation (from config)
                local_pts += torch.tensor(transform[0:3], device=self.device)

                # Append to lists
                all_local_points.append(local_pts)
                all_body_indices.append(torch.full((len(local_pts),), body_idx, device=self.device, dtype=torch.long))


        self.all_local_points = torch.cat(all_local_points, dim=0)
        self.point_body_indices = torch.cat(all_body_indices, dim=0)

        self.total_points = len(self.all_local_points)
        print(f"[PenetrationManager] Optimized: Managing {self.total_points} points total.")


    def compute_world_points(self, env_ids=None):
        """
        Vectorized computation of world points.
        """

        if env_ids == None:
            env_ids = torch.arange(0, self.env.scene.num_envs, device=self.env.device, dtype=int)

        # (Num_Envs, Num_Bodies, 3/4)
        body_pos = self.robot.data.body_pos_w[env_ids]
        body_quat = self.robot.data.body_quat_w[env_ids]

        # Expand Poses to match Points
        active_pos = body_pos[:, self.point_body_indices, :]
        active_quat = body_quat[:, self.point_body_indices, :]

        # Broadcast Local Points
        # Shape: (1, Total_Points, 3) -> broadcasts to (Num_Envs, Total_Points, 3)
        local_pts_batch = self.all_local_points.unsqueeze(0)

        # Transform: R * p + T
        num_envs = len(env_ids)

        flat_quat = active_quat.reshape(-1, 4)
        flat_local = local_pts_batch.expand(num_envs, -1, -1).reshape(-1, 3)

        rotated_pts = quat_apply(flat_quat, flat_local)

        # Add translation
        world_pts = rotated_pts + active_pos.reshape(-1, 3)

        # Return shape: (Num_Envs * Total_Points, 3)
        return world_pts



    def visualize(self, is_inside_list=None, env_ids=None):
        if not self.visualizer.is_visible():
            return

        if env_ids == None:
            env_ids = torch.arange(0, self.env.scene.num_envs, device=self.env.device, dtype=int)

        points = self.compute_world_points(env_ids)
        is_inside = is_inside_list[env_ids, :].view(-1, 1).squeeze()

        self.visualizer.visualize(translations=points, marker_indices=is_inside)