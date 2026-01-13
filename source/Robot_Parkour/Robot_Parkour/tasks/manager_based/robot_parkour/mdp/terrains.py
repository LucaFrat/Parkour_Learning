from __future__ import annotations

import isaaclab.terrains as terrain_gen
import trimesh
import numpy as np
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .terrain_cfg import SingleBoxTerrainCfg



def single_box_terrain(
    difficulty: float, cfg: SingleBoxTerrainCfg
    ) -> tuple[list[trimesh.Trimesh], np.ndarray]:

    meshes_list = list()
    terrain_height = 1.0

    # Generate the top box
    dim_z = cfg.box_range_z[0] + (cfg.box_range_z[1] - cfg.box_range_z[0]) * difficulty
    dim = (cfg.box_size_xy[0], cfg.box_size_xy[1], dim_z)
    pos = (cfg.size[0]/2 + 3, cfg.size[1]/2, dim_z/2)
    box_mesh = trimesh.creation.box(dim, trimesh.transformations.translation_matrix(pos))
    meshes_list.append(box_mesh)

    # Generate the ground
    pos = (0.5 * cfg.size[0], 0.5 * cfg.size[1], -terrain_height / 2)
    dim = (cfg.size[0], cfg.size[1], terrain_height)
    ground_mesh = trimesh.creation.box(dim, trimesh.transformations.translation_matrix(pos))
    meshes_list.append(ground_mesh)

    # specify the origin of the terrain
    origin = np.array([pos[0], pos[1], 0])

    return meshes_list, origin