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
    dim = (cfg.box_size[0], cfg.box_size[1], cfg.box_size[2])
    pos = (2, 2, cfg.box_size[2]/2 + 2*difficulty)
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