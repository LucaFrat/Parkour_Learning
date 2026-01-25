from __future__ import annotations

import isaaclab.terrains as terrain_gen
from isaaclab.terrains.trimesh.utils import make_border
import trimesh
import numpy as np
import torch
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .terrain_cfg import SingleBoxTerrainCfg, DoubleBoxTerrainCfg



def single_box_terrain(
    difficulty: float, cfg: SingleBoxTerrainCfg
    ) -> tuple[list[trimesh.Trimesh], np.ndarray]:

    meshes_list = list()
    terrain_height = 1.0

    # Generate the top box
    dim_z = cfg.box_range_z[0] + (cfg.box_range_z[1] - cfg.box_range_z[0]) * difficulty
    dim = (cfg.box_size_xy[0], cfg.box_size_xy[1], dim_z)
    pos = (cfg.size[0]/2 + cfg.box_pos_xy[0], cfg.size[1]/2 + cfg.box_pos_xy[1], dim_z/2)
    box_mesh = trimesh.creation.box(dim, trimesh.transformations.translation_matrix(pos))
    meshes_list.append(box_mesh)

    # Generate the ground
    pos = (0.5 * cfg.size[0], 0.5 * cfg.size[1], -terrain_height / 2)
    dim = (cfg.size[0], cfg.size[1], terrain_height)
    ground_mesh = trimesh.creation.box(dim, trimesh.transformations.translation_matrix(pos))
    meshes_list.append(ground_mesh)

    # create random grid mesh
    borders_grid_meshes, grid_mesh = get_random_grid_mesh(difficulty, cfg, terrain_height)
    meshes_list.append(borders_grid_meshes)
    meshes_list.append(grid_mesh)


    # specify the origin of the terrain
    origin = np.array([pos[0], pos[1], 0])

    return meshes_list, origin



def single_box_tilt_soft_terrain(
    difficulty: float, cfg: SingleBoxTerrainCfg
    ) -> tuple[list[trimesh.Trimesh], np.ndarray]:

    meshes_list = list()
    terrain_height = 1.0

    # Generate the top box
    dim_z = cfg.box_size_z
    dim = (cfg.box_size_xy[0], cfg.box_size_xy[1], dim_z)
    pos = (cfg.size[0]/2 + cfg.box_pos_xy[0], cfg.size[1]/2 + cfg.box_size_xy[1]/2, dim_z/2)
    box_mesh = trimesh.creation.box(dim, trimesh.transformations.translation_matrix(pos))
    meshes_list.append(box_mesh)

    # Generate the ground
    pos = (0.5 * cfg.size[0], 0.5 * cfg.size[1], -terrain_height / 2)
    dim = (cfg.size[0], cfg.size[1], terrain_height)
    ground_mesh = trimesh.creation.box(dim, trimesh.transformations.translation_matrix(pos))
    meshes_list.append(ground_mesh)

    # create random grid mesh
    borders_grid_meshes, grid_mesh = get_random_grid_mesh(difficulty, cfg, terrain_height)
    meshes_list.append(borders_grid_meshes)
    meshes_list.append(grid_mesh)


    # specify the origin of the terrain
    origin = np.array([pos[0], pos[1], 0])

    return meshes_list, origin

def get_random_grid_mesh(
    difficulty: float,
    cfg: SingleBoxTerrainCfg,
    terrain_height: float,
    ) -> list[trimesh.Trimesh]:

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    # check to ensure square terrain
    if cfg.size[0] != cfg.size[1]:
        raise ValueError(f"The terrain must be square. Received size: {cfg.size}.")
    # resolve the terrain configuration
    grid_height = cfg.grid_height_range[0] + difficulty * (cfg.grid_height_range[1] - cfg.grid_height_range[0])


    num_boxes_x = int(cfg.size[0] / cfg.grid_width)
    num_boxes_y = int(cfg.size[1] / cfg.grid_width)

    # # generate the border
    border_width = cfg.size[0] - min(num_boxes_x, num_boxes_y) * cfg.grid_width
    if border_width > 0:
        # compute parameters for the border
        border_center = (0.5 * cfg.size[0], 0.5 * cfg.size[1], -terrain_height / 2)
        border_inner_size = (cfg.size[0] - border_width, cfg.size[1] - border_width)
        # create border meshes
        borders_grid_meshes = make_border(cfg.size, border_inner_size, terrain_height, border_center)
        # meshes_list += make_borders
    else:
        raise RuntimeError("Border width must be greater than 0! Adjust the parameter 'cfg.grid_width'.")

    # create a template grid of terrain height
    grid_dim = [cfg.grid_width, cfg.grid_width, terrain_height]
    grid_position = [0.5 * cfg.grid_width, 0.5 * cfg.grid_width, -terrain_height / 2]
    template_box = trimesh.creation.box(grid_dim, trimesh.transformations.translation_matrix(grid_position))
    # extract vertices and faces of the box to create a template
    template_vertices = template_box.vertices  # (8, 3)
    template_faces = template_box.faces

    # repeat the template box vertices to span the terrain (num_boxes_x * num_boxes_y, 8, 3)
    vertices = torch.tensor(template_vertices, device=device).repeat(num_boxes_x * num_boxes_y, 1, 1)
    # create a meshgrid to offset the vertices
    x = torch.arange(0, num_boxes_x, device=device)
    y = torch.arange(0, num_boxes_y, device=device)
    xx, yy = torch.meshgrid(x, y, indexing="ij")
    xx = xx.flatten().view(-1, 1)
    yy = yy.flatten().view(-1, 1)
    xx_yy = torch.cat((xx, yy), dim=1)
    # offset the vertices
    offsets = cfg.grid_width * xx_yy + border_width / 2
    vertices[:, :, :2] += offsets.unsqueeze(1)

    # add noise to the vertices to have a random height over each grid cell
    num_boxes = len(vertices)
    # create noise for the z-axis
    h_noise = torch.zeros((num_boxes, 3), device=device)
    h_noise[:, 2].uniform_(-grid_height, grid_height)
    # reshape noise to match the vertices (num_boxes, 4, 3)
    # only the top vertices of the box are affected
    vertices_noise = torch.zeros((num_boxes, 4, 3), device=device)
    vertices_noise += h_noise.unsqueeze(1)
    # add height only to the top vertices of the box
    vertices[vertices[:, :, 2] == 0] += vertices_noise.view(-1, 3)
    # move to numpy
    vertices = vertices.reshape(-1, 3).cpu().numpy()

    # create faces for boxes (num_boxes, 12, 3). Each box has 6 faces, each face has 2 triangles.
    faces = torch.tensor(template_faces, device=device).repeat(num_boxes, 1, 1)
    face_offsets = torch.arange(0, num_boxes, device=device).unsqueeze(1).repeat(1, 12) * 8
    faces += face_offsets.unsqueeze(2)
    # move to numpy
    faces = faces.view(-1, 3).cpu().numpy()
    # convert to trimesh
    grid_mesh = trimesh.Trimesh(vertices=vertices, faces=faces)

    return borders_grid_meshes, grid_mesh




