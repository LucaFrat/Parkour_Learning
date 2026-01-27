
from dataclasses import MISSING

import isaaclab.terrains.height_field as terrain_gen_hf
import isaaclab.terrains.trimesh as terrain_gen_trimesh
from isaaclab.terrains import SubTerrainBaseCfg
from isaaclab.terrains import TerrainGeneratorCfg
from isaaclab.utils import configclass

from .terrains import single_box_terrain, single_box_tilt_soft_terrain, gap_box_hard_terrain



@configclass
class SingleBoxTerrainCfg(SubTerrainBaseCfg):
    function = single_box_terrain
    box_size_xy: tuple[float, float] = (1.0, 4.0)
    box_pos_xy: tuple[float, float] = (2.0, 0.0)
    box_range_z: tuple[float, float] = (0.01, 0.45)
    grid_width: float = 0.09
    grid_height_range: tuple[float, float] = (0.01, 0.05)

@configclass
class SingleBoxTiltSoftTerrainCfg(SubTerrainBaseCfg):
    function = single_box_tilt_soft_terrain
    box_size_xy: tuple[float, float] = (0.6, 2.0)
    box_size_z: float = 0.8
    box_pos_xy: tuple[float, float] = (2.0, 0.0)
    grid_width: float = 0.09
    grid_height_range: tuple[float, float] = (0.01, 0.05)

@configclass
class GapBoxTerrainCfg(SingleBoxTiltSoftTerrainCfg):
    function = gap_box_hard_terrain
    gap_width: tuple[float, float] = (0.28, 0.36)



TERRAIN_CFG_CLIMB_SOFT = TerrainGeneratorCfg(
    size=(10.0, 10.0),
    curriculum=True,
    border_width=10.0,
    num_rows=8,
    num_cols=8,
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    use_cache=False,
    sub_terrains={
        # "flat": terrain_gen_trimesh.MeshPlaneTerrainCfg(),
        # "random_rough": terrain_gen_hf.HfRandomUniformTerrainCfg(
        #     proportion=1.0, noise_range=(0.01, 0.08), noise_step=0.007, border_width=0.25
        # ),
        # "box": SingleBoxTerrainCfg(
        #     proportion=1.0,
        # ),
        "random": terrain_gen_trimesh.MeshRandomGridTerrainCfg(
            proportion=0.4, grid_width=0.09, grid_height_range=(0.01, 0.05), platform_width=1.5
        )
    },
)

TERRAIN_CFG_TILT_SOFT = TerrainGeneratorCfg(
    size=(10.0, 10.0),
    curriculum=True,
    border_width=10.0,
    num_rows=8,
    num_cols=8,
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    use_cache=False,
    sub_terrains={
        "box": SingleBoxTiltSoftTerrainCfg(
            proportion=1.0,
        ),
    },
)


TERRAIN_CFG_CLIMB_HARD = TerrainGeneratorCfg(
    size=(10.0, 10.0),
    curriculum=True,
    border_width=10.0,
    num_rows=8,
    num_cols=8,
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    use_cache=False,
    sub_terrains={
        "box": SingleBoxTerrainCfg(
            proportion=1.0,
        ),
    },
)

TERRAIN_CFG_TILT_HARD = TerrainGeneratorCfg(
    size=(10.0, 10.0),
    curriculum=True,
    border_width=10.0,
    num_rows=8,
    num_cols=8,
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    use_cache=False,
    sub_terrains={
        "box": GapBoxTerrainCfg(
            proportion=1.0,
        ),
    },
)