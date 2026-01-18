
from dataclasses import MISSING

import isaaclab.terrains.height_field as terrain_gen_hf
import isaaclab.terrains.trimesh as terrain_gen_trimesh
from isaaclab.terrains import SubTerrainBaseCfg
from isaaclab.terrains import TerrainGeneratorCfg
from isaaclab.utils import configclass

from .terrains import single_box_terrain



@configclass
class SingleBoxTerrainCfg(SubTerrainBaseCfg):
    function = single_box_terrain
    box_size_xy: tuple[float, float] = (1.0, 6.0)
    box_range_z: tuple[float, float] = (0.2, 0.45)



TERRAIN_CFG = TerrainGeneratorCfg(
    size=(10.0, 6.0),
    curriculum=True,
    border_width=20.0,
    num_rows=8,
    num_cols=8,
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    use_cache=False,
    sub_terrains={
        # "flat": terrain_gen_trimesh.MeshPlaneTerrainCfg(),
        "random_rough": terrain_gen_hf.HfRandomUniformTerrainCfg(
            proportion=1.0, noise_range=(0.02, 0.10), noise_step=0.02, border_width=0.25
        ),
        # "box": SingleBoxTerrainCfg(
        #     proportion=1.0,
        # )
    },
)