
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
    box_size: tuple[float, float, float] = (1.0, 2.0, 1.0)





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
        # "random_rough": terrain_gen_hf.HfRandomUniformTerrainCfg(
        #     proportion=0.2, noise_range=(0.02, 0.10), noise_step=0.02, border_width=0.25
        # ),
        # "boxed": terrain_gen_trimesh.MeshRepeatedBoxesTerrainCfg(
        #     proportion=0.4,
        #     object_type="box",
        #     platform_height=0.0,
        #     object_params_start=terrain_gen_trimesh.MeshRepeatedBoxesTerrainCfg.ObjectCfg(
        #         num_objects=2,
        #         height=2,
        #         size=(0.5, 1)),
        #     object_params_end=terrain_gen_trimesh.MeshRepeatedBoxesTerrainCfg.ObjectCfg(
        #         num_objects=5,
        #         height=3,
        #         size=(0.5, 1)),
        # ),
        "boxes": SingleBoxTerrainCfg(
            proportion=1.0,
            # size=(2, 4)
        )
    },
)