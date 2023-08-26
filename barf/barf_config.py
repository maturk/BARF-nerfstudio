"""
n2n configuration file.
"""

from nerfstudio.cameras.camera_optimizers import CameraOptimizerConfig
from nerfstudio.configs.base_config import ViewerConfig
from nerfstudio.data.datamanagers.base_datamanager import VanillaDataManagerConfig
from nerfstudio.data.dataparsers.instant_ngp_dataparser import InstantNGPDataParserConfig
from nerfstudio.data.dataparsers.nerfstudio_dataparser import NerfstudioDataParserConfig
from nerfstudio.data.dataparsers.blender_dataparser import BlenderDataParserConfig
from nerfstudio.engine.optimizers import AdamOptimizerConfig, RAdamOptimizerConfig
from nerfstudio.engine.schedulers import ExponentialDecaySchedulerConfig
from nerfstudio.engine.trainer import TrainerConfig
from nerfstudio.models.nerfacto import NerfactoModelConfig
from nerfstudio.models.vanilla_nerf import VanillaModelConfig
from nerfstudio.pipelines.base_pipeline import VanillaPipelineConfig
from nerfstudio.plugins.types import MethodSpecification

from barf.barf_pipeline import BARFPipelineConfig
from barf.barf import BARFFreqModelConfig, BARFHashModelConfig, BARFGradientHashModelConfig

max_num_iterations = 200000

barf_freq_method = MethodSpecification(
    config=TrainerConfig(
        method_name="barf-freq",
        steps_per_eval_batch=500,
        steps_per_save=2000,
        max_num_iterations=200000,
        mixed_precision=True,
        pipeline=BARFPipelineConfig(
            datamanager=VanillaDataManagerConfig(
                dataparser=NerfstudioDataParserConfig(),
                train_num_rays_per_batch=1024,  # 1024 for Blender, 2048 for Real-World
                eval_num_rays_per_batch=1024,  # 1024 for Blender, 2048 for Real-World
                # Camera Pose Learning Rate:
                #   Blender: 1e-3 -> 1e-5
                #   Real-World: 3e-3 -> 1e-5
                camera_optimizer=CameraOptimizerConfig(  # Blender synthetic data
                    mode="SO3xR3",
                    optimizer=AdamOptimizerConfig(lr=1e-4, eps=1e-8),
                    scheduler=ExponentialDecaySchedulerConfig(lr_final=1e-7),
                ),
            ),
            model=BARFFreqModelConfig(),
        ),
        optimizers={
            "fields": {
                # Learning Rate:
                #   Blender: 5e-4 -> 1e-4
                #   Real-World: 1e-3 -> 1e-4
                # Original BARF hyperparameter setting
                "optimizer": AdamOptimizerConfig(lr=5e-4, eps=1e-08),
                "scheduler": ExponentialDecaySchedulerConfig(lr_final=1e-4),
            },
        },
        viewer=ViewerConfig(num_rays_per_chunk=1 << 15),
        vis="viewer",
    ),
    description="Vanilla BARF c2f freq encoding implementation.",
)

barf_hash_method = MethodSpecification(
    config=TrainerConfig(
        method_name="barf-hash",
        steps_per_eval_batch=500,
        steps_per_save=2000,
        max_num_iterations=30000,
        mixed_precision=True,
        pipeline=BARFPipelineConfig(
            datamanager=VanillaDataManagerConfig(
                dataparser=NerfstudioDataParserConfig(),
                train_num_rays_per_batch=4096,
                eval_num_rays_per_batch=4096,
                camera_optimizer=CameraOptimizerConfig(  # Blender synthetic data
                    mode="SO3xR3",
                    optimizer=AdamOptimizerConfig(lr=1e-4, eps=1e-8),
                    scheduler=ExponentialDecaySchedulerConfig(lr_final=1e-7),
                ),
            ),
            model=BARFHashModelConfig(
                eval_num_rays_per_chunk=1 << 15,
            ),
        ),
        optimizers={
            "proposal_networks": {
                "optimizer": AdamOptimizerConfig(lr=1e-2, eps=1e-15),
                "scheduler": ExponentialDecaySchedulerConfig(lr_final=0.0001, max_steps=200000),
            },
            "fields": {
                "optimizer": AdamOptimizerConfig(lr=1e-2, eps=1e-15),
                "scheduler": ExponentialDecaySchedulerConfig(lr_final=0.0001, max_steps=200000),
            },
            # "camera_opt": {
            #     "optimizer": AdamOptimizerConfig(lr=1e-3, eps=1e-15),
            #     "scheduler": ExponentialDecaySchedulerConfig(lr_final=1e-4, max_steps=5000),
            # },
        },
        viewer=ViewerConfig(num_rays_per_chunk=1 << 15),
        vis="viewer",
    ),
    description="BARF c2f hash grid encoding implementation.",
)


barf_grad_hash_method = MethodSpecification(
    config=TrainerConfig(
        method_name="barf-grad",
        steps_per_eval_batch=500,
        steps_per_save=2000,
        max_num_iterations=30000,
        mixed_precision=True,
        pipeline=BARFPipelineConfig(
            datamanager=VanillaDataManagerConfig(
                dataparser=NerfstudioDataParserConfig(),
                train_num_rays_per_batch=4096,
                eval_num_rays_per_batch=4096,
                camera_optimizer=CameraOptimizerConfig(  # Blender synthetic data
                    mode="SO3xR3",
                    optimizer=AdamOptimizerConfig(lr=6e-4, eps=1e-8, weight_decay=1e-2),
                    scheduler=ExponentialDecaySchedulerConfig(lr_final=6e-6, max_steps=200000),
                ),
            ),
            model=BARFGradientHashModelConfig(
                eval_num_rays_per_chunk=1 << 15,
            ),
        ),
        optimizers={
            "proposal_networks": {
                "optimizer": AdamOptimizerConfig(lr=1e-2, eps=1e-15),
                "scheduler": ExponentialDecaySchedulerConfig(lr_final=0.0001, max_steps=200000),
            },
            "fields": {
                "optimizer": AdamOptimizerConfig(lr=1e-2, eps=1e-15),
                "scheduler": ExponentialDecaySchedulerConfig(lr_final=0.0001, max_steps=200000),
            },
            # "camera_opt": {
            #     "optimizer": AdamOptimizerConfig(lr=1e-3, eps=1e-15),
            #     "scheduler": ExponentialDecaySchedulerConfig(lr_final=1e-4, max_steps=5000),
            # },
        },
        viewer=ViewerConfig(num_rays_per_chunk=1 << 15),
        vis="viewer",
    ),
    description="BARF gradient scaled hash encoding implementation.",
)
