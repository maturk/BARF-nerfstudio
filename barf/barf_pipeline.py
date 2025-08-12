import typing
from dataclasses import dataclass, field
from typing import Literal, Optional, Type
import os
import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from nerfstudio.cameras.camera_optimizers import CameraOptimizerConfig
from nerfstudio.data.datamanagers.base_datamanager import (
    DataManagerConfig,
    VanillaDataManager,
    VanillaDataManagerConfig,
)
from torch.cuda.amp.grad_scaler import GradScaler

from nerfstudio.configs.base_config import PrintableConfig
from nerfstudio.data.dataparsers.nerfstudio_dataparser import NerfstudioDataParserConfig
from nerfstudio.models.base_model import Model, ModelConfig
from nerfstudio.pipelines.base_pipeline import (
    VanillaPipeline,
    VanillaPipelineConfig,
)
from nerfstudio.utils import profiler
from barf.barf import BARFFreqModelConfig
from barf.utils.exporter_utils import save_poses, create_pose_gif


@dataclass
class PosesConfig(PrintableConfig):
    save_poses: bool = True
    """save poses for visualization"""
    cam_depth: float = 1.0
    """camera depth for visualization"""
    xlim: tuple = (-3, 3)
    """x-axis limit for visualization"""
    ylim: tuple = (-3, 3)
    """y-axis limit for visualization"""
    zlim: tuple = (-3, 2.4)
    """z-axis limit for visualization"""
    # poses_dir: str = "[IGNORE]: set_by_trainer.py"
    poses_dir: str = os.getcwd() + "/poses" # See issue on README.md for more info.


@dataclass
class BARFPipelineConfig(VanillaPipelineConfig):
    """Configuration for pipeline instantiation"""

    _target: Type = field(default_factory=lambda: BARFPipeline)
    """target class to instantiate"""
    datamanager: DataManagerConfig = VanillaDataManagerConfig(
        dataparser=NerfstudioDataParserConfig(),
        train_num_rays_per_batch=1024,
        eval_num_rays_per_batch=1024,
    )
    """specifies the datamanager config"""
    model: ModelConfig = BARFFreqModelConfig(
        camera_optimizer=CameraOptimizerConfig(  # camera_optimizer is moved to model config
            mode="SO3xR3",
        ),
    )
    """specifies the model config"""

    vis_config: PosesConfig = PosesConfig()


class BARFPipeline(VanillaPipeline):
    def __init__(
        self,
        config: BARFPipelineConfig,
        device: str,
        test_mode: Literal["test", "val", "inference"] = "val",
        world_size: int = 1,
        local_rank: int = 0,
        grad_scaler: Optional[GradScaler] = None,
    ):
        super().__init__(
            config=config,
            device=device,
            test_mode=test_mode,
            world_size=world_size,
            local_rank=local_rank,
            grad_scaler=grad_scaler,
        )
        self.poses_dir = ""  # directory to save poses for visualization is set by trainer.py (must fix though)

    @profiler.time_function
    def get_eval_image_metrics_and_images(self, step: int):
        """This function gets your evaluation loss dict. It needs to get the data
        from the DataManager and feed it to the model's forward function

        Args:
            step: current iteration step
        """
        self.eval()
        camera, batch = self.datamanager.next_eval_image(step)
        outputs = self.model.get_outputs_for_camera(camera)
        metrics_dict, images_dict = self.model.get_image_metrics_and_images(outputs, batch)

        if self.model.camera_optimizer.config.mode == "SO3xR3" or self.model.camera_optimizer.config.mode == "SE3":
            assert "camera_opt_translation" not in metrics_dict
            metrics_dict["camera_opt_translation"] = torch.norm(
                self.model.camera_optimizer.pose_adjustment[:, :3], dim=-1
            ).mean()
            assert "camera_opt_rotation" not in metrics_dict
            metrics_dict["camera_opt_rotation"] = torch.norm(
                self.model.camera_optimizer.pose_adjustment[:, 3:], dim=-1
            ).mean()

        assert "num_rays" not in metrics_dict
        metrics_dict["num_rays"] = (camera.height * camera.width * camera.size).item()

        if self.config.vis_config.save_poses:
            fig_as_np_array, error = save_poses(step, self.config.vis_config, self.datamanager.train_dataset, self.model.camera_optimizer)
            images_dict["poses"] = torch.Tensor(fig_as_np_array)
            metrics_dict["rotation_error"] = np.rad2deg(error.R.mean().cpu())
            metrics_dict["translation_error"] = np.rad2deg(error.t.mean())
            create_pose_gif(self.config.vis_config.poses_dir)

        self.train()
        return metrics_dict, images_dict
