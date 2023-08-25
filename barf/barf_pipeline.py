import typing
import os
from dataclasses import dataclass, field
from typing import Literal, Optional, Type, Optional
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from nerfstudio.cameras.camera_optimizers import CameraOptimizer, CameraOptimizerConfig
from nerfstudio.data.datamanagers.base_datamanager import (
    DataManagerConfig,
    VanillaDataManager,
    VanillaDataManagerConfig,
)
from torch.cuda.amp.grad_scaler import GradScaler

from nerfstudio.data.dataparsers.nerfstudio_dataparser import NerfstudioDataParserConfig
from nerfstudio.data.datasets.base_dataset import InputDataset
from nerfstudio.engine.optimizers import AdamOptimizerConfig
from nerfstudio.engine.schedulers import ExponentialDecaySchedulerConfig
from nerfstudio.exporter.exporter_utils import (
    collect_camera_poses_for_dataset,
    # collect_init_camera_poses_for_dataset
)
from nerfstudio.models.base_model import Model, ModelConfig
from nerfstudio.pipelines.base_pipeline import (
    VanillaPipeline,
    VanillaPipelineConfig,
)
from nerfstudio.utils import profiler
from barf.barf import BARFFreqModelConfig
from barf.visualizer.util_vis import plot_save_poses_blender, camera
from easydict import EasyDict
import matplotlib.pyplot as plt
import imageio


@dataclass
class BARFPipelineConfig(VanillaPipelineConfig):
    """Configuration for pipeline instantiation"""

    _target: Type = field(default_factory=lambda: BARFPipeline)
    """target class to instantiate"""
    datamanager: DataManagerConfig = VanillaDataManagerConfig(
        dataparser=NerfstudioDataParserConfig(),
        train_num_rays_per_batch=1024,  # 1024 for Blender, 2048 for Real-World
        eval_num_rays_per_batch=1024,  # 1024 for Blender, 2048 for Real-World
        camera_optimizer=CameraOptimizerConfig(  # Blender synthetic data
            mode="SO3xR3",
            optimizer=AdamOptimizerConfig(lr=1e-4, eps=1e-8),
            scheduler=ExponentialDecaySchedulerConfig(lr_final=1e-7),
        ),
    )
    """specifies the datamanager config"""
    model: ModelConfig = BARFFreqModelConfig()
    """specifies the model config"""


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
        super(VanillaPipeline, self).__init__()
        self.config = config
        self.test_mode = test_mode

        self.datamanager: VanillaDataManager = config.datamanager.setup(
            device=device, test_mode=test_mode, world_size=world_size, local_rank=local_rank
        )
        self.datamanager.to(device)
        # TODO(ethan): get rid of scene_bounds from the model
        assert self.datamanager.train_dataset is not None, "Missing input dataset"

        self._model = config.model.setup(
            scene_box=self.datamanager.train_dataset.scene_box,
            num_train_data=len(self.datamanager.train_dataset),
            metadata=self.datamanager.train_dataset.metadata,
        )
        self.model.to(device)

        self.world_size = world_size
        if world_size > 1:
            self._model = typing.cast(Model, DDP(self._model, device_ids=[local_rank], find_unused_parameters=True))
            dist.barrier(device_ids=[local_rank])

        self.poses_dir = ""

    def parse_raw_camera(self, pose_raw):
        pose_flip = camera.pose(R=torch.diag(torch.tensor([1, -1, -1])))
        pose = camera.pose.compose([pose_flip, pose_raw[:3]])
        pose = camera.pose.invert(pose)
        return pose

    def get_all_camera_poses(self, frames):
        pose_raw_all = [torch.tensor(f["transform_matrix"], dtype=torch.float32) for f in frames]
        pose_canon_all = torch.stack([self.parse_raw_camera(p) for p in pose_raw_all], dim=0)
        return pose_canon_all

    @profiler.time_function
    def get_eval_image_metrics_and_images(self, step: int):
        """This function gets your evaluation loss dict. It needs to get the data
        from the DataManager and feed it to the model's forward function

        Args:
            step: current iteration step
        """
        self.eval()
        image_idx, camera_ray_bundle, batch = self.datamanager.next_eval_image(step)
        outputs = self.model.get_outputs_for_camera_ray_bundle(camera_ray_bundle)
        metrics_dict, images_dict = self.model.get_image_metrics_and_images(outputs, batch)
        assert "image_idx" not in metrics_dict
        metrics_dict["image_idx"] = image_idx
        assert "num_rays" not in metrics_dict
        metrics_dict["num_rays"] = len(camera_ray_bundle)
        # self.save_poses(step=step)
        # self.create_pose_gif()
        self.train()
        return metrics_dict, images_dict

    def save_poses(self, step: int):
        # export camera poses
        train_dataset = self.datamanager.train_dataset
        assert isinstance(train_dataset, InputDataset)

        eval_dataset = self.datamanager.eval_dataset
        assert isinstance(eval_dataset, InputDataset)

        train_camera_optimizer = self.datamanager.train_camera_optimizer
        assert isinstance(train_camera_optimizer, CameraOptimizer)

        eval_camera_optimizer = self.datamanager.eval_camera_optimizer
        assert isinstance(eval_camera_optimizer, CameraOptimizer)

        train_opt_frames = collect_camera_poses_for_dataset(train_dataset, train_camera_optimizer)
        train_init_frames = collect_init_camera_poses_for_dataset(train_dataset)
        collect_camera_poses_for_dataset(eval_dataset, eval_camera_optimizer)
        collect_init_camera_poses_for_dataset(eval_dataset)

        train_opt_poses = self.get_all_camera_poses(train_opt_frames["frames"])
        train_init_frames = self.get_all_camera_poses(train_init_frames["frames"])
        # eval_opt_poses = self.get_all_camera_poses(eval_opt_frames["frames"])
        # eval_init_poses = self.get_all_camera_poses(eval_init_frames["frames"])

        opt = EasyDict({"visdom": EasyDict({"cam_depth": 1})})  # adjust this as needed
        fig = plt.figure(figsize=(5, 5))

        if not os.path.exists(self.poses_dir):
            os.makedirs(self.poses_dir)

        plot_save_poses_blender(opt, fig, train_opt_poses, train_init_frames, path=self.poses_dir, ep=step)

        plt.close("all")

    def natural_sort_key(self, s):
        """
        Sort strings containing numbers in a way that '2' comes before '10'.
        """
        import re

        return [int(text) if text.isdigit() else text.lower() for text in re.split("([0-9]+)", s)]

    def create_pose_gif(self, duration=0.1):
        """
        Create a GIF from a folder of PNG images.

        Parameters:
        - input_folder: The folder containing the PNG images.
        - output_gif_name: The name of the output GIF file (including .gif extension).
        - duration: Duration for each frame in the GIF (default is 0.5 seconds).
        """

        # Get list of all PNG files in the folder
        filenames = [f for f in os.listdir(str(self.poses_dir)) if f.endswith(".png")]

        # Sort filenames numerically
        filenames = sorted(filenames, key=self.natural_sort_key)

        # Read each file and append to a list
        images = [imageio.imread(os.path.join(str(self.poses_dir), filename)) for filename in filenames]

        # Create the GIF
        output_gif_name = self.poses_dir + "/poses.gif"
        imageio.mimsave(output_gif_name, images, duration=duration)
