import imageio
import matplotlib.pyplot as plt
import os
import torch
from typing import Any, Optional, List, Dict
import re

from nerfstudio.cameras.camera_optimizers import CameraOptimizer
from nerfstudio.data.datasets.base_dataset import InputDataset
import nerfstudio.utils.poses as pose_utils

from barf.visualizer.util_vis import plot_save_poses_blender, camera


def collect_init_camera_poses_for_dataset(dataset: Optional[InputDataset]) -> List[Dict[str, Any]]:
    if dataset is None:
        return []

    cameras = dataset.cameras
    image_filenames = dataset.image_filenames

    frames: List[Dict[str, Any]] = []

    # new cameras are in cameras, whereas image paths are stored in a private member of the dataset
    for idx in range(len(cameras)):
        image_filename = image_filenames[idx]
        # Create a tensor with the camera index.
        transform = cameras.camera_to_worlds[idx]
        transform = torch.cat(
            [transform, torch.tensor([0, 0, 0, 1], dtype=torch.float32, device=transform.device).reshape(1, 4)], dim=0
        ).tolist()
        frames.append(
            {
                "file_path": str(image_filename),
                "transform_matrix": transform,
            }
        )
    camera_angle_x = 2 * torch.atan(cameras.width / (cameras.fx * 2))[0].item()
    transforms = {"camera_angle_x": camera_angle_x, "frames": frames}
    return transforms


def collect_camera_poses_for_dataset(
    dataset: Optional[InputDataset], optimizer: Optional[CameraOptimizer]
) -> List[Dict[str, Any]]:
    """Collects rescaled, translated and optimised camera poses for a dataset.

    Args:
        dataset: Dataset to collect camera poses for.

    Returns:
        List of dicts containing camera poses.
    """

    if dataset is None:
        return []

    cameras = dataset.cameras
    image_filenames = dataset.image_filenames

    frames: List[Dict[str, Any]] = []

    # new cameras are in cameras, whereas image paths are stored in a private member of the dataset
    for idx in range(len(cameras)):
        image_filename = image_filenames[idx]
        # Create a tensor with the camera index.
        transform = cameras.camera_to_worlds[idx]
        transform = torch.cat(
            [transform, torch.tensor([0, 0, 0, 1], dtype=torch.float32, device=transform.device).reshape(1, 4)]
        )
        if optimizer is not None:
            adjustment = optimizer(torch.tensor([idx], dtype=torch.long)).to(transform.device)
            adjustment = adjustment.squeeze()
            adjustment = torch.cat(
                [adjustment, torch.tensor([0, 0, 0, 1], dtype=torch.float32, device=transform.device).reshape(1, 4)]
            )

            # transform = torch.matmul(transform, adjustment).tolist()
            transform = pose_utils.multiply(transform, adjustment).tolist()
        frames.append(
            {
                "file_path": str(image_filename),
                "transform_matrix": transform,
            }
        )
    camera_angle_x = 2 * torch.atan(cameras.width / (cameras.fx * 2))[0].item()
    transforms = {"camera_angle_x": camera_angle_x, "frames": frames}
    return transforms


def parse_raw_camera(pose_raw):
    pose_flip = camera.pose(R=torch.diag(torch.tensor([1, -1, -1])))
    pose = camera.pose.compose([pose_flip, pose_raw[:3]])
    pose = camera.pose.invert(pose)
    return pose


def get_all_camera_poses(frames):
    pose_raw_all = [torch.tensor(f["transform_matrix"], dtype=torch.float32) for f in frames]
    pose_canon_all = torch.stack([parse_raw_camera(p) for p in pose_raw_all], dim=0)
    return pose_canon_all


def save_poses(step: int, vis_config, train_dataset: InputDataset, train_camera_optimizer: CameraOptimizer):
    # export camera poses
    # train_dataset = self.datamanager.train_dataset
    # assert isinstance(train_dataset, InputDataset)

    # eval_dataset = self.datamanager.eval_dataset
    # assert isinstance(eval_dataset, InputDataset)

    # train_camera_optimizer = self.model.camera_optimizer
    # assert isinstance(train_camera_optimizer, CameraOptimizer)

    # eval_camera_optimizer = self.datamanager.eval_camera_optimizer
    # assert isinstance(eval_camera_optimizer, CameraOptimizer)

    train_opt_frames = collect_camera_poses_for_dataset(train_dataset, train_camera_optimizer)
    train_init_frames = collect_init_camera_poses_for_dataset(train_dataset)
    # collect_camera_poses_for_dataset(eval_dataset, eval_camera_optimizer)
    # collect_init_camera_poses_for_dataset(eval_dataset)

    train_opt_poses = get_all_camera_poses(train_opt_frames["frames"])
    train_init_frames = get_all_camera_poses(train_init_frames["frames"])
    # eval_opt_poses = self.get_all_camera_poses(eval_opt_frames["frames"])
    # eval_init_poses = self.get_all_camera_poses(eval_init_frames["frames"])

    fig = plt.figure(figsize=(5, 5))
    poses_dir = vis_config.poses_dir

    if not os.path.exists(poses_dir):
        os.makedirs(poses_dir)

    plot_save_poses_blender(vis_config, fig, train_opt_poses, train_init_frames, path=poses_dir, ep=step)

    plt.close("all")


def create_pose_gif(poses_dir: str, duration=0.1):
    """
    Create a GIF from a folder of PNG images.

    Parameters:
    - input_folder: The folder containing the PNG images.
    - output_gif_name: The name of the output GIF file (including .gif extension).
    - duration: Duration for each frame in the GIF (default is 0.5 seconds).
    """

    # Get list of all PNG files in the folder
    filenames = [f for f in os.listdir(str(poses_dir)) if f.endswith(".png")]

    # Sort strings containing numbers in a way that '2' comes before '10'.
    def natural_sort_key(s):
        return [(int(text) if text.isdigit() else text.lower()) for text in re.split("([0-9]+)", s)]

    filenames = sorted(filenames, key=natural_sort_key)

    # Read each file and append to a list
    images = [imageio.imread(os.path.join(str(poses_dir), filename)) for filename in filenames]

    # Create the GIF
    output_gif_name = poses_dir + "/poses.gif"
    imageio.mimsave(output_gif_name, images, duration=duration)
