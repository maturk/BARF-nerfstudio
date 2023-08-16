"""
BARF implementations.
"""

from __future__ import annotations

import functools
from contextlib import nullcontext
from dataclasses import dataclass, field
from typing import Dict, List, Literal, Optional, Tuple, Type

import numpy as np
from pathlib import Path
import os
import numpy as np
import cv2
import torch
from torch.nn import Parameter
from torchmetrics import PeakSignalNoiseRatio
from torchmetrics.functional import structural_similarity_index_measure
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

from nerfstudio.configs.config_utils import to_immutable_dict
from nerfstudio.cameras.camera_optimizers import CameraOptimizer, CameraOptimizerConfig
from nerfstudio.cameras.lie_groups import exp_map_SE3, exp_map_SO3xR3
from nerfstudio.cameras.rays import RayBundle, RaySamples
from nerfstudio.engine.callbacks import (
    TrainingCallback,
    TrainingCallbackAttributes,
    TrainingCallbackLocation,
)
from nerfstudio.field_components.field_heads import FieldHeadNames
from nerfstudio.field_components.spatial_distortions import SceneContraction
from nerfstudio.fields.barf_field import BARFFieldNerfacto
from nerfstudio.fields.density_fields import HashMLPDensityField
from nerfstudio.fields.nerfacto_field import NerfactoField
from nerfstudio.model_components.losses import (
    MSELoss,
    distortion_loss,
    interlevel_loss,
    orientation_loss,
    pred_normal_loss,
    scale_gradients_by_distance_squared,
)
from nerfstudio.model_components.ray_samplers import (
    ProposalNetworkSampler,
    UniformSampler,
)
from nerfstudio.model_components.renderers import (
    AccumulationRenderer,
    DepthRenderer,
    NormalsRenderer,
    RGBRenderer,
)
from nerfstudio.model_components.scene_colliders import NearFarCollider
from nerfstudio.model_components.shaders import NormalsShader
from nerfstudio.models.base_model import Model, ModelConfig
from nerfstudio.models.vanilla_nerf import NeRFModel, VanillaModelConfig
from nerfstudio.models.nerfacto import NerfactoModel, NerfactoModelConfig
from nerfstudio.utils import colormaps
from nerfstudio.utils import poses as pose_utils


def save_image(image, image_out, log=True):
    """Save image to file"""
    if not Path(os.path.dirname(image_out)).exists():
        Path(os.path.dirname(image_out)).mkdir()
    if not isinstance(image_out, str):
        image_out = str(image_out)
    if torch.is_tensor(image):
        image = (image.detach().cpu().numpy() * 255).astype(np.uint8)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    if not Path(os.path.dirname(image_out)).exists():
        Path(os.path.dirname(image_out)).mkdir()
    cv2.imwrite(image_out, image)
    if log:
        print(f"Image saved to path {image_out}")


# BARF Configs
@dataclass
class BARFHashModelConfig(NerfactoModelConfig):
    """BARF hashgrid config"""

    _target: Type = field(default_factory=lambda: BARFModelNerfacto)

    use_gradient_scaling: bool = False
    """Use gradient scaler where the gradients are lower for points closer to the camera."""
    use_uncertainty_loss: bool = False
    """Predict uncertainty (variance) for render output"""
    disable_scene_contraction: bool = True
    """Whether to disable scene contraction or not."""
    use_average_appearance_embedding: bool = True
    """Whether to use average appearance embedding or zeros for inference."""
    predict_normals: bool = False
    """Whether to predict normals or not."""
    disable_scene_contraction: bool = False
    """Whether to disable scene contraction or not."""

    # BARF Nerfacto Configs
    camera_optimizer: CameraOptimizerConfig = CameraOptimizerConfig(mode="SO3xR3")
    """Config of the camera optimizer to use"""
    bundle_adjust: bool = True
    """Coarse to fine hash grid frequency optimization"""
    coarse_to_fine_iters: tuple = (0.0, 0.1)
    """Iterations (as a percentage of total iterations) at which c2f hash grid freq optimization starts and ends.
    Linear interpolation between (start, end) and full activation from end onwards."""


@dataclass
class BARFFreqModelConfig(VanillaModelConfig):
    # TODO: Add config for frequency barf

    _target: Type = field(default_factory=lambda: BARFModelFreq)

    # BARF Vanilla configs
    camera_optimizer: CameraOptimizerConfig = CameraOptimizerConfig(mode="SO3xR3")
    """Config of the camera optimizer to use."""
    bundle_adjust: bool = True
    """BARF coarse to fine hash grid optimization."""
    coarse_to_fine_iters: tuple = (0.1, 0.5)
    """Iterations (as a percentage of total iterations) at which coarse_to_fine hash grid optimization starts and ends.
    Linear interpolation between (start, end) and full hash grid activation from end onwards."""


# BARF Models
class BARFModelNerfacto(NerfactoModel):
    """BARF implementation using iNGP hash grid encoding"""

    config: BARFHashModelConfig

    def populate_modules(self):
        super().populate_modules()

        if self.config.disable_scene_contraction:
            scene_contraction = None
        else:
            scene_contraction = SceneContraction(order=float("inf"))

        # BARF Field
        self.field = BARFFieldNerfacto(
            self.scene_box.aabb,
            hidden_dim=self.config.hidden_dim,
            num_levels=self.config.num_levels,
            max_res=self.config.max_res,
            log2_hashmap_size=self.config.log2_hashmap_size,
            hidden_dim_color=self.config.hidden_dim_color,
            hidden_dim_transient=self.config.hidden_dim_transient,
            spatial_distortion=scene_contraction,
            num_images=self.num_train_data,
            use_pred_normals=self.config.predict_normals,
            use_average_appearance_embedding=self.config.use_average_appearance_embedding,
            appearance_embedding_dim=self.config.appearance_embed_dim,
            implementation=self.config.implementation,
            bundle_adjust=self.config.bundle_adjust,
            coarse_to_fine_iters=self.config.coarse_to_fine_iters,
        )

        self.camera_optimizer: CameraOptimizer = self.config.camera_optimizer.setup(
            num_cameras=self.num_train_data, device="cpu"
        )

    def get_param_groups(self) -> Dict[str, List[Parameter]]:
        param_groups = {}
        param_groups["proposal_networks"] = list(self.proposal_networks.parameters())
        param_groups["fields"] = list(self.field.parameters())
        camera_opt_params = list(self.camera_optimizer.parameters())
        if self.config.camera_optimizer.mode != "off":
            assert len(camera_opt_params) > 0
            param_groups["camera_opt"] = camera_opt_params
        else:
            assert len(camera_opt_params) == 0
        return param_groups

    def get_outputs(self, ray_bundle: RayBundle):
        if self.training:
            self.camera_optimizer.apply_to_raybundle(ray_bundle)
        ray_samples: RaySamples
        ray_samples, weights_list, ray_samples_list = self.proposal_sampler(ray_bundle, density_fns=self.density_fns)
        field_outputs = self.field.forward(ray_samples, compute_normals=self.config.predict_normals)
        if self.config.use_gradient_scaling:
            field_outputs = scale_gradients_by_distance_squared(field_outputs, ray_samples)

        weights = ray_samples.get_weights(field_outputs[FieldHeadNames.DENSITY])
        weights_list.append(weights)
        ray_samples_list.append(ray_samples)

        rgb = self.renderer_rgb(rgb=field_outputs[FieldHeadNames.RGB], weights=weights)
        depth = self.renderer_depth(weights=weights, ray_samples=ray_samples)
        accumulation = self.renderer_accumulation(weights=weights)

        outputs = {
            "rgb": rgb,
            "accumulation": accumulation,
            "depth": depth,
        }

        if self.config.predict_normals:
            normals = self.renderer_normals(normals=field_outputs[FieldHeadNames.NORMALS], weights=weights)
            pred_normals = self.renderer_normals(field_outputs[FieldHeadNames.PRED_NORMALS], weights=weights)
            outputs["normals"] = self.normals_shader(normals)
            outputs["pred_normals"] = self.normals_shader(pred_normals)
        # These use a lot of GPU memory, so we avoid storing them for eval.
        if self.training:
            outputs["weights_list"] = weights_list
            outputs["ray_samples_list"] = ray_samples_list

        if self.training and self.config.predict_normals:
            outputs["rendered_orientation_loss"] = orientation_loss(
                weights.detach(), field_outputs[FieldHeadNames.NORMALS], ray_bundle.directions
            )

            outputs["rendered_pred_normal_loss"] = pred_normal_loss(
                weights.detach(),
                field_outputs[FieldHeadNames.NORMALS].detach(),
                field_outputs[FieldHeadNames.PRED_NORMALS],
            )

        for i in range(self.config.num_proposal_iterations):
            outputs[f"prop_depth_{i}"] = self.renderer_depth(weights=weights_list[i], ray_samples=ray_samples_list[i])
        return outputs

    def get_training_callbacks(
        self, training_callback_attributes: TrainingCallbackAttributes
    ) -> List[TrainingCallback]:
        callbacks = []
        if self.config.use_proposal_weight_anneal:
            # anneal the weights of the proposal network before doing PDF sampling
            N = self.config.proposal_weights_anneal_max_num_iters

            def set_anneal(step):
                # https://arxiv.org/pdf/2111.12077.pdf eq. 18
                train_frac = np.clip(step / N, 0, 1)

                def bias(x, b):
                    return b * x / ((b - 1) * x + 1)

                anneal = bias(train_frac, self.config.proposal_weights_anneal_slope)
                self.proposal_sampler.set_anneal(anneal)

            callbacks.append(
                TrainingCallback(
                    where_to_run=[TrainingCallbackLocation.BEFORE_TRAIN_ITERATION],
                    update_every_num_iters=1,
                    func=set_anneal,
                )
            )
            callbacks.append(
                TrainingCallback(
                    where_to_run=[TrainingCallbackLocation.AFTER_TRAIN_ITERATION],
                    update_every_num_iters=1,
                    func=self.proposal_sampler.step_cb,
                )
            )

            def get_init_camera_poses(
                self,
                training_callback_attributes: TrainingCallbackAttributes,
                step: int,
            ):  # pylint: disable=unused-argument
                self.cameras = training_callback_attributes.pipeline.datamanager.train_dataparser_outputs.cameras
                # print(self.cameras.camera_to_worlds[:3])
                # self.cameras.camera_to_worlds = self.init_train_camera_poses()

            callbacks.append(
                TrainingCallback(
                    where_to_run=[
                        TrainingCallbackLocation.BEFORE_TRAIN,
                    ],
                    func=get_init_camera_poses,
                    update_every_num_iters=100,
                    args=[self, training_callback_attributes],
                )
            )

            callbacks.append(
                TrainingCallback(
                    where_to_run=[TrainingCallbackLocation.AFTER_TRAIN_ITERATION],
                    update_every_num_iters=1,
                    func=self.field.step_cb,
                )
            )

            # Delete this maybe
            def save_images(
                self,
                training_callback_attributes: TrainingCallbackAttributes,
                step: int,
            ):  # pylint: disable=unused-argument
                self.eval()
                (
                    camera_ray_bundle,
                    batch,
                ) = training_callback_attributes.pipeline.datamanager.at_train_end()
                outputs, ref = self.at_train_end(camera_ray_bundle=camera_ray_bundle, batch=batch)
                pred = outputs["rgb"]
                self.save_path = f"/home/maturk/data/scratch/nerfacto/pred_{step+ 1:05d}.png"
                self.ref_save_path = f"/home/maturk/data/scratch/ref_rgb.png"
                if self.step % 100 == 0:
                    # save_image(pred, self.save_path, log=True)
                    # save_image(ref, self.ref_save_path, log=False)
                    pass

            callbacks.append(
                TrainingCallback(
                    where_to_run=[
                        TrainingCallbackLocation.AFTER_TRAIN,
                        TrainingCallbackLocation.AFTER_TRAIN_ITERATION,
                    ],
                    func=save_images,
                    update_every_num_iters=100,
                    args=[self, training_callback_attributes],
                )
            )

    # DELETE
    def at_train_end(self, **kwargs):
        camera_ray_bundle = kwargs["camera_ray_bundle"]
        batch = kwargs["batch"]
        outputs = self.get_outputs_for_camera_ray_bundle(camera_ray_bundle)
        ref = batch["image"].to(self.device)
        return outputs, ref


class BARFModelFreq(NeRFModel):
    """TODO: Vanilla BARF implementation based on the original paper."""
    config: BARFFreqModelConfig

    def populate_modules(self):
        super().populate_modules()

    def get_outputs(self, ray_bundle: RayBundle):
        #TODO: