# Copyright 2022 The Nerfstudio Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Implementation of vanilla nerf.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Type

import numpy as np
import torch
from torch.nn import Parameter
from torchmetrics import PeakSignalNoiseRatio
from torchmetrics.functional import structural_similarity_index_measure
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

from nerfstudio.cameras.camera_optimizers import CameraOptimizer, CameraOptimizerConfig
from nerfstudio.cameras.rays import RayBundle, RaySamples
from nerfstudio.engine.callbacks import (
    TrainingCallback,
    TrainingCallbackAttributes,
    TrainingCallbackLocation,
)
from nerfstudio.field_components.field_heads import FieldHeadNames
from nerfstudio.field_components.spatial_distortions import SceneContraction
from nerfstudio.model_components.losses import (
    MSELoss,
    scale_gradients_by_distance_squared,
)
from nerfstudio.model_components.ray_samplers import PDFSampler, UniformSampler
from nerfstudio.model_components.renderers import (
    AccumulationRenderer,
    DepthRenderer,
    RGBRenderer,
)
from nerfstudio.models.base_model import Model, ModelConfig
from nerfstudio.models.nerfacto import NerfactoModel, NerfactoModelConfig
from nerfstudio.utils import colormaps, colors, misc
from nerfstudio.utils.writer import GLOBAL_BUFFER

from barf.barf_field import BARFFieldFreq, BARFHashField, BARFGradientHashField
from barf.utils.encodings import BARFEncodingFreq, ScaledHashEncoding

# BARF Configs

@dataclass
class BARFFreqModelConfig(ModelConfig):
    """BARF Model Config"""

    _target: Type = field(default_factory=lambda: BARFFreqModel)
    num_coarse_samples: int = 128
    """Number of samples in coarse field evaluation"""
    num_importance_samples: int = 128
    """Number of samples in fine field evaluation"""
    fine_field_enabled: bool = False
    """Whether or not to use a fine network"""
    coarse_to_fine_iters: tuple = (0.1, 0.5)
    """Iterations (as a percentage of total iterations) at which c2f freq optimization starts and ends.
    Linear interpolation between (start, end) and full activation from end onwards."""
    camera_optimizer: CameraOptimizerConfig = CameraOptimizerConfig(mode="SO3xR3")
    """Config of the camera optimizer to use"""
    freeze_fields: Optional[List[float]]  = None
    """ List of windows (as a percentage of total iterations) where the fields will not be trained. """
    freeze_cam: Optional[List[float]]  = None
    """ List of windows (as a percentage of total iterations) where the camera optimizer will not be trained. """

@dataclass
class BARFHashModelConfig(NerfactoModelConfig):
    """BARF hashgrid config"""

    _target: Type = field(default_factory=lambda: BARFHashModel)

    use_average_appearance_embedding: bool = False
    """Whether to use average appearance embedding or zeros for inference."""
    camera_optimizer: CameraOptimizerConfig = CameraOptimizerConfig(mode="SO3xR3")
    """Config of the camera optimizer to use"""
    bundle_adjust: bool = True
    """Coarse to fine hash grid frequency optimization"""
    coarse_to_fine_iters: tuple = (0.0, 0.1)
    """Iterations (as a percentage of total iterations) at which c2f hash grid freq optimization starts and ends.
    Linear interpolation between (start, end) and full activation from end onwards."""


@dataclass
class BARFGradientHashModelConfig(NerfactoModelConfig):
    """BARF gradient modulated hashgrid config"""

    _target: Type = field(default_factory=lambda: BARFGradientHashModel)

    coarse_to_fine_iters: Optional[Tuple[int, int]] = (0, 1000)
    """(start, end) iterations at which gradients of hash grid levels are modulated. Linear interpolation between (start, end) and full activation from end onwards."""
    camera_optimizer: CameraOptimizerConfig = CameraOptimizerConfig(mode="SO3xR3")
    """Config of the camera optimizer to use"""


# BARF models
class BARFFreqModel(Model):
    """Bundle Adjusting Radiance Field adaptation

    Args:
        config: Basic BARF configuration to instantiate model
    """

    config: BARFFreqModelConfig

    def __init__(
        self,
        config: BARFFreqModelConfig,
        **kwargs,
    ) -> None:
        self.field_coarse = None
        self.field_fine = None

        super().__init__(
            config=config,
            **kwargs,
        )

    def populate_modules(self):
        """Set the fields and modules"""
        super().populate_modules()

        # fields
        position_encoding = BARFEncodingFreq(
            in_dim=3,
            num_frequencies=10,
            min_freq_exp=0.0,
            max_freq_exp=9.0,
            coarse_to_fine_iters=self.config.coarse_to_fine_iters,
            include_input=True,
        )
        direction_encoding = BARFEncodingFreq(
            in_dim=3,
            num_frequencies=4,
            min_freq_exp=0.0,
            max_freq_exp=3.0,
            coarse_to_fine_iters=self.config.coarse_to_fine_iters,
            include_input=True,
        )

        self.field_coarse = BARFFieldFreq(
            position_encoding=position_encoding,
            direction_encoding=direction_encoding,
        )
        if self.config.fine_field_enabled:
            self.field_fine = BARFFieldFreq(
                position_encoding=position_encoding,
                direction_encoding=direction_encoding,
            )

        # samplers
        self.sampler_uniform = UniformSampler(num_samples=self.config.num_coarse_samples)
        if self.config.fine_field_enabled:
            self.sampler_pdf = PDFSampler(num_samples=self.config.num_importance_samples)

        # renderers
        self.renderer_rgb = RGBRenderer(background_color=colors.WHITE)
        self.renderer_accumulation = AccumulationRenderer()
        self.renderer_depth = DepthRenderer()

        # losses
        self.rgb_loss = MSELoss()

        # metrics
        self.psnr = PeakSignalNoiseRatio(data_range=1.0)
        self.ssim = structural_similarity_index_measure
        self.lpips = LearnedPerceptualImagePatchSimilarity(normalize=True)
        self._step = 0

        # camera optimizer
        self.camera_optimizer: CameraOptimizer = self.config.camera_optimizer.setup(
            num_cameras=self.num_train_data, device="cuda"
        )

        assert self.config.freeze_fields is None or len(self.config.freeze_fields) % 2 == 0
        self.freeze_fields = []
        if self.config.freeze_fields is not None:
            for i in range(0, len(self.config.freeze_fields), 2):
                assert freeze_fields[i] < freeze_fields[i + 1]
                self.freeze_fields.append((freeze_fields[i], freeze_fields[i + 1]))

        assert self.config.freeze_cam is None or len(self.config.freeze_cam) % 2 == 0
        self.freeze_cam = []
        if self.config.freeze_cam is not None:
            for i in range(0, len(self.config.freeze_cam), 2):
                assert self.config.freeze_cam[i] < self.config.freeze_cam[i + 1]
                self.freeze_cam.append((self.config.freeze_cam[i], self.config.freeze_cam[i + 1]))

    def get_param_groups(self) -> Dict[str, List[Parameter]]:
        param_groups = {}
        if self.field_coarse is None or (self.config.fine_field_enabled and self.field_fine is None):
            raise ValueError("populate_fields() must be called before get_param_groups")
        if self.config.fine_field_enabled:
            param_groups["fields"] = list(self.field_coarse.parameters()) + list(self.field_fine.parameters())
        else:
            param_groups["fields"] = list(self.field_coarse.parameters())

        camera_opt_params = list(self.camera_optimizer.parameters())
        if self.config.camera_optimizer.mode != "off":
            camera_opt_params = list(self.camera_optimizer.parameters())
            assert len(camera_opt_params) > 0
            param_groups["camera_opt"] = camera_opt_params
        else:
            assert len(camera_opt_params) == 0
        return param_groups

    def get_outputs(self, ray_bundle: RayBundle) -> Dict[str, torch.Tensor]:
        progress: float = self._step / GLOBAL_BUFFER.get("max_iter", 0)
        for freeze_window in self.freeze_fields:
            if progress >= freeze_window[0] and progress <= freeze_window[1]:
                for param in self.field_coarse.parameters():
                    param.requires_grad = False
                if self.config.fine_field_enabled:
                    for param in self.field_fine.parameters():
                        param.requires_grad = False
                break
            else:
                for param in self.field_coarse.parameters():
                    param.requires_grad = True
                if self.config.fine_field_enabled:
                    for param in self.field_fine.parameters():
                        param.requires_grad = True

        for freeze_window in self.freeze_cam:
            if progress >= freeze_window[0] and progress <= freeze_window[1]:
                for param in self.camera_optimizer.parameters():
                    param.requires_grad = False
                break
            else:
                for param in self.camera_optimizer.parameters():
                    param.requires_grad = True

        if self.training and hasattr(self.camera_optimizer, "apply_to_raybundle"):
            self.camera_optimizer.apply_to_raybundle(ray_bundle)
        outputs = {}
        if self.field_coarse is None or (self.config.fine_field_enabled and self.field_fine is None):
            raise ValueError("populate_fields() must be called before get_outputs")

        # uniform sampling
        ray_samples_uniform = self.sampler_uniform(ray_bundle)

        # coarse field:
        field_outputs_coarse = self.field_coarse.forward(ray_samples_uniform, self._step)
        weights_coarse = ray_samples_uniform.get_weights(field_outputs_coarse[FieldHeadNames.DENSITY])
        rgb_coarse = self.renderer_rgb(
            rgb=field_outputs_coarse[FieldHeadNames.RGB],
            weights=weights_coarse,
        )
        density_coarse = field_outputs_coarse[FieldHeadNames.DENSITY]
        accumulation_coarse = self.renderer_accumulation(weights_coarse)
        depth_coarse = self.renderer_depth(weights_coarse, ray_samples_uniform)

        if self.config.fine_field_enabled:
            # pdf sampling
            ray_samples_pdf = self.sampler_pdf(ray_bundle, ray_samples_uniform, weights_coarse)

            # fine field:
            field_outputs_fine = self.field_fine.forward(ray_samples_pdf, self._step)
            weights_fine = ray_samples_pdf.get_weights(field_outputs_fine[FieldHeadNames.DENSITY])
            rgb_fine = self.renderer_rgb(
                rgb=field_outputs_fine[FieldHeadNames.RGB],
                weights=weights_fine,
            )
            density_fine = field_outputs_fine[FieldHeadNames.DENSITY]
            accumulation_fine = self.renderer_accumulation(weights_fine)
            depth_fine = self.renderer_depth(weights_fine, ray_samples_pdf)

        outputs = {
            "rgb_coarse": rgb_coarse,
            "accumulation_coarse": accumulation_coarse,
            "depth_coarse": depth_coarse,
            "density_coarse": density_coarse,
        }
        if self.config.fine_field_enabled:
            outputs["rgb_fine"] = rgb_fine
            outputs["accumulation_fine"] = accumulation_fine
            outputs["depth_fine"] = depth_fine
            outputs["density_fine"] = density_fine

        return outputs

    def step_cb(self, step):
        """Callback to register a training step has passed. This is used to keep track of the sampling schedule"""
        self._step = step
        # self._steps_since_update += 1

    def get_training_callbacks(
        self, training_callback_attributes: TrainingCallbackAttributes
    ) -> List[TrainingCallback]:
        callbacks = [
            TrainingCallback(
                where_to_run=[TrainingCallbackLocation.AFTER_TRAIN_ITERATION],
                update_every_num_iters=1,
                func=self.step_cb,
            )
        ]

        return callbacks

    def get_loss_dict(self, outputs, batch, metrics_dict=None) -> Dict[str, torch.Tensor]:
        # Scaling metrics by coefficients to create the losses.
        device = outputs["rgb_coarse"].device
        image = batch["image"].to(device)
        rgb_loss_coarse = self.rgb_loss(image, outputs["rgb_coarse"])

        if self.config.fine_field_enabled:
            rgb_loss_fine = self.rgb_loss(image, outputs["rgb_fine"])
            loss_dict = {"rgb_loss_coarse": rgb_loss_coarse, "rgb_loss_fine": rgb_loss_fine}
        else:
            loss_dict = {"rgb_loss": rgb_loss_coarse}

        loss_dict = misc.scale_dict(loss_dict, self.config.loss_coefficients)
        return loss_dict

    def get_image_metrics_and_images(
        self, outputs: Dict[str, torch.Tensor], batch: Dict[str, torch.Tensor]
    ) -> Tuple[Dict[str, float], Dict[str, torch.Tensor]]:
        image = batch["image"].to(outputs["rgb_coarse"].device)
        rgb_coarse = outputs["rgb_coarse"]
        acc_coarse = colormaps.apply_colormap(outputs["accumulation_coarse"])
        depth_coarse = colormaps.apply_depth_colormap(
            outputs["depth_coarse"],
            accumulation=outputs["accumulation_coarse"],
            near_plane=self.config.collider_params["near_plane"],
            far_plane=self.config.collider_params["far_plane"],
        )

        if self.config.fine_field_enabled:
            rgb_fine = outputs["rgb_fine"]
            acc_fine = colormaps.apply_colormap(outputs["accumulation_fine"])
            depth_fine = colormaps.apply_depth_colormap(
                outputs["depth_fine"],
                accumulation=outputs["accumulation_fine"],
                near_plane=self.config.collider_params["near_plane"],
                far_plane=self.config.collider_params["far_plane"],
            )
            combined_rgb = torch.cat([image, rgb_coarse, rgb_fine], dim=1)
            combined_acc = torch.cat([acc_coarse, acc_fine], dim=1)
            combined_depth = torch.cat([depth_coarse, depth_fine], dim=1)

        else:
            combined_rgb = torch.cat([image, rgb_coarse], dim=1)
            combined_acc = torch.cat([acc_coarse], dim=1)
            combined_depth = torch.cat([depth_coarse], dim=1)

        # Switch images from [H, W, C] to [1, C, H, W] for metrics computations
        image = torch.moveaxis(image, -1, 0)[None, ...]
        rgb_coarse = torch.moveaxis(rgb_coarse, -1, 0)[None, ...]

        coarse_psnr = self.psnr(image, rgb_coarse)
        coarse_ssim = self.ssim(image, rgb_coarse)
        coarse_lpips = self.lpips(image, rgb_coarse)

        if self.config.fine_field_enabled:
            rgb_fine = torch.moveaxis(rgb_fine, -1, 0)[None, ...]
            fine_psnr = self.psnr(image, rgb_fine)
            fine_ssim = self.ssim(image, rgb_fine)
            fine_lpips = self.lpips(image, rgb_fine)
            metrics_dict = {
                "coarse_psnr": float(coarse_psnr),
                "coarse_ssim": float(coarse_ssim),
                "coarse_lpips": float(coarse_lpips),
                "fine_psnr": float(fine_psnr),
                "fine_ssim": float(fine_ssim),
                "fine_lpips": float(fine_lpips),
            }
        else:
            metrics_dict = {
                "psnr": float(coarse_psnr),
                "ssim": float(coarse_ssim),
                "lpips": float(coarse_lpips),
            }
        images_dict = {"img": combined_rgb, "accumulation": combined_acc, "depth": combined_depth}
        return metrics_dict, images_dict


class BARFHashModel(NerfactoModel):
    """BARF implementation using hash grid encoding that masks hash grid levels."""

    config: BARFHashModelConfig

    def populate_modules(self):
        super().populate_modules()

        if self.config.disable_scene_contraction:
            scene_contraction = None
        else:
            scene_contraction = SceneContraction(order=float("inf"))

        # BARF Field
        self.field = BARFHashField(
            self.scene_box.aabb,
            hidden_dim=self.config.hidden_dim,
            num_levels=self.config.num_levels,
            max_res=self.config.max_res,
            base_res=self.config.base_res,
            features_per_level=self.config.features_per_level,
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
            num_cameras=self.num_train_data, device="cuda"
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
        if self.training and hasattr(self.camera_optimizer, "apply_to_raybundle"):
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

            callbacks.append(
                TrainingCallback(
                    where_to_run=[TrainingCallbackLocation.AFTER_TRAIN_ITERATION],
                    update_every_num_iters=1,
                    func=self.field.step_cb,
                )
            )
        return callbacks


class BARFGradientHashModel(NerfactoModel):
    """BARF implementation using hash grid that scales gradients at different level resolutions."""

    config: BARFGradientHashModelConfig

    def populate_modules(self):
        super().populate_modules()

        if self.config.disable_scene_contraction:
            scene_contraction = None
        else:
            scene_contraction = SceneContraction(order=float("inf"))

        # BARF Field
        self.field = BARFGradientHashField(
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
            coarse_to_fine_iters=self.config.coarse_to_fine_iters,
        )
        # camera optimizer
        self.camera_optimizer: CameraOptimizer = self.config.camera_optimizer.setup(
            num_cameras=self.num_train_data, device="cuda"
        )

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

            callbacks.append(
                TrainingCallback(
                    where_to_run=[TrainingCallbackLocation.AFTER_TRAIN_ITERATION],
                    update_every_num_iters=1,
                    func=self.field.step_cb,
                )
            )
        return callbacks
