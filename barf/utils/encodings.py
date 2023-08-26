"""
Encoding functions
"""

import itertools
from abc import abstractmethod
from typing import Literal, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from jaxtyping import Float, Int, Shaped
from torch import Tensor, nn

from nerfstudio.field_components.base_field_component import FieldComponent
from nerfstudio.field_components.encodings import Encoding
from nerfstudio.utils.math import components_from_spherical_harmonics, expected_sin
from nerfstudio.utils.printing import print_tcnn_speed_warning
from nerfstudio.utils.writer import GLOBAL_BUFFER

try:
    import tinycudann as tcnn

    TCNN_EXISTS = True
except ModuleNotFoundError:
    TCNN_EXISTS = False


class BARFEncodingFreq(Encoding):
    """Multi-scale sinusoidal encodings. Support ``integrated positional encodings`` if covariances are provided.
    Each axis is encoded with frequencies ranging from 2^min_freq_exp to 2^max_freq_exp.

    Args:
        in_dim: Input dimension of tensor
        num_frequencies: Number of encoded frequencies per axis
        min_freq_exp: Minimum frequency exponent
        max_freq_exp: Maximum frequency exponent
        include_input: Append the input coordinate to the encoding
    """

    def __init__(
        self,
        in_dim: int,
        num_frequencies: int,
        min_freq_exp: float,
        max_freq_exp: float,
        coarse_to_fine_iters: Tuple[float, float] = (0.1, 0.5),
        include_input: bool = False,
    ) -> None:
        super().__init__(in_dim)

        self.num_frequencies = num_frequencies
        self.min_freq = min_freq_exp
        self.max_freq = max_freq_exp
        self.coarse_to_fine_iters = coarse_to_fine_iters
        self.include_input = include_input

    def get_out_dim(self) -> int:
        if self.in_dim is None:
            raise ValueError("Input dimension has not been set")
        out_dim = self.in_dim * self.num_frequencies * 2
        if self.include_input:
            out_dim += self.in_dim
        return out_dim

    def forward(
        self,
        in_tensor: Shaped[Tensor, "bs input_dim"],
        step: int,
        covs: Optional[Shaped[Tensor, "bs input_dim input_dim"]] = None,
    ) -> Shaped[Tensor, "bs output_dim"]:
        """Calculates NeRF encoding. If covariances are provided the encodings will be integrated as proposed
            in mip-NeRF.

        Args:
            in_tensor: For best performance, the input tensor should be between 0 and 1.
            covs: Covariances of input points.
        Returns:
            Output values will be between -1 and 1
        """
        scaled_in_tensor = 2 * torch.pi * in_tensor  # scale to [0, 2pi]
        freqs = 2 ** torch.linspace(self.min_freq, self.max_freq, self.num_frequencies).to(in_tensor.device)
        # [..., "input_dim", "num_scales"]
        scaled_inputs = scaled_in_tensor[..., None] * freqs
        # [..., "input_dim" * "num_scales"]
        scaled_inputs = scaled_inputs.view(*scaled_inputs.shape[:-2], -1)

        start, end = self.coarse_to_fine_iters
        self.max_iters = GLOBAL_BUFFER.get("max_iter", 0)
        progress = step / self.max_iters
        alpha = (progress - start) / (end - start) * self.num_frequencies
        k = torch.arange(self.num_frequencies, dtype=torch.float32, device=in_tensor.device)
        weights = (1 - (alpha - k).clamp_(min=0, max=1).mul_(torch.pi).cos_()) / 2

        if covs is None:
            encoded_inputs = torch.sin(torch.cat([scaled_inputs, scaled_inputs + torch.pi / 2.0], dim=-1))
            shape = encoded_inputs.shape
            encoded_inputs = (encoded_inputs.view(-1, self.num_frequencies) * weights).view(*shape)
        else:
            input_var = torch.diagonal(covs, dim1=-2, dim2=-1)[..., :, None] * freqs[None, :] ** 2
            input_var = input_var.reshape((*input_var.shape[:-2], -1))

            encoded_inputs = expected_sin(
                torch.cat([scaled_inputs, scaled_inputs + torch.pi / 2.0], dim=-1), torch.cat(2 * [input_var], dim=-1)
            )
            shape = encoded_inputs.shape
            encoded_inputs = (encoded_inputs.view(-1, self.num_frequencies) * weights).view(*shape)

        if self.include_input:
            encoded_inputs = torch.cat([encoded_inputs, in_tensor], dim=-1)

        return encoded_inputs


class _HashGradientScaler(torch.autograd.Function):  # typing: ignore
    """
    Scales the gradients of hash features based on a provided mask
    """

    @staticmethod
    def forward(ctx, value: Float[Tensor, "bs feat_dim"], mask: Float[Tensor, "feat_dim"]):
        ctx.save_for_backward(mask)
        return value, mask

    @staticmethod
    def backward(ctx, output_grad, grad_scaling):
        (mask,) = ctx.saved_tensors
        N = mask.shape[1]
        D = mask.shape[2]
        B = output_grad.shape[0]
        in_shape = output_grad.shape
        output_grad = (output_grad.view(B, N, D) * mask).view(*in_shape)
        return output_grad, grad_scaling


class ScaledHashEncoding(Encoding):
    """Hash encoding that incorporates gradient scaling

    Args:
        num_levels: Number of feature grids.
        min_res: Resolution of smallest feature grid.
        max_res: Resolution of largest feature grid.
        log2_hashmap_size: Size of hash map is 2^log2_hashmap_size.
        features_per_level: Number of features per level.
        hash_init_scale: Value to initialize hash grid.
        implementation: Implementation of hash encoding. Fallback to torch if tcnn not available.
        interpolation: Interpolation override for tcnn hashgrid. Not supported for torch unless linear.
        coarse_to_fine_iters: (start, end) iterations at which gradients of hash grid levels are modulated. Linear interpolation between (start, end) and full activation from end onwards.
    """

    def __init__(
        self,
        num_levels: int = 16,
        min_res: int = 16,
        max_res: int = 1024,
        log2_hashmap_size: int = 19,
        features_per_level: int = 2,
        hash_init_scale: float = 0.001,
        implementation: Literal["tcnn", "torch"] = "tcnn",
        interpolation: Optional[Literal["Nearest", "Linear", "Smoothstep"]] = None,
        coarse_to_fine_iters: Optional[Tuple[int, int]] = None,
    ) -> None:
        super().__init__(in_dim=3)
        self.num_levels = num_levels
        self.features_per_level = features_per_level
        self.log2_hashmap_size = log2_hashmap_size
        self.hash_table_size = 2**log2_hashmap_size
        self.coarse_to_fine_iters = coarse_to_fine_iters
        self.step = 0

        levels = torch.arange(num_levels)
        growth_factor = np.exp((np.log(max_res) - np.log(min_res)) / (num_levels - 1)) if num_levels > 1 else 1
        self.scalings = torch.floor(min_res * growth_factor**levels)

        self.hash_offset = levels * self.hash_table_size

        self.tcnn_encoding = None
        self.hash_table = torch.empty(0)
        if implementation == "tcnn" and not TCNN_EXISTS:
            print_tcnn_speed_warning("HashEncoding")
            implementation = "torch"

        if implementation == "tcnn":
            encoding_config = {
                "otype": "HashGrid",
                "n_levels": self.num_levels,
                "n_features_per_level": self.features_per_level,
                "log2_hashmap_size": self.log2_hashmap_size,
                "base_resolution": min_res,
                "per_level_scale": growth_factor,
            }
            if interpolation is not None:
                encoding_config["interpolation"] = interpolation

            self.tcnn_encoding = tcnn.Encoding(
                n_input_dims=3,
                encoding_config=encoding_config,
            )
        elif implementation == "torch":
            self.hash_table = torch.rand(size=(self.hash_table_size * num_levels, features_per_level)) * 2 - 1
            self.hash_table *= hash_init_scale
            self.hash_table = nn.Parameter(self.hash_table)

        if self.tcnn_encoding is None:
            assert (
                interpolation is None or interpolation == "Linear"
            ), f"interpolation '{interpolation}' is not supported for torch encoding backend"

    def get_out_dim(self) -> int:
        return self.num_levels * self.features_per_level

    def set_step(self, step: int) -> None:
        self.step = step

    def hash_fn(self, in_tensor: Int[Tensor, "*bs num_levels 3"]) -> Shaped[Tensor, "*bs num_levels"]:
        """Returns hash tensor using method described in Instant-NGP

        Args:
            in_tensor: Tensor to be hashed
        """

        # min_val = torch.min(in_tensor)
        # max_val = torch.max(in_tensor)
        # assert min_val >= 0.0
        # assert max_val <= 1.0

        in_tensor = in_tensor * torch.tensor([1, 2654435761, 805459861]).to(in_tensor.device)
        x = torch.bitwise_xor(in_tensor[..., 0], in_tensor[..., 1])
        x = torch.bitwise_xor(x, in_tensor[..., 2])
        x %= self.hash_table_size
        x += self.hash_offset.to(x.device)
        return x

    def pytorch_fwd(self, in_tensor: Float[Tensor, "*bs input_dim"]) -> Float[Tensor, "*bs output_dim"]:
        """Forward pass using pytorch. Significantly slower than TCNN implementation."""

        assert in_tensor.shape[-1] == 3
        in_tensor = in_tensor[..., None, :]  # [..., 1, 3]
        scaled = in_tensor * self.scalings.view(-1, 1).to(in_tensor.device)  # [..., L, 3]
        scaled_c = torch.ceil(scaled).type(torch.int32)
        scaled_f = torch.floor(scaled).type(torch.int32)

        offset = scaled - scaled_f

        hashed_0 = self.hash_fn(scaled_c)  # [..., num_levels]
        hashed_1 = self.hash_fn(torch.cat([scaled_c[..., 0:1], scaled_f[..., 1:2], scaled_c[..., 2:3]], dim=-1))
        hashed_2 = self.hash_fn(torch.cat([scaled_f[..., 0:1], scaled_f[..., 1:2], scaled_c[..., 2:3]], dim=-1))
        hashed_3 = self.hash_fn(torch.cat([scaled_f[..., 0:1], scaled_c[..., 1:2], scaled_c[..., 2:3]], dim=-1))
        hashed_4 = self.hash_fn(torch.cat([scaled_c[..., 0:1], scaled_c[..., 1:2], scaled_f[..., 2:3]], dim=-1))
        hashed_5 = self.hash_fn(torch.cat([scaled_c[..., 0:1], scaled_f[..., 1:2], scaled_f[..., 2:3]], dim=-1))
        hashed_6 = self.hash_fn(scaled_f)
        hashed_7 = self.hash_fn(torch.cat([scaled_f[..., 0:1], scaled_c[..., 1:2], scaled_f[..., 2:3]], dim=-1))

        f_0 = self.hash_table[hashed_0]  # [..., num_levels, features_per_level]
        f_1 = self.hash_table[hashed_1]
        f_2 = self.hash_table[hashed_2]
        f_3 = self.hash_table[hashed_3]
        f_4 = self.hash_table[hashed_4]
        f_5 = self.hash_table[hashed_5]
        f_6 = self.hash_table[hashed_6]
        f_7 = self.hash_table[hashed_7]

        f_03 = f_0 * offset[..., 0:1] + f_3 * (1 - offset[..., 0:1])
        f_12 = f_1 * offset[..., 0:1] + f_2 * (1 - offset[..., 0:1])
        f_56 = f_5 * offset[..., 0:1] + f_6 * (1 - offset[..., 0:1])
        f_47 = f_4 * offset[..., 0:1] + f_7 * (1 - offset[..., 0:1])

        f0312 = f_03 * offset[..., 1:2] + f_12 * (1 - offset[..., 1:2])
        f4756 = f_47 * offset[..., 1:2] + f_56 * (1 - offset[..., 1:2])

        encoded_value = f0312 * offset[..., 2:3] + f4756 * (
            1 - offset[..., 2:3]
        )  # [..., num_levels, features_per_level]

        return torch.flatten(encoded_value, start_dim=-2, end_dim=-1)  # [..., num_levels * features_per_level]

    def scale_grad_by_freq(self, outputs: Float[Tensor, "bs output_dim"]) -> Float[Tensor, "bs output_dim"]:
        """Scale gradients by frequency of hash table entries"""
        if self.coarse_to_fine_iters is None:
            return outputs
        B = outputs.shape[0]
        N = self.num_levels
        D = self.features_per_level
        # formula for getting frequency mask
        start, end = self.coarse_to_fine_iters
        assert (
            start >= 0 and end >= 0
        ), f"start and end iterations for bundle adjustment have to be positive, got start = {start} and end = {end}"
        L = N
        # From https://arxiv.org/pdf/2104.06405.pdf equation 14
        alpha = (self.step - start) / (end - start) * L
        k = torch.arange(L, dtype=outputs.dtype, device=outputs.device)
        mask_vals = (1.0 - (alpha - k).clamp_(min=0, max=1).mul_(np.pi).cos_()) / 2
        mask_vals = mask_vals[None, ..., None].repeat((B, 1, D))
        out, _ = _HashGradientScaler.apply(outputs, mask_vals)  # type: ignore
        return out

    def forward(self, in_tensor: Float[Tensor, "*bs input_dim"]) -> Float[Tensor, "*bs output_dim"]:
        if self.tcnn_encoding is not None:
            out = self.tcnn_encoding(in_tensor)
        else:
            out = self.pytorch_fwd(in_tensor)
        return self.scale_grad_by_freq(out)
