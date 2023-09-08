""" Helper functions for visualizing outputs """

from dataclasses import dataclass
from typing import Literal, Optional

import matplotlib
import torch
from matplotlib import cm

from torchtyping import TensorType
from torch import Tensor
from jaxtyping import Bool, Float

def apply_uncertainty_colormap(image: TensorType["bs":..., 1], cmap="viridis") -> TensorType["bs":..., "rgb":3]:
    colormap = cm.get_cmap(cmap)
    colormap = torch.tensor(colormap.colors).to(image.device)  # type: ignore
    image = torch.nan_to_num(image, 0)
    image_long = (image * 255).long()
    image_long_min = torch.min(image_long)
    image_long_max = torch.max(image_long)
    if (image_long_min < 0):
        print(f"the min value is {image_long_min}")
    if(image_long_max > 255):
        print(f"the max value is {image_long_max}")

    image_long = (image_long / image_long_max * 255).long()
    
    return colormap[image_long[..., 0]]
