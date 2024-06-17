from typing import Dict, Literal

import numpy as np
import rawpy
import torch
from jaxtyping import Float
from torch import Tensor

from nerfstudio.data.dataparsers.base_dataparser import DataparserOutputs
from nerfstudio.data.datasets.base_dataset import InputDataset
from nerfstudio.data.utils.data_utils import get_image_mask_tensor_from_path
from nerfstudio.utils import raw_utils


class RawImageDataset(InputDataset):
    def __init__(self, dataparser_outputs: DataparserOutputs, scale_factor: float = 1.0):
        super().__init__(dataparser_outputs, 1.0)
        self.scale_factor = scale_factor

    def get_image_raw(self, image_idx: int) -> Float[Tensor, "image_height image_width num_channels"]:
        image_filename = self._dataparser_outputs.image_filenames[image_idx]

        with open(image_filename, "rb") as f:
            raw = rawpy.imread(f).raw_image

        def processing_fn(raw):
            raw = raw.astype(np.float32)
            black_level = self._dataparser_outputs.cameras.metadata["black_level"][image_idx].numpy()  # type: ignore
            white_level = self._dataparser_outputs.cameras.metadata["white_level"][image_idx].numpy()  # type: ignore
            im = (raw - black_level) / (white_level - black_level)
            # Demosaic Bayer images (preserves the measured RGGB values).
            im = raw_utils.bilinear_demosaic(im)
            if self.scale_factor != 1.0:
                downscale_factor = int(1.0 / self.scale_factor)
                im = raw_utils.downsample(im, downscale_factor)
            im = torch.from_numpy(im).to(torch.float32)
            return im

        im = processing_fn(raw)
        return im

    def get_data(self, image_idx: int, image_type: Literal["uint8", "float32"] = "float32") -> Dict:
        image = self.get_image_raw(image_idx)
        data = {"image_idx": image_idx, "image": image}

        if self._dataparser_outputs.mask_filenames is not None:
            mask_filepath = self._dataparser_outputs.mask_filenames[image_idx]
            data["mask"] = get_image_mask_tensor_from_path(filepath=mask_filepath, scale_factor=self.scale_factor)
            assert (
                data["mask"].shape[:2] == data["image"].shape[:2]
            ), f"Mask and image have different shapes. Got {data['mask'].shape[:2]} and {data['image'].shape[:2]}"
        if self.mask_color:
            data["image"] = torch.where(
                data["mask"] == 1.0, data["image"], torch.ones_like(data["image"]) * torch.tensor(self.mask_color)
            )
        metadata = self.get_metadata(data)
        data.update(metadata)
        return data
