from typing import Dict, Literal

from jaxtyping import Float
from torch import Tensor

from nerfstudio.data.dataparsers.base_dataparser import DataparserOutputs
from nerfstudio.data.datasets.base_dataset import InputDataset


class RawImageDataset(InputDataset):
    def __init__(self, dataparser_outputs: DataparserOutputs, scale_factor: float = 1.0):
        super().__init__(dataparser_outputs, scale_factor)

    def get_image_raw(self, image_idx: int) -> Float[Tensor, "image_height image_width num_channels"]:
        image_filename = self._dataparser_outputs.image_filenames[image_idx]

    def get_data(self, image_idx: int, image_type: Literal["uint8", "float32"] = "float32") -> Dict:
        image = self.get_image_raw(image_idx)
        data = {}
        return data

    def get_metadata(self, image_idx: int) -> Dict:
        return {}
