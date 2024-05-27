"""
Dr Song's look up table for light estimation.
"""

from typing import Dict, Optional, Tuple, Type

import torch
from jaxtyping import Float
from torch import Tensor, nn

import nerfstudio.utils.poses as pose_utils
from nerfstudio.cameras.rays import RayBundle
from nerfstudio.field_components.encodings import Encoding, Identity
from nerfstudio.field_components.field_heads import FieldHead, FieldHeadNames, RGBAffineFieldHead
from nerfstudio.field_components.mlp import MLP
from nerfstudio.field_components.spatial_distortions import SpatialDistortion


class LUT3DField(nn.Module):
    def __init__(
        self,
        positional_encoding: Encoding = Identity(in_dim=3),
        base_mlp_num_layers: int = 4,
        base_mlp_layer_width: int = 128,
        head_mlp_num_lyaers: int = 2,
        head_mlp_layer_width: int = 64,
        skip_connections: Tuple[int] = (2,),
        field_heads: Optional[Tuple[Type[FieldHead]]] = (RGBAffineFieldHead,),
        spatial_distortion: Optional[SpatialDistortion] = None,
    ) -> None:
        super().__init__()
        self.position_encoding = positional_encoding
        self.spatial_distortion = spatial_distortion

        # MLP base
        self.mlp_base = MLP(
            in_dim=self.position_encoding.get_out_dim(),
            num_layers=base_mlp_num_layers,
            layer_width=base_mlp_layer_width,
            skip_connections=skip_connections,
            out_activation=nn.ReLU(),
        )
        if field_heads:
            self.mlp_head = MLP(
                in_dim=self.mlp_base.get_out_dim(),
                num_layers=head_mlp_num_lyaers,
                layer_width=head_mlp_layer_width,
            )
        self.field_heads = nn.ModuleList([field_head() for field_head in field_heads] if field_heads else [])  # type: ignore
        for field_head in self.field_heads:
            field_head.set_in_dim(self.mlp_head.get_out_dim())  # type ignore

    def forward(self, ray_bundle: RayBundle, depth: Float[Tensor, "*bs 1"]) -> Dict[FieldHeadNames, Tensor]:
        outputs = {}
        # Generate points from depth measurements
        point = ray_bundle.origins + ray_bundle.directions * depth
        w2cs = pose_utils.inverse(ray_bundle.metadata["camera_to_worlds"].view(ray_bundle.shape[0], 3, 4))
        if not self.training:
            print(f"w2cs: {w2cs.shape}")
        point_local = (torch.matmul(w2cs[..., :3, :3], point.unsqueeze(-1)) + w2cs[..., :3, 3:]).squeeze(-1)

        # Compute output from the query points.
        if self.spatial_distortion is not None:
            point_local = self.spatial_distortion(point_local)
        encoded_xyz = self.position_encoding(point_local)
        base_mlp_out = self.mlp_base(encoded_xyz)

        for field_head in self.field_heads:
            mlp_out = self.mlp_head(base_mlp_out)  # type: ignore
            outputs[field_head.field_head_name] = field_head(mlp_out)
        if not self.training:
            print(f"Done: {outputs[field_head.field_head_name].shape}")
        return outputs
