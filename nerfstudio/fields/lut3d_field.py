"""
Dr Song's look up table for light estimation.
"""

from typing import Dict, Literal, Optional, Tuple, Type

import torch
from jaxtyping import Float
from torch import Tensor, nn

import nerfstudio.utils.poses as pose_utils
from nerfstudio.cameras.rays import RayBundle, RaySamples
from nerfstudio.data.scene_box import SceneBox
from nerfstudio.field_components.encodings import Encoding, Identity, SHEncoding
from nerfstudio.field_components.field_heads import FieldHead, FieldHeadNames, RGBAffineFieldHead
from nerfstudio.field_components.mlp import MLP, MLPWithHashEncoding
from nerfstudio.field_components.spatial_distortions import SpatialDistortion
from nerfstudio.fields.base_field import get_normalized_directions


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
                out_activation=nn.ReLU(),
            )
        self.field_heads = nn.ModuleList([field_head() for field_head in field_heads] if field_heads else [])  # type: ignore
        for field_head in self.field_heads:
            field_head.set_in_dim(self.mlp_head.get_out_dim())  # type ignore

    def forward(self, ray_bundle: RayBundle, depth: Float[Tensor, "*bs 1"]) -> Dict[FieldHeadNames, Tensor]:
        outputs = {}
        # Generate points from depth measurements
        point = ray_bundle.origins + ray_bundle.directions * depth
        w2cs = pose_utils.inverse(ray_bundle.metadata["camera_to_worlds"].view(ray_bundle.shape[0], 3, 4))
        point_local = (torch.matmul(w2cs[..., :3, :3], point.unsqueeze(-1)) + w2cs[..., :3, 3:]).squeeze(-1)

        # Compute output from the query points.
        if self.spatial_distortion is not None:
            point_local = self.spatial_distortion(point_local)
            point_local = (point_local + 2.0) / 4.0
        encoded_xyz = self.position_encoding(point_local)

        base_mlp_out = self.mlp_base(encoded_xyz)

        for field_head in self.field_heads:
            mlp_out = self.mlp_head(base_mlp_out)  # type: ignore
            outputs[field_head.field_head_name] = field_head(mlp_out)
        return outputs


class LUT3DFieldHashEncoding(nn.Module):
    aabb: Tensor

    def __init__(
        self,
        aabb: Tensor,
        num_levels: int = 16,
        base_res: int = 16,
        max_res: int = 2048,
        log2_hashmap_size: int = 19,
        features_per_level: int = 2,
        num_layers: int = 2,
        hidden_dim: int = 64,
        geo_feat_dim: int = 16,
        spatial_distortion: Optional[SpatialDistortion] = None,
        implementation: Literal["tcnn", "torch"] = "tcnn",
    ) -> None:
        super().__init__()

        self.register_buffer("aabb", aabb)
        self.geo_feat_dim = geo_feat_dim

        self.register_buffer("max_res", torch.tensor(max_res))
        self.register_buffer("num_levels", torch.tensor(num_levels))
        self.register_buffer("log2_hashmap_size", torch.tensor(log2_hashmap_size))

        self.spatial_distortion = spatial_distortion

        self.direction_encoding = SHEncoding(levels=4, implementation=implementation)

        # MLP base
        self.mlp_base = MLPWithHashEncoding(
            num_levels=num_levels,
            min_res=base_res,
            max_res=max_res,
            log2_hashmap_size=log2_hashmap_size,
            features_per_level=features_per_level,
            num_layers=num_layers,
            layer_width=hidden_dim,
            out_dim=geo_feat_dim,
            activation=nn.ReLU(),
            out_activation=nn.ReLU(),
            implementation=implementation,
        )

        # MLP head
        self.mlp_head = MLP(
            in_dim=self.direction_encoding.get_out_dim() + self.geo_feat_dim,
            num_layers=3,
            layer_width=64,
            out_dim=3,
            activation=nn.ReLU(),
            out_activation=nn.Sigmoid(),
            implementation=implementation,
        )

    # def forward(self, ray_bundle: RayBundle, depth: Float[Tensor, "*bs 1"]) -> Dict[FieldHeadNames, Tensor]:
    #     outputs = {}
    #     # Generate points from depth measurements
    #     positions = ray_bundle.origins + ray_bundle.directions * depth
    #     w2cs = pose_utils.inverse(ray_bundle.metadata["camera_to_worlds"].view(ray_bundle.shape[0], 3, 4))
    #     positions = (torch.matmul(w2cs[..., :3, :3], positions.unsqueeze(-1)) + w2cs[..., :3, 3:]).squeeze(-1)

    #     if self.spatial_distortion is not None:
    #         positions = self.spatial_distortion(positions)
    #         positions = (positions + 2.0) / 4.0
    #     else:
    #         positions = SceneBox.get_normalized_positions(positions, self.aabb)
    #     # Make sure the tcnn gets inputs between 0 and 1.
    #     selector = ((positions > 0.0) & (positions < 1.0)).all(dim=-1)
    #     positions = positions * selector[..., None]

    #     base_mlp_out = self.mlp_base(positions)
    #     rgb = self.mlp_head(base_mlp_out).to(ray_bundle.directions)
    #     outputs.update({FieldHeadNames.RGB_AFFINE: rgb})
    #     return outputs

    def forward(
        self,
        ray_samples: RaySamples,
        densities: Float[Tensor, "*batch num_samples 1"],
        normals: Float[Tensor, "*batch 3"],
        camera_to_worlds: Float[Tensor, "*batch 3 4"],
    ) -> Dict[FieldHeadNames, Tensor]:
        outputs = {}
        # Get all sample positions.
        positions = ray_samples.frustums.get_positions()
        w2cs = pose_utils.inverse(camera_to_worlds).unsqueeze(1)
        # Convert all sample positions to the local camera coordinate space
        positions = (torch.matmul(w2cs[..., :3, :3], positions.unsqueeze(-1)) + w2cs[..., :3, 3:]).squeeze(-1)

        if self.spatial_distortion is not None:
            positions = self.spatial_distortion(positions)
            positions = (positions + 2.0) / 4.0
        else:
            positions = SceneBox.get_normalized_positions(positions, self.aabb)

        # Convert all normals to the local camera coordinate space
        w2cs = w2cs.squeeze(1)
        n_cam = get_normalized_directions((torch.matmul(w2cs[..., :3, :3], normals.unsqueeze(-1))).squeeze(-1))
        n_cam = n_cam.unsqueeze(1)
        n_cam = n_cam.repeat(1, ray_samples.shape[1], 1).view(-1, 3)
        d = self.direction_encoding(n_cam)

        # Make sure the tcnn gets inputs between 0 and 1.
        selector = ((positions > 0.0) & (positions < 1.0)).all(dim=-1)
        positions = positions * selector[..., None]
        positions_flat = positions.view(-1, 3)
        base_mlp_out = self.mlp_base(positions_flat)

        h = torch.cat([d, base_mlp_out], dim=-1)

        rgb = self.mlp_head(h).view(*ray_samples.frustums.shape, -1).to(densities)

        # Map densities to a normalized weight using a temperatured softmax operation.
        # weight = nn.functional.normalize(densities, dim=1, p=1)
        weight = torch.softmax(
            densities,
            dim=-2,
        )
        rgb = torch.sum(weight * rgb, dim=-2)

        outputs.update({FieldHeadNames.RGB_AFFINE: rgb})
        return outputs
