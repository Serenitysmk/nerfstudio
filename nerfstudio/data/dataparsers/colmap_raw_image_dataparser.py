"""Data parser for COLMAP with raw image data."""

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Type

import numpy as np
import rawpy
import torch

from nerfstudio.cameras import camera_utils
from nerfstudio.cameras.cameras import CAMERA_MODEL_TO_TYPE, Cameras
from nerfstudio.data.dataparsers.base_dataparser import DataparserOutputs
from nerfstudio.data.dataparsers.colmap_dataparser import ColmapDataParser, ColmapDataParserConfig
from nerfstudio.data.scene_box import SceneBox
from nerfstudio.utils import raw_utils


@dataclass
class ColmapRawImageDataParserConfig(ColmapDataParserConfig):
    _target: Type = field(default_factory=lambda: ColmapRawImageDataParser)
    images_path: Path = Path("srgb")
    raw_path: Path = Path("raw")


class ColmapRawImageDataParser(ColmapDataParser):
    """COLMAP with raw images DatasetParser."""

    config: ColmapRawImageDataParserConfig

    def __init__(self, config: ColmapRawImageDataParserConfig):
        super().__init__(config)
        self.config = config
        self._downscale_factor = None

    def _generate_dataparser_outputs(self, split: str = "train", **kwargs):
        assert self.config.data.exists(), f"Data directory {self.config.data} does not exist."
        colmap_path = self.config.data / self.config.colmap_path
        assert colmap_path.exists(), f"Colmap path {colmap_path} does not exist."

        meta = self._get_all_images_and_cameras(colmap_path)
        camera_type = CAMERA_MODEL_TO_TYPE[meta["camera_model"]]

        image_filenames = []
        mask_filenames = []
        depth_filenames = []
        poses = []

        fx = []
        fy = []
        cx = []
        cy = []
        height = []
        width = []
        distort = []

        for frame in meta["frames"]:
            fx.append(float(frame["fl_x"]))
            fy.append(float(frame["fl_y"]))
            cx.append(float(frame["cx"]))
            cy.append(float(frame["cy"]))
            height.append(int(frame["h"]))
            width.append(int(frame["w"]))
            distort.append(
                camera_utils.get_distortion_params(
                    k1=float(frame["k1"]) if "k1" in frame else 0.0,
                    k2=float(frame["k2"]) if "k2" in frame else 0.0,
                    k3=float(frame["k3"]) if "k3" in frame else 0.0,
                    k4=float(frame["k4"]) if "k4" in frame else 0.0,
                    p1=float(frame["p1"]) if "p1" in frame else 0.0,
                    p2=float(frame["p2"]) if "p2" in frame else 0.0,
                )
            )
            fname = Path(frame["file_path"]).with_suffix(".DNG").name
            image_filenames.append(self.config.data / self.config.raw_path / fname)
            poses.append(frame["transform_matrix"])
            if "mask_path" in frame:
                mask_filenames.append(Path(frame["mask_path"]))
            if "depth_path" in frame:
                depth_filenames.append(Path(frame["depth_path"]))

        assert len(mask_filenames) == 0 or (len(mask_filenames) == len(image_filenames)), """
        Different number of image and mask filenames.
        You should check that mask_path is specified for every frame (or zero frames) in transforms.json.
        """
        assert len(depth_filenames) == 0 or (len(depth_filenames) == len(image_filenames)), """
        Different number of image and depth filenames.
        You should check that depth_file_path is specified for every frame (or zero frames) in transforms.json.
        """
        poses = torch.from_numpy(np.array(poses).astype(np.float32))
        poses, transform_matrix = camera_utils.auto_orient_and_center_poses(
            poses,
            method=self.config.orientation_method,
            center_method=self.config.center_method,
        )

        # Scale poses
        scale_factor = 1.0
        if self.config.auto_scale_poses:
            scale_factor /= float(torch.max(torch.abs(poses[:, :3, 3])))
        scale_factor *= self.config.scale_factor
        poses[:, :3, 3] *= scale_factor

        exif_filenames = [path.with_suffix(".json") for path in image_filenames]

        # Choose image_filenames and poses based on split, but after auto orient and scaling the poses.
        indices = self._get_image_indices(image_filenames, split)

        image_filenames = [image_filenames[i] for i in indices]
        mask_filenames = [mask_filenames[i] for i in indices] if len(mask_filenames) > 0 else []
        depth_filenames = [depth_filenames[i] for i in indices] if len(depth_filenames) > 0 else []

        idx_tensor = torch.tensor(indices, dtype=torch.long)
        poses = poses[idx_tensor]

        # in x,y,z order
        # assumes that the scene is centered at the origin
        aabb_scale = self.config.scene_scale
        scene_box = SceneBox(
            aabb=torch.tensor(
                [[-aabb_scale, -aabb_scale, -aabb_scale], [aabb_scale, aabb_scale, aabb_scale]], dtype=torch.float32
            )
        )

        fx = torch.tensor(fx, dtype=torch.float32)[idx_tensor]
        fy = torch.tensor(fy, dtype=torch.float32)[idx_tensor]
        cx = torch.tensor(cx, dtype=torch.float32)[idx_tensor]
        cy = torch.tensor(cy, dtype=torch.float32)[idx_tensor]
        height = torch.tensor(height, dtype=torch.int32)[idx_tensor]
        width = torch.tensor(width, dtype=torch.int32)[idx_tensor]
        distortion_params = torch.stack(distort, dim=0)[idx_tensor]

        # Read camera metadata
        def read_exif(p):
            with open(p.as_posix(), "rb") as f:
                exif = json.load(f)[0]
                return exif

        meta = raw_utils.process_exif([read_exif(p) for p in exif_filenames])

        exposure = meta["ShutterSpeed"]
        tgt_exposure = np.mean(exposure)
        exposure_scale = tgt_exposure / exposure
        black_level = meta["BlackLevel"] if meta["BlackLevel"].ndim == 1 else np.mean(meta["BlackLevel"], axis=-1)
        white_level = meta["WhiteLevel"]
        cam2rgb = meta["cam2rgb"]

        # Read and process one raw image to determine the fixed exposure value
        with open(image_filenames[0].as_posix(), "rb") as f:
            raw0 = rawpy.imread(f).raw_image
            raw0 = raw0.astype(np.float32)
            im0 = (raw0 - black_level[0]) / (white_level[0] - black_level[0])
            im0 = raw_utils.bilinear_demosaic(im0)
            im0_linear = np.matmul(im0, cam2rgb[0].T)
            exposure = np.percentile(im0_linear, 97)

        cam_meta = {
            "black_level": torch.from_numpy(black_level)[idx_tensor].unsqueeze(-1),
            "white_level": torch.from_numpy(white_level)[idx_tensor].unsqueeze(-1),
            "exposure_scale": torch.from_numpy(exposure_scale)[idx_tensor].unsqueeze(-1),
            "cam2rgb": torch.from_numpy(cam2rgb)[idx_tensor].reshape(-1, 9),
            "exposure": torch.tensor([exposure]),
        }

        cameras = Cameras(
            fx=fx,
            fy=fy,
            cx=cx,
            cy=cy,
            distortion_params=distortion_params,
            height=height,
            width=width,
            camera_to_worlds=poses[:, :3, :4],
            camera_type=camera_type,
            metadata=cam_meta,
        )

        cameras.rescale_output_resolution(
            scaling_factor=1.0 / int(self.config.downscale_factor),
            scale_rounding_mode=self.config.downscale_rounding_mode,
        )

        if "applied_transform" in meta:
            applied_transform = torch.tensor(meta["applied_transform"], dtype=transform_matrix.dtype)
            transform_matrix = transform_matrix @ torch.cat(
                [applied_transform, torch.tensor([[0, 0, 0, 1]], dtype=transform_matrix.dtype)], 0
            )
        if "applied_scale" in meta:
            applied_scale = float(meta["applied_scale"])
            scale_factor *= applied_scale

        metadata = {}
        if self.config.load_3D_points:
            # Load 3D points
            metadata.update(self._load_3D_points(colmap_path, transform_matrix, scale_factor))

        dataparser_outputs = DataparserOutputs(
            image_filenames=image_filenames,
            cameras=cameras,
            scene_box=scene_box,
            mask_filenames=mask_filenames if len(mask_filenames) > 0 else None,
            dataparser_scale=scale_factor,
            dataparser_transform=transform_matrix,
            metadata={
                "depth_filenames": depth_filenames if len(depth_filenames) > 0 else None,
                "depth_unit_scale_factor": self.config.depth_unit_scale_factor,
                **metadata,
            },
        )
        return dataparser_outputs
