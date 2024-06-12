from typing import Any, Mapping, MutableMapping, Optional, Sequence

import numpy as np

# The following code for raw data processing comes from RawNeRF:
# https://github.com/google-research/multinerf/blob/main/internal/raw_utils.py


def linear_to_srgb(linear: np.ndarray, eps: Optional[float] = None) -> np.ndarray:
    """Assumes `linear` is in [0, 1], see https://en.wikipedia.org/wiki/SRGB."""
    if eps is None:
        eps = np.finfo(np.float32).eps
    srgb0 = 323 / 25 * linear
    srgb1 = (211 * np.maximum(eps, linear) ** (5 / 12) - 11) / 200
    return np.where(linear <= 0.0031308, srgb0, srgb1)


def srgb_to_linear(srgb: np.ndarray, eps: Optional[float] = None) -> np.ndarray:
    """Assumes `srgb` is in [0, 1], see https://en.wikipedia.org/wiki/SRGB."""
    if eps is None:
        eps = np.finfo(np.float32).eps
    linear0 = 25 / 323 * srgb
    linear1 = np.maximum(eps, ((200 * srgb + 11) / (211))) ** (12 / 5)
    return np.where(srgb <= 0.04045, linear0, linear1)


def postprocess_raw(
    raw: np.ndarray,
    camtorgb: np.ndarray,
    exposure: Optional[float] = None,
) -> np.ndarray:
    """Converts demosaicked raw to sRGB with a minimal postprocessing pipeline.

    Numpy array inputs will be automatically converted to Jax arrays.

    Args:
      raw: [H, W, 3], demosaicked raw camera image.
      camtorgb: [3, 3], color correction transformation to apply to raw image.
      exposure: color value to be scaled to pure white after color correction.
                If None, "autoexposes" at the 97th percentile.
      xnp: either numpy or jax.numpy.

    Returns:
      srgb: [H, W, 3], color corrected + exposed + gamma mapped image.
    """
    if raw.shape[-1] != 3:
        raise ValueError(f"raw.shape[-1] is {raw.shape[-1]}, expected 3")
    if camtorgb.shape != (3, 3):
        raise ValueError(f"camtorgb.shape is {camtorgb.shape}, expected (3, 3)")
    # Convert from camera color space to standard linear RGB color space.
    rgb_linear = np.matmul(raw, camtorgb.T)
    if exposure is None:
        exposure = np.percentile(rgb_linear, 100)
    # "Expose" image by mapping the input exposure level to white and clipping.
    rgb_linear_scaled = np.clip(rgb_linear / exposure, 0, 1)
    # Apply sRGB gamma curve to serve as a simple tonemap.
    srgb = linear_to_srgb(rgb_linear_scaled)
    return srgb


def bilinear_demosaic(bayer: np.ndarray) -> np.ndarray:
    """Converts Bayer data into a full RGB image using bilinear demosaicking.

    Input data should be ndarray of shape [height, width] with 2x2 mosaic pattern:
      -------------
      |red  |green|
      -------------
      |green|blue |
      -------------
    Red and blue channels are bilinearly upsampled 2x, missing green channel
    elements are the average of the neighboring 4 values in a cross pattern.

    Args:
      bayer: [H, W] array, Bayer mosaic pattern input image.
      xnp: either numpy or jax.numpy.

    Returns:
      rgb: [H, W, 3] array, full RGB image.
    """

    def reshape_quads(*planes):
        """Reshape pixels from four input images to make tiled 2x2 quads."""
        planes = np.stack(planes, -1)
        shape = planes.shape[:-1]
        # Create [2, 2] arrays out of 4 channels.
        zup = planes.reshape(
            shape
            + (
                2,
                2,
            )
        )
        # Transpose so that x-axis dimensions come before y-axis dimensions.
        zup = np.transpose(zup, (0, 2, 1, 3))
        # Reshape to 2D.
        zup = zup.reshape((shape[0] * 2, shape[1] * 2))
        return zup

    def bilinear_upsample(z):
        """2x bilinear image upsample."""
        # Using np.roll makes the right and bottom edges wrap around. The raw image
        # data has a few garbage columns/rows at the edges that must be discarded
        # anyway, so this does not matter in practice.
        # Horizontally interpolated values.
        zx = 0.5 * (z + np.roll(z, -1, axis=-1))
        # Vertically interpolated values.
        zy = 0.5 * (z + np.roll(z, -1, axis=-2))
        # Diagonally interpolated values.
        zxy = 0.5 * (zx + np.roll(zx, -1, axis=-2))
        return reshape_quads(z, zx, zy, zxy)

    def upsample_green(g1, g2):
        """Special 2x upsample from the two green channels."""
        z = np.zeros_like(g1)
        z = reshape_quads(z, g1, g2, z)
        alt = 0
        # Grab the 4 directly adjacent neighbors in a "cross" pattern.
        for i in range(4):
            axis = -1 - (i // 2)
            roll = -1 + 2 * (i % 2)
            alt = alt + 0.25 * np.roll(z, roll, axis=axis)
        # For observed pixels, alt = 0, and for unobserved pixels, alt = avg(cross),
        # so alt + z will have every pixel filled in.
        return alt + z

    r, g1, g2, b = [bayer[(i // 2) :: 2, (i % 2) :: 2] for i in range(4)]
    r = bilinear_upsample(r)
    # Flip in x and y before and after calling upsample, as bilinear_upsample
    # assumes that the samples are at the top-left corner of the 2x2 sample.
    b = bilinear_upsample(b[::-1, ::-1])[::-1, ::-1]
    g = upsample_green(g1, g2)
    rgb = np.stack([r, g, b], -1)
    return rgb


# Relevant fields to extract from raw image EXIF metadata.
# For details regarding EXIF parameters, see:
# https://www.adobe.com/content/dam/acom/en/products/photoshop/pdfs/dng_spec_1.4.0.0.pdf.
_EXIF_KEYS = (
    "BlackLevel",  # Black level offset added to sensor measurements.
    "WhiteLevel",  # Maximum possible sensor measurement.
    "AsShotNeutral",  # RGB white balance coefficients.
    "ColorMatrix2",  # XYZ to camera color space conversion matrix.
    "NoiseProfile",  # Shot and read noise levels.
)

# Color conversion from reference illuminant XYZ to RGB color space.
# See http://www.brucelindbloom.com/index.html?Eqn_RGB_XYZ_Matrix.html.
_RGB2XYZ = np.array(
    [
        [0.4124564, 0.3575761, 0.1804375],
        [0.2126729, 0.7151522, 0.0721750],
        [0.0193339, 0.1191920, 0.9503041],
    ]
)


def process_exif(exifs: Sequence[Mapping[str, Any]]) -> MutableMapping[str, Any]:
    """Processes list of raw image EXIF data into useful metadata dict.

    Input should be a list of dictionaries loaded from JSON files.
    These JSON files are produced by running
      $ exiftool -json IMAGE.dng > IMAGE.json
    for each input raw file.

    We extract only the parameters relevant to
    1. Rescaling the raw data to [0, 1],
    2. White balance and color correction, and
    3. Noise level estimation.

    Args:
      exifs: a list of dicts containing EXIF data as loaded from JSON files.

    Returns:
      meta: a dict of the relevant metadata for running RawNeRF.
    """
    meta = {}
    exif = exifs[0]
    # Convert from array of dicts (exifs) to dict of arrays (meta).
    for key in _EXIF_KEYS:
        exif_value = exif.get(key)
        if exif_value is None:
            continue
        # Values can be a single int or float...
        if isinstance(exif_value, (float, int)):
            vals = [x[key] for x in exifs]
        # Or a string of numbers with ' ' between.
        elif isinstance(exif_value, str):
            vals = [[float(z) for z in x[key].split(" ")] for x in exifs]
        meta[key] = np.squeeze(np.array(vals))
    # Shutter speed is a special case, a string written like 1/N.
    meta["ShutterSpeed"] = np.fromiter((1.0 / float(exif["ShutterSpeed"].split("/")[1]) for exif in exifs), float)

    # Create raw-to-sRGB color transform matrices. Pipeline is:
    # cam space -> white balanced cam space ("camwb") -> XYZ space -> RGB space.
    # 'AsShotNeutral' is an RGB triplet representing how pure white would measure
    # on the sensor, so dividing by these numbers corrects the white balance.
    whitebalance = meta["AsShotNeutral"].reshape(-1, 3)
    cam2camwb = np.array([np.diag(1.0 / x) for x in whitebalance])
    # ColorMatrix2 converts from XYZ color space to "reference illuminant" (white
    # balanced) camera space.
    xyz2camwb = meta["ColorMatrix2"].reshape(-1, 3, 3)
    rgb2camwb = xyz2camwb @ _RGB2XYZ
    # We normalize the rows of the full color correction matrix, as is done in
    # https://github.com/AbdoKamel/simple-camera-pipeline.
    rgb2camwb /= rgb2camwb.sum(axis=-1, keepdims=True)
    # Combining color correction with white balance gives the entire transform.
    cam2rgb = np.linalg.inv(rgb2camwb) @ cam2camwb
    meta["cam2rgb"] = cam2rgb

    return meta


def downsample(img, factor):
    """Area downsample img (factor must evenly divide img height and width)."""
    sh = img.shape
    if not (sh[0] % factor == 0 and sh[1] % factor == 0):
        raise ValueError(f"Downsampling factor {factor} does not " f"evenly divide image shape {sh[:2]}")
    img = img.reshape((sh[0] // factor, factor, sh[1] // factor, factor) + sh[2:])
    img = img.mean((1, 3))
    return img
