from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from napari.utils.transforms import Affine
from scipy.ndimage import affine_transform
from skimage.transform import matrix_transform
from tifffile.tifffile import imwrite

if TYPE_CHECKING:
    from napari.types import FullLayerData


def write_transformed_layers(path: str, layer_data: list[FullLayerData]) -> list[str]:
    images = []
    transforms = []
    for data, attrs, _ in layer_data:
        images.append(data)
        transforms.append(Affine(affine_matrix=attrs["affine"]))
    results, out_shape = transform_arrays(images, transforms)
    imwrite(path, np.stack(results).astype(np.float32), imagej=True)
    return path


def transform_arrays(arrays, affines):
    corners = np.array([[0, 0, 1, 1], [0, 1, 0, 1]])

    all_corners = []
    for arr, aff in zip(arrays, affines):
        h, w = arr.shape
        corners_scaled = corners.T * [h, w]
        transformed_corners = matrix_transform(corners_scaled, aff)
        all_corners.append(transformed_corners.T)

    all_corners = np.array(all_corners)
    min_corner = np.floor(np.min(all_corners, axis=(0, 2))).astype(int)
    max_corner = np.ceil(np.max(all_corners, axis=(0, 2))).astype(int)

    output_shape = tuple(max_corner - min_corner)

    outputs = []
    translation = Affine(translate=-min_corner)
    for arr, aff in zip(arrays, affines):
        # Apply the transformation
        output = affine_transform(
            input=arr,
            matrix=translation.compose(aff).inverse,
            output_shape=output_shape,
        )
        outputs.append(output)

    return outputs, output_shape
