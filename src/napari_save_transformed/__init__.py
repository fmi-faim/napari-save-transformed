from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import yaml
from napari.utils.transforms import Affine  # type: ignore[import-untyped]
from pydantic import Field, RootModel
from scipy.ndimage import affine_transform
from skimage.transform import matrix_transform
from tifffile.tifffile import imwrite

if TYPE_CHECKING:
    from napari.types import FullLayerData  # type: ignore[import-untyped]


class TransformDict(RootModel):
    root: dict[str, np.ndarray] = Field(default_factory=dict)

    model_config = {"arbitrary_types_allowed": True}  # noqa: RUF012

    def __getitem__(self, key):
        return self.root[key]

    def __setitem__(self, key, value):
        self.root[key] = value

    @classmethod
    def from_yaml(cls, yaml_str: str) -> TransformDict:
        loaded_data = yaml.safe_load(yaml_str)
        return cls({k: np.array(v).reshape(3, 3) for k, v in loaded_data.items()})

    def to_yaml(self) -> str:
        return yaml.safe_dump(self.model_dump(), default_flow_style=None)

    @classmethod
    def from_file(cls, file_path: str | Path) -> TransformDict:
        with open(file_path) as file:
            return cls.from_yaml(file.read())

    def to_file(self, file_path: str | Path) -> None:
        with open(file_path, "w") as file:
            file.write(self.to_yaml())


def write_transformed_layers(path: str, layer_data: list[FullLayerData]) -> list[str]:
    images = []
    transforms = []
    mapping = TransformDict()
    for data, attrs, _ in layer_data:
        images.append(data)
        mapping[attrs["name"]] = attrs["affine"].tolist()
        transforms.append(Affine(affine_matrix=attrs["affine"]))
    results, out_shape = transform_arrays(images, transforms)
    imwrite(path, np.stack(results), imagej=True)
    out_path = Path(path)
    mapping_file = out_path.parent / f"{out_path.stem}_transforms.yml"
    mapping.to_file(mapping_file)
    return [path]


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
