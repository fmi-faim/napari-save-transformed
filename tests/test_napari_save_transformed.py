import numpy as np
from napari.layers import Image  # type: ignore[import-untyped]
from tifffile.tifffile import imread

from napari_save_transformed import (
    TransformDict,
    write_transformed_layers,
)


def test_write_transformed_layers(tmp_path):
    layers = []
    shape = (100, 200)
    image0 = np.ones(shape=shape, dtype=np.uint16)
    layers.append(Image(image0))
    image1 = image0.copy()
    layers.append(
        Image(
            image1,
            affine=[[1, 0, 30], [0, 1, 10], [0, 0, 1]],
        )
    )
    out_path = tmp_path / "output.tif"
    layer_data = [layer.as_layer_data_tuple() for layer in layers]
    write_transformed_layers(path=out_path, layer_data=layer_data)

    assert out_path.exists()

    output = imread(out_path)
    assert output.dtype == np.uint16
    assert output.shape == (2, 130, 210)

    transform_file = out_path.parent / f"{out_path.stem}_transforms.yml"
    assert transform_file.exists()

    mapping = TransformDict.from_file(transform_file)
    assert np.array_equal(
        mapping["image0"],
        [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
    )
    assert np.array_equal(
        mapping["image1"],
        [[1, 0, 30], [0, 1, 10], [0, 0, 1]],
    )
